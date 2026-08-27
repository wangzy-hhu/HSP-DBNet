import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .faster_rcnn import FasterRCNN
from ..builder import DETECTORS


@DETECTORS.register_module()
class MultiModalRCNN(FasterRCNN):
    """Faster R-CNN variant for RGB + normal-map inputs.

    The visualization path builds one detector-level CAM from the actual
    detection graph.  Final RoI classification is the main target, with a
    small RPN objectness term so the map can show proposal/context evidence
    instead of being limited to final boxes only.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(MultiModalRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      normal_map,
                      freq_prior,
                      env_prior,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img, normal_map)
        losses = dict()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
            gt_masks,
            freq_prior=freq_prior,
            env_prior=env_prior,
            **kwargs)
        losses.update(roi_losses)
        return losses

    def extract_feat(self, img, normal_map):
        x = self.backbone(img, normal_map)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing FLOPs with RGB + normal-map inputs."""
        outs = ()
        normal_map = torch.zeros_like(img)

        x = self.extract_feat(img, normal_map)
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)

        proposals = torch.randn(1000, 4).to(img.device)
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def simple_test(self,
                    img,
                    img_metas,
                    normal_map,
                    freq_prior,
                    env_prior,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        if isinstance(normal_map, list):
            normal_map = normal_map[0]
        if isinstance(freq_prior, list):
            freq_prior = freq_prior[0]
        if isinstance(env_prior, list):
            env_prior = env_prior[0]

        if self._gradcam_enabled():
            with torch.enable_grad():
                x = self.extract_feat(img, normal_map)
                with torch.no_grad():
                    if proposals is None:
                        proposal_list = self._simple_test_rpn(x, img_metas)
                    else:
                        proposal_list = proposals
                    proposal_list = self._detach_proposals(proposal_list)
                    results = self.roi_head.simple_test(
                        x,
                        proposal_list,
                        img_metas,
                        rescale=rescale,
                        freq_prior=freq_prior,
                        env_prior=env_prior)
                self._save_gradcam_maps(
                    img,
                    img_metas,
                    x,
                    results,
                    rescale=rescale)
                return results

        x = self.extract_feat(img, normal_map)
        if proposals is None:
            proposal_list = self._simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        proposal_list = self._detach_proposals(proposal_list)

        return self.roi_head.simple_test(
            x,
            proposal_list,
            img_metas,
            rescale=rescale,
            freq_prior=freq_prior,
            env_prior=env_prior)

    def aug_test(self, imgs, img_metas, rescale=False, **kwargs):
        normal_maps = kwargs['normal_map']
        freq_priors = kwargs['freq_prior']
        env_priors = kwargs['env_prior']

        feats = []
        for img, normal_map in zip(imgs, normal_maps):
            feats.append(self.extract_feat(img, normal_map))

        proposal_list = self.rpn_head.aug_test_rpn(feats, img_metas)
        return self.roi_head.aug_test(
            feats,
            proposal_list,
            img_metas,
            rescale=rescale,
            freq_prior=freq_priors,
            env_prior=env_priors)

    def _simple_test_rpn(self, x, img_metas):
        rpn_outs = self.rpn_head(x)
        try:
            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas)
            if isinstance(proposal_list, tuple):
                proposal_list = proposal_list[0]
        except TypeError:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        return proposal_list

    def _detach_proposals(self, proposal_list):
        if proposal_list is None:
            return proposal_list
        return [proposal.detach() for proposal in proposal_list]

    def _gradcam_enabled(self):
        return bool(getattr(self.backbone, 'vis_enable', False)) and \
            self._cam_method() in ('gradcam', 'grad_cam')

    def _cam_method(self):
        vis_cfg = getattr(self.backbone, 'vis_cfg', {})
        return str(vis_cfg.get('cam_method', 'gradcam')).lower()

    def _vis_value(self, name, default=None):
        return getattr(self.backbone, 'vis_cfg', {}).get(name, default)

    def _full_gradcam_maps(self,
                           feats,
                           det_results,
                           img_metas,
                           rescale=False):
        batch_size = len(img_metas)
        target_terms = []

        det_weight = float(self._vis_value('gradcam_detection_target_weight',
                                           0.05))
        if det_weight > 0:
            det_target = self._gradcam_detection_target(
                feats, det_results, img_metas, rescale=rescale)
            if det_target is not None:
                target_terms.append(det_target * det_weight)

        rpn_weight = float(self._vis_value('gradcam_rpn_target_weight', 1.0))
        if rpn_weight > 0 and self.with_rpn:
            rpn_target = self._gradcam_rpn_target(feats, batch_size)
            if rpn_target is not None:
                target_terms.append(rpn_target * rpn_weight)

        if not target_terms:
            return [None for _ in range(batch_size)]
        target = torch.stack(target_terms).sum()
        if not target.requires_grad or not torch.isfinite(target).item():
            return [None for _ in range(batch_size)]

        grad_levels = [
            (level, feat) for level, feat in enumerate(feats)
            if torch.is_tensor(feat) and feat.requires_grad
        ]
        if not grad_levels:
            return [None for _ in range(batch_size)]

        grad_inputs = tuple(feat for _, feat in grad_levels)
        grads = torch.autograd.grad(
            target,
            grad_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True)
        return self._project_feature_gradcams(
            grad_levels,
            grads,
            img_metas,
            num_levels=len(feats))

    def _gradcam_detection_target(self,
                                  feats,
                                  det_results,
                                  img_metas,
                                  rescale=False):
        terms = self._gradcam_det_box_terms(
            feats, det_results, img_metas, rescale=rescale)
        if not terms:
            return None
        values = [
            term['target'] for term in terms
            if term['target'].requires_grad and
               torch.isfinite(term['target']).item()
        ]
        if not values:
            return None
        return torch.stack(values).mean()

    def _gradcam_det_box_terms(self,
                               feats,
                               det_results,
                               img_metas,
                               rescale=False):
        if det_results is None or not feats:
            return []

        device = feats[0].device
        dtype = feats[0].dtype
        roi_rows = []
        specs = []
        for img_idx, img_meta in enumerate(img_metas):
            targets = self._collect_gradcam_det_targets(
                self._get_single_det_result(det_results, img_idx),
                img_meta,
                rescale=rescale)
            max_targets = int(self._vis_value('gradcam_det_max_targets', 30))
            if max_targets > 0:
                targets = targets[:max_targets]
            img_h, img_w = img_meta.get('img_shape')[:2]
            for score, cls_id, coords in targets:
                x1, y1, x2, y2 = coords.astype(np.float32)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                roi_rows.append([img_idx, x1, y1, x2, y2])
                specs.append(dict(
                    img_idx=img_idx,
                    score=float(score),
                    cls_id=int(cls_id),
                    box=np.array([
                        np.clip(x1, 0, img_w - 1),
                        np.clip(y1, 0, img_h - 1),
                        np.clip(x2, 0, img_w - 1),
                        np.clip(y2, 0, img_h - 1)
                    ], dtype=np.float32)))

        if not roi_rows:
            return []

        rois = torch.tensor(roi_rows, device=device, dtype=dtype)
        bbox_results = self.roi_head._bbox_forward(feats, rois)
        cls_score = bbox_results.get('cls_score', None)
        if cls_score is None or cls_score.numel() == 0:
            return []

        target_matrix = self._rcnn_score_matrix(
            cls_score,
            use_logits=bool(self._vis_value('gradcam_use_logits', True)))
        if target_matrix is None or target_matrix.numel() == 0:
            return []

        terms = []
        for roi_idx, spec in enumerate(specs):
            cls_id = int(spec['cls_id'])
            if cls_id >= target_matrix.size(1):
                continue
            score_weight = float(np.sqrt(np.clip(spec['score'], 0.0, 1.0)))
            terms.append(dict(
                target=target_matrix[roi_idx, cls_id] * score_weight,
                img_idx=int(spec['img_idx']),
                score=float(spec['score']),
                cls_id=cls_id,
                box=spec['box']))
        return terms

    def _rcnn_score_matrix(self, cls_score, use_logits=True):
        num_classes = int(getattr(
            self.roi_head.bbox_head, 'num_classes', cls_score.size(1) - 1))
        if bool(use_logits):
            return cls_score[:, :num_classes] \
                if cls_score.size(1) > num_classes else cls_score

        scores = F.softmax(cls_score, dim=1)
        return scores[:, :num_classes] \
            if scores.size(1) > num_classes else scores

    def _gradcam_rpn_target(self, feats, batch_size):
        try:
            rpn_outs = self.rpn_head(feats)
        except Exception:
            return None
        if not isinstance(rpn_outs, (list, tuple)) or len(rpn_outs) == 0:
            return None

        cls_scores = rpn_outs[0]
        iou_preds = rpn_outs[2] if len(rpn_outs) >= 3 else None
        if not isinstance(cls_scores, (list, tuple)):
            return None

        target_mode = str(self._vis_value(
            'gradcam_rpn_target_mode', 'soft_dense')).lower()
        topk = int(self._vis_value('gradcam_rpn_topk', 0))
        use_sigmoid = bool(getattr(self.rpn_head, 'use_sigmoid_cls', True))
        cls_out_channels = int(getattr(self.rpn_head, 'cls_out_channels', 1))
        score_power = float(self._vis_value('gradcam_rpn_score_power', 0.7))
        score_floor = float(self._vis_value('gradcam_rpn_score_floor', 0.01))
        score_power = max(score_power, 1e-6)
        score_floor = float(np.clip(score_floor, 0.0, 0.95))
        terms = []

        for img_idx in range(batch_size):
            image_rank_scores = []
            image_target_values = []
            for level, cls_score in enumerate(cls_scores):
                if img_idx >= cls_score.size(0):
                    continue
                logits = cls_score[img_idx].float()
                height, width = logits.shape[-2:]
                if cls_out_channels > 1 and \
                        logits.size(0) % cls_out_channels == 0:
                    logits = logits.view(
                        -1, cls_out_channels, height, width)
                    if use_sigmoid:
                        rank_score, cls_inds = logits.sigmoid().max(dim=1)
                        target_value = logits.gather(
                            1, cls_inds[:, None]).squeeze(1)
                    else:
                        prob = logits.softmax(dim=1)
                        rank_score, cls_inds = prob[:, 1:].max(dim=1)
                        cls_inds = cls_inds + 1
                        target_value = logits.gather(
                            1, cls_inds[:, None]).squeeze(1)
                else:
                    rank_score = logits.sigmoid()
                    target_value = logits

                rank_score = rank_score.reshape(-1)
                target_value = target_value.reshape(-1)
                if iou_preds is not None and level < len(iou_preds):
                    iou_logits = iou_preds[level][img_idx].float().reshape(-1)
                    if iou_logits.numel() == rank_score.numel():
                        iou_score = iou_logits.sigmoid()
                        rank_score = torch.sqrt(
                            torch.clamp(rank_score * iou_score,
                                        min=0.0) + 1e-6)
                        target_value = target_value + 0.5 * iou_logits
                image_rank_scores.append(rank_score.detach())
                image_target_values.append(target_value)

            if not image_rank_scores:
                continue
            rank_scores = torch.cat(image_rank_scores)
            target_values = torch.cat(image_target_values)
            valid = torch.isfinite(rank_scores) & torch.isfinite(target_values)
            if not valid.any():
                continue
            rank_scores = rank_scores[valid]
            target_values = target_values[valid]
            if topk > 0 and topk < rank_scores.numel():
                top_inds = rank_scores.topk(topk).indices
                rank_scores = rank_scores[top_inds]
                target_values = target_values[top_inds]

            if target_mode in ('topk', 'hard_topk', 'hard'):
                terms.append(target_values.mean())
                continue

            weights = torch.clamp(rank_scores, min=score_floor)
            weights = torch.pow(weights, score_power)
            weights = weights / weights.sum().clamp(min=1e-6)
            terms.append((target_values * weights).sum())

        if not terms:
            return None
        return torch.stack(terms).mean()

    def _collect_gradcam_det_targets(self,
                                     det_result,
                                     img_meta,
                                     rescale=False,
                                     score_thr=None):
        if det_result is None:
            return []

        if score_thr is None:
            score_thr = self._vis_value(
                'gradcam_det_score_thr',
                self._vis_value('draw_score_thr', 0.30))
        score_thr = float(score_thr)
        scale_factor = img_meta.get('scale_factor', None)
        if scale_factor is not None:
            scale_factor = np.array(scale_factor, dtype=np.float32)
        img_h, img_w = img_meta.get('img_shape')[:2]

        targets = []
        for cls_id, cls_bboxes in enumerate(det_result):
            if cls_bboxes is None or len(cls_bboxes) == 0:
                continue
            if torch.is_tensor(cls_bboxes):
                cls_bboxes = cls_bboxes.detach().cpu().numpy()
            for bbox in cls_bboxes:
                if len(bbox) < 5 or bbox[4] < score_thr:
                    continue
                coords = bbox[:4].astype(np.float32).copy()
                if rescale and scale_factor is not None:
                    coords = coords * scale_factor
                coords[[0, 2]] = np.clip(coords[[0, 2]], 0, img_w - 1)
                coords[[1, 3]] = np.clip(coords[[1, 3]], 0, img_h - 1)
                if coords[2] <= coords[0] or coords[3] <= coords[1]:
                    continue
                targets.append((float(bbox[4]), int(cls_id), coords))

        targets.sort(key=lambda item: item[0], reverse=True)
        return targets

    def _project_feature_gradcams(self,
                                  grad_levels,
                                  grads,
                                  img_metas,
                                  num_levels=None):
        batch_size = len(img_metas)
        full_cams = [
            np.zeros(img_meta.get('img_shape')[:2], dtype=np.float32)
            for img_meta in img_metas
        ]
        keep_levels = self._gradcam_feature_levels(num_levels or len(grad_levels))
        response_mode = str(self._vis_value(
            'gradcam_response', 'sensitivity')).lower()
        aggregate = str(self._vis_value(
            'gradcam_feature_aggregate', 'sum')).lower()
        normalize_each_level = bool(self._vis_value(
            'gradcam_normalize_each_level', True))

        for level_info, grad in zip(grad_levels, grads):
            level, feat = level_info
            if grad is None or (keep_levels is not None and level not in keep_levels):
                continue

            act = feat.detach().float()
            grad = grad.detach().float()
            if response_mode in ('positive', 'standard', 'gradcam'):
                weights = grad.mean(dim=(2, 3), keepdim=True)
                raw_cams = F.relu((weights * act).sum(dim=1))
            elif response_mode in ('hires', 'hirescam', 'elementwise'):
                raw_cams = (grad * act).abs().mean(dim=1)
            else:
                raw_cams = (grad * act).abs().mean(dim=1)

            cams = raw_cams.detach().cpu().numpy()
            for img_idx in range(min(batch_size, cams.shape[0])):
                img_h, img_w = img_metas[img_idx].get('img_shape')[:2]
                level_cam = self._crop_feature_cam_to_img(
                    cams[img_idx], img_metas[img_idx])
                level_smooth = float(self._vis_value(
                    'gradcam_level_smooth_ratio', 0.0))
                if level_smooth > 0:
                    sigma = max(0.0, float(level_smooth))
                    level_cam = cv2.GaussianBlur(
                        level_cam.astype(np.float32),
                        (0, 0),
                        sigmaX=sigma,
                        sigmaY=sigma,
                        borderType=cv2.BORDER_REPLICATE)
                level_cam = cv2.resize(
                    level_cam,
                    (img_w, img_h),
                    interpolation=cv2.INTER_LINEAR)
                level_cam = np.clip(level_cam.astype(np.float32), 0.0, None)
                if level_cam.max() <= 1e-8:
                    continue
                if normalize_each_level:
                    level_cam = self._percentile_normalize(
                        level_cam,
                        self._vis_value('gradcam_level_low_percentile', 0.0),
                        self._vis_value('gradcam_level_high_percentile', 99.5))
                if aggregate == 'max':
                    np.maximum(full_cams[img_idx], level_cam, out=full_cams[img_idx])
                else:
                    full_cams[img_idx] += level_cam

        smooth_ratio = float(self._vis_value('gradcam_smooth_ratio', 0.035))
        blur_ratio = float(self._vis_value('gradcam_blur_ratio', 0.0))
        gamma = float(self._vis_value('gradcam_gamma', 0.75))
        low_p = self._vis_value('gradcam_low_percentile', 0.0)
        high_p = self._vis_value('gradcam_high_percentile', 99.0)
        out = []
        for cam in full_cams:
            if cam.max() <= 1e-8:
                out.append(None)
                continue
            raw_cam = self._normalize_cam(cam.copy())
            display = cam.astype(np.float32)
            for ratio in (smooth_ratio, blur_ratio):
                if ratio <= 0:
                    continue
                sigma = max(0.0, min(display.shape[:2]) * ratio)
                if sigma > 0:
                    display = cv2.GaussianBlur(
                        display,
                        (0, 0),
                        sigmaX=sigma,
                        sigmaY=sigma,
                        borderType=cv2.BORDER_REPLICATE)
            display = self._percentile_normalize(display, low_p, high_p)
            display = np.power(self._normalize_cam(display), max(gamma, 1e-6))
            out.append(dict(raw=raw_cam, display=self._normalize_cam(display)))
        return out

    def _crop_feature_cam_to_img(self, cam, img_meta):
        cam = cam.astype(np.float32)
        height, width = cam.shape[:2]
        img_h, img_w = img_meta.get('img_shape')[:2]
        pad_shape = img_meta.get('pad_shape', None)
        if pad_shape is None:
            return cam
        pad_h, pad_w = pad_shape[:2]
        if pad_h <= 0 or pad_w <= 0:
            return cam

        valid_h = int(round(height * float(img_h) / float(pad_h)))
        valid_w = int(round(width * float(img_w) / float(pad_w)))
        valid_h = int(np.clip(valid_h, 1, height))
        valid_w = int(np.clip(valid_w, 1, width))
        return cam[:valid_h, :valid_w]

    def _gradcam_feature_levels(self, num_levels):
        levels = self._vis_value('gradcam_feature_levels', None)
        if levels is None:
            return set(range(min(4, num_levels)))
        if isinstance(levels, int):
            levels = [levels]
        elif isinstance(levels, str):
            levels = [item.strip() for item in levels.split(',') if item.strip()]

        keep = set()
        for level in levels:
            try:
                level = int(level)
            except (TypeError, ValueError):
                continue
            if 0 <= level < num_levels:
                keep.add(level)
        return keep or None

    def _save_gradcam_maps(self,
                           img,
                           img_metas,
                           feats,
                           det_results=None,
                           rescale=False):
        cams = self._full_gradcam_maps(
            feats, det_results, img_metas, rescale=rescale)
        if not cams:
            cams = [None for _ in range(min(img.size(0), len(img_metas)))]

        out_dir = self._vis_value('out_dir', 'heat_map')
        save_dir = os.path.join(out_dir, 'gradcam')
        self._write_cam_maps(
            img,
            img_metas,
            cams,
            det_results,
            save_dir,
            rescale=rescale)

    def _write_cam_maps(self,
                        img,
                        img_metas,
                        cams,
                        det_results,
                        save_dir,
                        rescale=False):
        os.makedirs(save_dir, exist_ok=True)

        save_raw_cam = bool(self._vis_value('save_raw_cam', False))
        if save_raw_cam:
            raw_dir = os.path.join(save_dir, '_raw')
            os.makedirs(raw_dir, exist_ok=True)

        batch_size = min(img.size(0), len(img_metas), len(cams))
        for idx in range(batch_size):
            img_bgr = self._img_tensor_to_bgr(img[idx], img_metas[idx])
            img_h, img_w = img_metas[idx].get('img_shape', img_bgr.shape)[:2]
            img_bgr = np.ascontiguousarray(img_bgr[:img_h, :img_w])
            raw_cam, display_cam = self._split_cam_payload(cams[idx])
            name = self._safe_vis_name(img_metas[idx], idx)
            det_result = self._get_single_det_result(det_results, idx)

            if save_raw_cam and raw_cam is not None:
                raw_cam_u8 = np.uint8(np.clip(raw_cam, 0.0, 1.0) * 255.0)
                cv2.imwrite(os.path.join(raw_dir, f'{name}.png'), raw_cam_u8)
                heat_cam = cv2.applyColorMap(raw_cam_u8, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(raw_dir, f'{name}_heat.png'), heat_cam)
                if display_cam is not raw_cam:
                    display_cam_u8 = np.uint8(
                        np.clip(display_cam, 0.0, 1.0) * 255.0)
                    cv2.imwrite(
                        os.path.join(raw_dir, f'{name}_display.png'),
                        display_cam_u8)
                    display_heat = cv2.applyColorMap(
                        display_cam_u8, cv2.COLORMAP_JET)
                    cv2.imwrite(
                        os.path.join(raw_dir, f'{name}_display_heat.png'),
                        display_heat)

            overlay = self._overlay_cam(img_bgr, display_cam) \
                if display_cam is not None else img_bgr.copy()
            if self._vis_value('draw_bboxes', True):
                overlay = self._draw_det_bboxes(
                    overlay, det_result, img_metas[idx], rescale=rescale)
            cv2.imwrite(os.path.join(save_dir, f'{name}.jpg'), overlay)

    def _split_cam_payload(self, cam):
        if cam is None:
            return None, None
        if isinstance(cam, dict):
            raw = cam.get('raw', None)
            display = cam.get('display', raw)
            if raw is None:
                raw = display
            return raw, display
        return cam, cam

    def _normalize_cam(self, cam):
        cam = cam.astype(np.float32)
        if cam.size == 0 or not np.isfinite(cam).any():
            return np.zeros_like(cam, dtype=np.float32)
        cam = cam - float(np.nanmin(cam))
        denom = float(np.nanmax(cam) - np.nanmin(cam))
        if denom < 1e-6:
            return np.zeros_like(cam, dtype=np.float32)
        return cam / (denom + 1e-6)

    def _percentile_normalize(self, cam, low_p=2.0, high_p=99.5):
        cam = cam.astype(np.float32)
        sample = cam.reshape(-1)
        sample = sample[np.isfinite(sample)]
        if sample.size < 8:
            return self._normalize_cam(cam)
        low_p = float(np.clip(low_p, 0.0, 99.0))
        high_p = float(np.clip(high_p, low_p + 0.1, 100.0))
        low, high = np.percentile(sample, [low_p, high_p])
        if high <= low:
            return self._normalize_cam(cam)
        return np.clip((cam - low) / (high - low + 1e-6), 0.0, 1.0)

    def _suppress_low_cam_values(self, cam, threshold=0.0):
        threshold = float(np.clip(threshold, 0.0, 0.95))
        if threshold <= 0:
            return cam.astype(np.float32)
        cam = self._normalize_cam(cam)
        return np.clip((cam - threshold) / (1.0 - threshold + 1e-6),
                       0.0, 1.0)

    def _img_tensor_to_bgr(self, img_tensor, img_meta):
        img_np = img_tensor.detach().cpu().float().permute(1, 2, 0).numpy()
        norm_cfg = img_meta.get('img_norm_cfg', None)
        if norm_cfg is not None:
            mean = np.array(norm_cfg['mean'], dtype=np.float32)
            std = np.array(norm_cfg['std'], dtype=np.float32)
            img_np = img_np * std + mean
            if norm_cfg.get('to_rgb', True):
                img_np = img_np[..., ::-1]
        return np.clip(img_np, 0, 255).astype(np.uint8)

    def _overlay_cam(self, img_bgr, cam):
        if bool(self._vis_value('overlay_renormalize', False)):
            cam = self._percentile_normalize(
                cam,
                self._vis_value('overlay_low_percentile', 0.0),
                self._vis_value('overlay_high_percentile', 99.5))
        else:
            cam = np.clip(cam.astype(np.float32), 0.0, 1.0)

        cam = self._suppress_low_cam_values(
            cam, self._vis_value('overlay_suppress_below', 0.0))
        display_gamma = float(self._vis_value('overlay_gamma', 1.0))
        display_cam = np.power(self._normalize_cam(cam),
                               max(display_gamma, 1e-6))

        cam_u8 = np.uint8(np.clip(display_cam, 0.0, 1.0) * 255.0)
        heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        max_alpha = float(self._vis_value('heatmap_max_alpha', 0.50))
        max_alpha = float(np.clip(max_alpha, 0.0, 1.0))

        if not bool(self._vis_value('adaptive_overlay', True)):
            overlay = img_bgr.astype(np.float32) * (1.0 - max_alpha) + \
                      heatmap.astype(np.float32) * max_alpha
            return np.ascontiguousarray(np.clip(overlay, 0, 255).astype(np.uint8))

        min_alpha = float(self._vis_value('overlay_min_alpha', 0.0))
        alpha_thr = float(self._vis_value('overlay_alpha_threshold', 0.02))
        alpha_gamma = float(self._vis_value('overlay_alpha_gamma', 0.85))
        alpha_base = np.clip(
            (display_cam - alpha_thr) / (1.0 - alpha_thr + 1e-6),
            0.0,
            1.0)
        alpha_map = np.clip(min_alpha, 0.0, 1.0) + \
                    (max_alpha - np.clip(min_alpha, 0.0, 1.0)) * \
                    np.power(alpha_base, max(alpha_gamma, 1e-6))
        alpha_map = alpha_map[..., None].astype(np.float32)
        overlay = img_bgr.astype(np.float32) * (1.0 - alpha_map) + \
                  heatmap.astype(np.float32) * alpha_map
        return np.ascontiguousarray(np.clip(overlay, 0, 255).astype(np.uint8))

    def _safe_vis_name(self, img_meta, index):
        filename = img_meta.get('ori_filename', None)
        if filename is None:
            filename = img_meta.get('filename', f'sample_{index}')
        stem = os.path.splitext(os.path.basename(str(filename)))[0]
        safe = ''.join(
            ch if ch.isalnum() or ch in ('-', '_', '.') else '_'
            for ch in stem)
        return safe or f'sample_{index}'

    def _get_single_det_result(self, det_results, index):
        if det_results is None:
            return None
        if isinstance(det_results, tuple):
            det_results = det_results[0]
        if not isinstance(det_results, list) or len(det_results) == 0:
            return None
        first = det_results[0]
        if isinstance(first, tuple):
            return det_results[index][0] if index < len(det_results) else None
        if isinstance(first, list):
            return det_results[index] if index < len(det_results) else None
        return det_results

    def _get_class_name(self, cls_id):
        class_names = getattr(self, 'CLASSES', None)
        if class_names is not None and cls_id < len(class_names):
            return class_names[cls_id]
        return str(cls_id)

    def _draw_det_bboxes(self, img_bgr, det_result, img_meta, rescale=False):
        img_bgr = np.ascontiguousarray(img_bgr)
        if det_result is None:
            return img_bgr

        score_thr = float(self._vis_value('draw_score_thr', 0.30))
        max_bboxes = int(self._vis_value('draw_max_bboxes', 100))
        scale_factor = img_meta.get('scale_factor', None)
        if scale_factor is not None:
            scale_factor = np.array(scale_factor, dtype=np.float32)

        height, width = img_bgr.shape[:2]
        candidates = []
        for cls_id, cls_bboxes in enumerate(det_result):
            if cls_bboxes is None or len(cls_bboxes) == 0:
                continue
            if torch.is_tensor(cls_bboxes):
                cls_bboxes = cls_bboxes.detach().cpu().numpy()
            for bbox in cls_bboxes:
                if len(bbox) < 5 or bbox[4] < score_thr:
                    continue
                candidates.append((
                    float(bbox[4]),
                    cls_id,
                    bbox[:4].astype(np.float32)))

        candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
        for score, cls_id, coords in candidates[:max_bboxes]:
            if rescale and scale_factor is not None:
                coords = coords * scale_factor
            x1, y1, x2, y2 = coords
            x1, x2 = np.clip([x1, x2], 0, width - 1).astype(np.int32)
            y1, y2 = np.clip([y1, y2], 0, height - 1).astype(np.int32)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (230, 230, 230), 1)
            label = f'{self._get_class_name(cls_id)}:{score:.2f}'
            cv2.putText(img_bgr, label, (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1,
                        cv2.LINE_AA)
        return img_bgr
