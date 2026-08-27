import torch
from mmdet.core import bbox2roi
from ..builder import HEADS, build_head
from .prob_roi_head import ProbRoIHead
import torch.nn.functional as F

from .physics_weight import PhysicsWeight


@HEADS.register_module()
class PhysicsProbRoIHead(ProbRoIHead):
    def __init__(self,
                 physics_weight=None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.physics_weight = None
        if physics_weight:

            physics_weight.update(gamma=self.gamma)

            if self.with_bbox:
                physics_weight.update(
                    roi_feat_size=self.bbox_head.roi_feat_size
                )

            self.physics_weight = build_head(physics_weight)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      freq_prior=None,
                      env_prior=None):

        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []

            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()

        if self.with_bbox:
            if self.boost:
                bbox_results = self._bbox_forward_train_boost(
                    x, sampling_results,
                    gt_bboxes, gt_labels,
                    img_metas,
                    freq_prior=freq_prior,
                    env_prior=env_prior)
            else:
                bbox_results = self._bbox_forward_train(
                    x, sampling_results,
                    gt_bboxes, gt_labels,
                    img_metas,
                    freq_prior=freq_prior,
                    env_prior=env_prior)
            losses.update(bbox_results['loss_bbox'])

        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
        return losses

    def _bbox_forward_train_boost(self, x, sampling_results,
                                  gt_bboxes, gt_labels, img_metas,
                                  freq_prior=None, env_prior=None):

        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(x, rois)

        labels, label_weights_std, bbox_targets, bbox_weights = \
            self.bbox_head.get_targets(
                sampling_results,
                gt_bboxes,
                gt_labels,
                self.train_cfg)

        physics_weights = self.physics_weight(
            rois=rois,
            labels=labels,
            bg_class_ind=self.bbox_head.num_classes,
            env_prior=env_prior,
            freq_prior=freq_prior
        ).detach()

        final_weights = label_weights_std * physics_weights

        loss_bbox_raw = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois,
            labels,
            label_weights_std,
            bbox_targets,
            bbox_weights,
            reduction_override='none'
        )

        if not hasattr(self, '_debug_printed'):
            print(loss_bbox_raw['loss_cls'].shape)
            print(loss_bbox_raw['loss_bbox'].shape)
            self._debug_printed = True
        if not hasattr(self, '_debug_weight'):
            print('physics_weights:',
                  physics_weights.min().item(),
                  physics_weights.max().item(),
                  physics_weights.mean().item())
            self._debug_weight = True
        avg_factor = max(label_weights_std.sum().item(), 1.0)
        loss_cls = self.norm_loss(
            loss_bbox_raw['loss_cls'],
            final_weights,
            avg_factor=avg_factor
        )

        num_pos = max((labels < self.bbox_head.num_classes).sum().item(), 1)

        loss_bbox_reg = loss_bbox_raw['loss_bbox'].mean()

        loss_bbox = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox_reg
        )

        bbox_results.update(loss_bbox=loss_bbox)

        return bbox_results

    def _bbox_forward_train(self, x, sampling_results,
                            gt_bboxes, gt_labels, img_metas,
                            freq_prior=None, env_prior=None):
        rois = bbox2roi([res.bboxes for res in sampling_results])

        bbox_results = self._bbox_forward(x, rois)

        labels, label_weights_std, bbox_targets, bbox_weights = \
            self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)

        final_weights = label_weights_std

        loss_bbox_raw = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois,
            labels,
            label_weights_std,
            bbox_targets,
            bbox_weights,
            reduction_override='none'
        )

        avg_factor = max(label_weights_std.sum().item(), 1.0)
        loss_cls = self.norm_loss(
            loss_bbox_raw['loss_cls'],
            final_weights,
            avg_factor=avg_factor
        )

        num_pos = max((labels < self.bbox_head.num_classes).sum().item(), 1)

        loss_bbox_reg = loss_bbox_raw['loss_bbox'].mean()

        loss_bbox = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox_reg
        )

        bbox_results.update(loss_bbox=loss_bbox)

        return bbox_results

    def aug_test(self, x, proposal_list, img_metas, rescale=False, **kwargs):

        return super().aug_test(x, proposal_list, img_metas, rescale=rescale)

    def simple_test(self, x, proposal_list, img_metas, rescale=False,
                    freq_prior=None, env_prior=None, **kwargs):

        return super(MyPhysicsProbRoIHead, self).simple_test(
            x, proposal_list, img_metas, rescale=rescale, **kwargs
        )
