import os.path as osp
import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formating import to_tensor
from mmcv.image.photometric import imnormalize
from mmdet.datasets.pipelines.transforms import RandomFlip


@PIPELINES.register_module()
class LoadImageAndPriors(object):
    """
    1. 加载原始图像和多个先验图
    2. 将它们堆叠成一个 (H, W, C_total) 的 Numpy 数组
    """

    def __init__(self,
                 prior_map_keys,
                 to_float32=True,
                 color_type='color',
                 prior_color_type='grayscale',
                 prior_loaders_cfg=None,
                 prior_map_dirs=None,
                 prior_file_extensions=None):

        self.prior_map_keys = prior_map_keys
        self.to_float32 = to_float32
        self.color_type = color_type
        self.default_prior_color_type = prior_color_type
        self.prior_loaders_cfg = prior_loaders_cfg if prior_loaders_cfg else {}
        self.prior_map_dirs = prior_map_dirs
        self.prior_file_extensions = prior_file_extensions

        if self.prior_map_dirs is None:
            raise ValueError("prior_map_dirs 必须在配置中指定!")
        if self.prior_file_extensions is None:
            raise ValueError("prior_file_extensions 字典必须在配置中指定!")

    def __call__(self, results):
        main_img_filename = results['img_info']['filename']
        img_path = osp.join(results['img_prefix'], main_img_filename)
        img = mmcv.imread(img_path, flag=self.color_type)
        if img is None:
            raise FileNotFoundError(f"主图像未找到于: {img_path}")
        if img.ndim == 2:
            img = mmcv.imgray2bgr(img)

        results['img_path'] = img_path
        results['filename'] = img_path
        results['ori_filename'] = results['img_info']['filename']
        results['ori_shape'] = img.shape

        priors_list = []
        base_filename = osp.splitext(osp.basename(main_img_filename))[0]

        prior_map_channels = {}

        for key in self.prior_map_keys:
            ext = self.prior_file_extensions[key]
            prior_filename = base_filename + ext
            prior_dir = self.prior_map_dirs[key]
            prior_path = osp.join(prior_dir, prior_filename)

            key_loader_cfg = self.prior_loaders_cfg.get(key, {})
            current_color_type = key_loader_cfg.get('color_type', self.default_prior_color_type)

            prior_map = mmcv.imread(prior_path, flag=current_color_type)

            if prior_map is None:
                raise FileNotFoundError(f"先验图 {key} 未找到于: {prior_path}")
            if prior_map.shape[:2] != img.shape[:2]:
                raise ValueError(f"先验图 {key} 形状为 {prior_map.shape[:2]}, 图像形状为 {img.shape[:2]}!")

            if prior_map.ndim == 2:
                prior_map = prior_map[..., None]

            priors_list.append(prior_map)
            prior_map_channels[key] = prior_map.shape[2]

        img_stack = np.concatenate([img] + priors_list, axis=2)

        if self.to_float32:
            img_stack = img_stack.astype(np.float32)

        results['img'] = img_stack
        results['img_shape'] = img_stack.shape
        results['img_fields'] = ['img']
        results['prior_map_keys_internal'] = self.prior_map_keys
        results['prior_map_channels_internal'] = prior_map_channels
        return results


@PIPELINES.register_module()
class UnpackPriors(object):
    def __init__(self, num_img_channels=3, prior_map_channels=None):
        self.num_img_channels = num_img_channels
        self.prior_map_channels = prior_map_channels

    def __call__(self, results):
        img_stack = results['img']

        keys = results['prior_map_keys_internal']
        channels_dict = results.get('prior_map_channels_internal', self.prior_map_channels)

        if channels_dict is None:
            raise ValueError("prior_map_channels 必须在配置中或由 LoadImageAndPriors 提供")

        img = img_stack[..., :self.num_img_channels]
        results['img'] = np.ascontiguousarray(img)  # (H, W, 3)

        priors_stack = img_stack[..., self.num_img_channels:]
        start_idx = 0
        for key in keys:
            num_channels = channels_dict[key]
            prior_map = priors_stack[..., start_idx: start_idx + num_channels]

            if num_channels == 1:
                prior_map = prior_map[..., np.newaxis] if prior_map.ndim == 2 else prior_map

            results[key] = np.ascontiguousarray(prior_map)
            start_idx += num_channels

        return results


@PIPELINES.register_module()
class MyToTensor(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            tensor_img = to_tensor(img.transpose(2, 0, 1))
            results[key] = DC(tensor_img, stack=True, pad_dims=2)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'


@PIPELINES.register_module()
class NormalizeCustom:
    def __init__(self, mean, std, to_rgb=False, keys=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        if keys is None:
            self.keys = ['img']
        elif isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            if key in results:
                results[key] = imnormalize(results[key], self.mean, self.std,
                                           self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb}, keys={self.keys})'
        return repr_str


@PIPELINES.register_module()
class CorrectNormalFlip:
    def __call__(self, results):
        if results.get('flip', False) and results.get('flip_direction') == 'horizontal':
            if 'normal_map' in results:
                nm = results['normal_map']

                if not nm.flags.writeable:
                    nm = nm.copy()

                nm[:, :, 2] = 255 - nm[:, :, 2]

                results['normal_map'] = nm

        return results


@PIPELINES.register_module()
class CustomPad(object):
    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            tensor = results[key]

            h, w = tensor.shape[:2]

            if self.size_divisor:
                pad_h = int(np.ceil(h / self.size_divisor)) * self.size_divisor
                pad_w = int(np.ceil(w / self.size_divisor)) * self.size_divisor
            else:
                pad_h, pad_w = h, w

            if pad_h == h and pad_w == w:
                if key == 'img':
                    results['pad_shape'] = tensor.shape
                    results['pad_fixed_size'] = None
                    results['pad_size_divisor'] = self.size_divisor
                continue

            if tensor.ndim == 2:
                tensor = tensor[:, :, None]

            c = tensor.shape[2]

            canvas = np.zeros((pad_h, pad_w, c), dtype=tensor.dtype)

            if isinstance(self.pad_val, (tuple, list)):
                assert len(self.pad_val) == c, \
                    f"CustomPad Error: pad_val 长度 ({len(self.pad_val)}) 与图像 '{key}' 通道数 ({c}) 不匹配!"

                fill_vals = np.array(self.pad_val, dtype=tensor.dtype).reshape(1, 1, -1)
                canvas[:] = fill_vals
            else:
                canvas[:] = self.pad_val

            canvas[:h, :w, :] = tensor

            results[key] = canvas

            if key == 'img':
                results['pad_shape'] = canvas.shape
                results['pad_fixed_size'] = None
                results['pad_size_divisor'] = self.size_divisor

        if 'valid_flags' in results:
            valid_flags = results['valid_flags']
            pad_h = results['pad_shape'][0]
            pad_w = results['pad_shape'][1]
            valid_flags = mmcv.impad(valid_flags, shape=(pad_h, pad_w), pad_val=0)
            results['valid_flags'] = valid_flags

        return results
