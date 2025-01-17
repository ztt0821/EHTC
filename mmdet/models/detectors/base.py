import warnings
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn

from mmdet.core import auto_fp16, get_classes, tensor2imgs
from mmdet.utils import print_log
import cv2



from mmcv.image import imread, imwrite

class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors"""

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

             **kwargs: specific to concrete implementation
        """
        pass

    async def async_simple_test(self, img, img_meta, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')

    async def aforward_test(self, *, img, img_meta, **kwargs):
        for var, name in [(img, 'img'), (img_meta, 'img_meta')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img)
        if num_augs != len(img_meta):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img), len(img_meta)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = img[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_meta[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    # def show_result(self, data, result, dataset=None, score_thr=0.3, show=False, out_file=None):
    #     result = (result[0], result[1][0])
    #     if isinstance(result, tuple):
    #         bbox_result, segm_result = result
    #     else:
    #         bbox_result, segm_result = result, None
    #
    #     img_tensor = data['img'][0]
    #     img_metas = data['img_meta'][0].data[0]
    #     imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    #     assert len(imgs) == len(img_metas)
    #
    #     dataset='mydata'
    #     if dataset is None:
    #         class_names = self.CLASSES
    #     elif isinstance(dataset, str):
    #         class_names = get_classes(dataset)
    #         print(1)
    #     elif isinstance(dataset, (list, tuple)):
    #         class_names = dataset
    #     else:
    #         raise TypeError(
    #             'dataset must be a valid dataset name or a sequence'
    #             ' of class names, not {}'.format(type(dataset)))
    #     # print(class_names)
    #     for img, img_meta in zip(imgs, img_metas):
    #         h, w, _ = img_meta['img_shape']
    #         img_show = img[:h, :w, :]
    #
    #         bboxes = np.vstack(bbox_result)
    #         # draw segmentation masks
    #         if segm_result is not None:
    #             segms = mmcv.concat_list(segm_result)
    #             inds = np.where(bboxes[:, -1] > score_thr)[0]
    #             for i in inds:
    #                 color_mask = np.random.randint(
    #                     0, 256, (1, 3), dtype=np.uint8)
    #                 mask = maskUtils.decode(segms[i]).astype(np.bool)
    #                 img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
    #         # draw bounding boxes
    #         labels = [
    #             np.full(bbox.shape[0], i, dtype=np.int32)
    #             for i, bbox in enumerate(bbox_result)
    #         ]
    #         labels = np.concatenate(labels)
    #         mmcv.imshow_det_bboxes(
    #             img_show,
    #             bboxes,
    #             labels,
    #             class_names=class_names,
    #             score_thr=score_thr,
    #             out_file=out_file)
    #         if not (show or out_file):
    #             warnings.warn('show==False and out_file is not specified, only '
    #                           'result image will be returned')
    #             return img

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.
        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        result = (result[0], result[1][0])
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        dataset = 'mydata'
        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
            print(1)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                     'dataset must be a valid dataset name or a sequence'
                     ' of class names, not {}'.format(type(dataset)))
        print(class_names)
        # draw segmentation masks
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        mmcv.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=class_names,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img



    def show_result2(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.
        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        result = (result[0], result[1][0])
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        dataset = 'mydata'
        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
            print(1)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                     'dataset must be a valid dataset name or a sequence'
                     ' of class names, not {}'.format(type(dataset)))
        print(class_names)
        # draw segmentation masks
        if segm_result is not None:
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = maskUtils.decode(segms[i]).astype(np.bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        img = imread(img)

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        bbox_color = (0, 255, 0)
        text_color = (0, 255, 0)

        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, bbox_color, thickness=thickness)
            label_text = class_names[
                label] if class_names is not None else 'cls {}'.format(label)
            if len(bbox) > 4:
                label_text += '|{:.02f}'.format(bbox[-1])
            # cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
            #             cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

        if out_file is not None:
            imwrite(img, out_file)
        # mmcv.imshow_det_bboxes(
        #     img,
        #     bboxes,
        #     labels,
        #     class_names=class_names,
        #     score_thr=score_thr,
        #     bbox_color=bbox_color,
        #     text_color=text_color,
        #     thickness=thickness,
        #     font_scale=font_scale,
        #     win_name=win_name,
        #     show=show,
        #     wait_time=wait_time,
        #     out_file=out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img
