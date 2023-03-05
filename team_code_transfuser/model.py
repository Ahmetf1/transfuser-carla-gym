import copy
from collections import deque

import numpy as np
import torch.nn.functional as F
import cv2

from utils import *
from transfuser import TransfuserBackbone, SegDecoder, DepthDecoder
from geometric_fusion import GeometricFusionBackbone
from late_fusion import LateFusionBackbone
from latentTF import latentTFBackbone
from point_pillar import PointPillarNet

from PIL import Image, ImageFont, ImageDraw
from torchvision import models

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin

# safety controller
from queue import Queue

# RL_old
# from RL.Agent import Agent
# from torch.utils.tensorboard import SummaryWriter
# import json

# Rl tianshou


# import argparse
# import os
# import pickle
# import pprint
#
# #import gym
# import numpy as np
# import torch
# from torch.utils.tensorboard import SummaryWriter
#
# from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
# from tianshou.env import DummyVectorEnv
# from tianshou.policy import RainbowPolicy
# from tianshou.trainer import offpolicy_trainer
# from tianshou.utils import TensorboardLogger
# from tianshou.utils.net.common import Net
# from tianshou.utils.net.discrete import NoisyLinear


@HEADS.register_module()
class LidarCenterNetHead(BaseDenseHead, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_dir_class=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_dir_res=dict(type='SmoothL1Loss', loss_weight=1.0),
                 loss_velocity=dict(type='L1Loss', loss_weight=1.0),
                 loss_brake=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(LidarCenterNetHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.num_dir_bins = train_cfg.num_dir_bins
        self.yaw_class_head = self._build_head(in_channel, feat_channel, self.num_dir_bins)
        self.yaw_res_head = self._build_head(in_channel, feat_channel, 1)
        self.velocity_head = self._build_head(in_channel, feat_channel, 1)
        self.brake_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_dir_class = build_loss(loss_dir_class)
        self.loss_dir_res = build_loss(loss_dir_res)
        self.loss_velocity = build_loss(loss_velocity)
        self.loss_brake = build_loss(loss_brake)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = train_cfg.fp16_enabled
        self.i = 0

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(self.train_cfg.center_net_bias_init_with_prob)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=self.train_cfg.center_net_normal_init_std)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        yaw_class_pred = self.yaw_class_head(feat)
        yaw_res_pred = self.yaw_res_head(feat)
        velocity_pred = self.velocity_head(feat)
        brake_pred = self.brake_head(feat)

        return center_heatmap_pred, wh_pred, offset_pred, yaw_class_pred, yaw_res_pred, velocity_pred, brake_pred

    @force_fp32(apply_to=(
            'center_heatmap_preds', 'wh_preds', 'offset_preds', 'yaw_class_preds', 'yaw_res_preds', 'velocity_pred',
            'brake_pred'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             yaw_class_preds,
             yaw_res_preds,
             velocity_preds,
             brake_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]
        yaw_class_pred = yaw_class_preds[0]
        yaw_res_pred = yaw_res_preds[0]
        velocity_pred = velocity_preds[0]
        brake_pred = brake_preds[0]

        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels, gt_bboxes_ignore,
                                                     center_heatmap_pred.shape)

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        yaw_class_target = target_result['yaw_class_target']
        yaw_res_target = target_result['yaw_res_target']
        offset_target = target_result['offset_target']
        velocity_target = target_result['velocity_target']
        brake_target = target_result['brake_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_yaw_class = self.loss_dir_class(
            yaw_class_pred,
            yaw_class_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)
        loss_yaw_res = self.loss_dir_res(
            yaw_res_pred,
            yaw_res_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)
        loss_velocity = self.loss_velocity(
            velocity_pred,
            velocity_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)
        loss_brake = self.loss_brake(
            brake_pred,
            brake_target,
            wh_offset_target_weight[:, :1, ...],
            avg_factor=avg_factor)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_yaw_class=loss_yaw_class,
            loss_yaw_res=loss_yaw_res,
            loss_velocity=loss_velocity,
            loss_brake=loss_brake)

    def angle2class(self, angle):
        """Convert continuous angle to a discrete class and a residual.
        Convert continuous angle to a discrete class and a small
        regression number from class center angle to current angle.
        Args:
            angle (torch.Tensor): Angle is from 0-2pi (or -pi~pi),
                class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N).
        Returns:
            tuple: Encoded discrete class and residual.
        """
        angle = angle % (2 * np.pi)
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        # NOTE changed this to not trigger a warning anymore. Rounding trunc should be the same as floor as long as angle is positive.
        # I kept it trunc to not change the behavior and keep backwards compatibility. When training a new model "floor" might be the better option.
        angle_cls = torch.div(shifted_angle, angle_per_class, rounding_mode="trunc")
        angle_res = shifted_angle - (angle_cls * angle_per_class + angle_per_class / 2)
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        """Inverse function to angle2class.
        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].
        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle

    def get_targets(self, gt_bboxes, gt_labels, gt_ignores, feat_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = self.train_cfg.lidar_resolution_height, self.train_cfg.lidar_resolution_width
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        yaw_class_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w]).long()
        yaw_res_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w])
        velocity_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w])
        brake_target = gt_bboxes[-1].new_zeros([bs, 1, feat_h, feat_w]).long()

        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[0][batch_id]
            gt_label = gt_labels[0][batch_id]
            gt_ignore = gt_ignores[0][batch_id]

            center_x = gt_bbox[:, [0]] * width_ratio
            center_y = gt_bbox[:, [1]] * width_ratio
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                if gt_ignore[j]:
                    continue

                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = gt_bbox[j, 3] * height_ratio
                scale_box_w = gt_bbox[j, 2] * width_ratio

                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.1)
                radius = max(2, int(radius))
                ind = gt_label[j].long()

                gen_gaussian_target(center_heatmap_target[batch_id, ind], [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                yaw_class, yaw_res = self.angle2class(gt_bbox[j, 4])

                yaw_class_target[batch_id, 0, cty_int, ctx_int] = yaw_class
                yaw_res_target[batch_id, 0, cty_int, ctx_int] = yaw_res

                velocity_target[batch_id, 0, cty_int, ctx_int] = gt_bbox[j, 5]
                brake_target[batch_id, 0, cty_int, ctx_int] = gt_bbox[j, 6].long()

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int
                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            yaw_class_target=yaw_class_target.squeeze(1),
            yaw_res_target=yaw_res_target,
            offset_target=offset_target,
            velocity_target=velocity_target,
            brake_target=brake_target.squeeze(1),
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   yaw_class_preds,
                   yaw_res_preds,
                   velocity_preds,
                   brake_preds,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1

        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            yaw_class_preds[0],
            yaw_res_preds[0],
            velocity_preds[0],
            brake_preds[0],
            k=self.train_cfg.top_k_center_keypoints,
            kernel=self.train_cfg.center_net_max_pooling_kernel)

        if with_nms:
            det_results = []
            for (det_bboxes, det_labels) in zip(batch_det_bboxes,
                                                batch_labels):
                det_bbox, det_label = self._bboxes_nms(det_bboxes, det_labels,
                                                       self.test_cfg)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
            ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       yaw_class_pred,
                       yaw_res_pred,
                       velocity_pred,
                       brake_pred,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        yaw_class = transpose_and_gather_feat(yaw_class_pred, batch_index)
        yaw_res = transpose_and_gather_feat(yaw_res_pred, batch_index)
        velocity = transpose_and_gather_feat(velocity_pred, batch_index)
        brake = transpose_and_gather_feat(brake_pred, batch_index)
        brake = torch.argmax(brake, -1)
        velocity = velocity[..., 0]

        # convert class + res to yaw
        yaw_class = torch.argmax(yaw_class, -1)
        yaw = self.class2angle(yaw_class, yaw_res.squeeze(2))
        # speed

        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]

        ratio = 4.

        batch_bboxes = torch.stack([topk_xs, topk_ys, wh[..., 0], wh[..., 1], yaw, velocity, brake], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        batch_bboxes[:, :, :4] *= ratio

        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() == 0:
            return bboxes, labels

        out_bboxes, keep = batched_nms(bboxes[:, :4].contiguous(),
                                       bboxes[:, -1].contiguous(), labels,
                                       cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]

        return out_bboxes, out_labels


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class LidarCenterNet(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        in_channels: input channels
    """

    def __init__(self, config, device, backbone, image_architecture='resnet34', lidar_architecture='resnet18',
                 use_velocity=True):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len
        self.use_target_point_image = config.use_target_point_image
        self.gru_concat_target_point = config.gru_concat_target_point
        self.use_point_pillars = config.use_point_pillars

        if (self.use_point_pillars == True):
            self.point_pillar_net = PointPillarNet(config.num_input, config.num_features,
                                                   min_x=config.min_x, max_x=config.max_x,
                                                   min_y=config.min_y, max_y=config.max_y,
                                                   pixels_per_meter=int(config.pixels_per_meter),
                                                   )

        self.backbone = backbone

        if (backbone == 'transFuser'):
            self._model = TransfuserBackbone(config, image_architecture, lidar_architecture,
                                             use_velocity=use_velocity).to(self.device)
        elif (backbone == 'late_fusion'):
            self._model = LateFusionBackbone(config, image_architecture, lidar_architecture,
                                             use_velocity=use_velocity).to(self.device)
        elif (backbone == 'geometric_fusion'):
            self._model = GeometricFusionBackbone(config, image_architecture, lidar_architecture,
                                                  use_velocity=use_velocity).to(self.device)
        elif (backbone == 'latentTF'):
            self._model = latentTFBackbone(config, image_architecture, lidar_architecture,
                                           use_velocity=use_velocity).to(self.device)
        else:
            raise (
                "The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")

        if config.multitask:
            self.seg_decoder = SegDecoder(self.config, self.config.perception_output_features).to(self.device)
            self.depth_decoder = DepthDecoder(self.config, self.config.perception_output_features).to(self.device)

        channel = config.channel

        self.pred_bev = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, kernel_size=(1, 1), stride=1, padding=0, bias=True)
        ).to(self.device)

        # prediction heads
        self.head = LidarCenterNetHead(channel, channel, 1, train_cfg=config).to(self.device)
        self.i = 0

        # waypoints prediction
        self.join = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        ).to(self.device)

        self.decoder = nn.GRUCell(input_size=4 if self.gru_concat_target_point else 2,  # 2 represents x,y coordinate
                                  hidden_size=self.config.gru_hidden_size).to(self.device)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(self.config.gru_hidden_size, 3).to(self.device)

        # pid controller
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD,
                                             n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD,
                                              n=config.speed_n)

        # safety controller
        # self.safety_controller = SafetyController(3, self.config, self.device, False)

        # RL controller
        # self.rl_controller = RlController(self.config, self.device, None, None)
        self.rl_data_logger = RlDataLogger(self.config)
        # self.rl_controller = RlController(self.config, self.rl_data_logger)

    def forward_gru(self, z, target_point):
        z = self.join(z)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

        target_point = target_point.clone()
        target_point[:, 1] *= -1

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            if self.gru_concat_target_point:
                x_in = torch.cat([x, target_point], dim=1)
            else:
                x_in = x

            z = self.decoder(x_in, z)
            dx = self.output(z)

            x = dx[:, :2] + x

            output_wp.append(x[:, :2])

        pred_wp = torch.stack(output_wp, dim=1)

        # pred the wapoints in the vehicle coordinate and we convert it to lidar coordinate here because the GT waypoints is in lidar coordinate
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - self.config.lidar_pos[0]

        pred_brake = None
        steer = None
        throttle = None
        brake = None

        return pred_wp, pred_brake, steer, throttle, brake

    def control_pid(self, waypoints, velocity, is_stuck):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        '''
        assert (waypoints.size(0) == 1)
        waypoints = waypoints[0].data.cpu().numpy()
        # when training we transform the waypoints to lidar coordinate, so we need to change is back when control
        waypoints[:, 0] += self.config.lidar_pos[0]

        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0

        if is_stuck:
            desired_speed = np.array(self.config.default_speed)  # default speed of 14.4 km/h

        brake = ((desired_speed < self.config.brake_speed) or ((speed / desired_speed) > self.config.brake_ratio))

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.clip_throttle)
        throttle = throttle if not brake else 0.0
        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if (speed < 0.01):
            angle = 0.0  # When we don't move we don't want the angle error to accumulate in the integral
        if brake:
            angle = 0.0

        steer = self.turn_controller.step(angle)

        steer = np.clip(steer, -1.0, 1.0)  # Valid steering values are in [-1,1]

        return steer, throttle, brake

    def forward_ego(self, rgb, lidar_bev, target_point, target_point_image, ego_vel, bev_points=None, cam_points=None,
                    save_path=None, expert_waypoints=None,
                    stuck_detector=0, forced_move=False, num_points=None, rgb_back=None, debug=False):

        if (self.use_point_pillars == True):
            lidar_bev = self.point_pillar_net(lidar_bev, num_points)
            lidar_bev = torch.rot90(lidar_bev, -1, dims=(2, 3))  # For consitency this is also done in voxelization

        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)

        if (self.backbone == 'transFuser'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'late_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'geometric_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel, bev_points, cam_points)
        elif (self.backbone == 'latentTF'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        else:
            raise (
                "The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")

        pred_wp, _, _, _, _ = self.forward_gru(fused_features, target_point)

        preds = self.head([features[0]])
        results = self.head.get_bboxes(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6])
        bboxes, _ = results[0]

        # filter bbox based on the confidence of the prediction
        bboxes = bboxes[bboxes[:, -1] > self.config.bb_confidence_threshold]
        # pred_wp, safety_on = self.safety_controller.forward(pred_wp, bboxes, self.i)
        rotated_bboxes = []
        for bbox in bboxes.detach().cpu().numpy():
            bbox = self.get_bbox_local_metric(bbox)
            rotated_bboxes.append(bbox)

        self.i += 1
        # if debug and self.i % 2 == 0 and not (save_path is None):
        if True:
            pred_bev = self.pred_bev(features[0])
            pred_bev = F.interpolate(pred_bev, (self.config.bev_resolution_height, self.config.bev_resolution_width),
                                     mode='bilinear', align_corners=True)
            pred_semantic = self.seg_decoder(image_features_grid)
            pred_depth = self.depth_decoder(image_features_grid)

            self.visualize_model_io(save_path, self.i, self.config, rgb, lidar_bev, target_point,
                                    pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, self.device,
                                    gt_bboxes=None, expert_waypoints=expert_waypoints, stuck_detector=stuck_detector,
                                    forced_move=forced_move, safety_on=False)

        self.rl_data_logger.waypoints = self.rl_data_logger.transform_waypoints(pred_wp)
        self.rl_data_logger.hd_map = self.rl_data_logger.transform_hdmap(pred_bev)
        self.rl_data_logger.bboxes = self.rl_data_logger.transform_bboxes(rotated_bboxes)

        return pred_wp, rotated_bboxes

    def forward(self, rgb, lidar_bev, ego_waypoint, target_point, target_point_image, ego_vel, bev, label, depth,
                semantic, num_points=None, save_path=None, bev_points=None, cam_points=None):
        loss = {}

        if (self.use_point_pillars == True):
            lidar_bev = self.point_pillar_net(lidar_bev, num_points)
            lidar_bev = torch.rot90(lidar_bev, -1, dims=(2, 3))  # For consitency this is also done in voxelization

        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)

        if (self.backbone == 'transFuser'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'late_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        elif (self.backbone == 'geometric_fusion'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel, bev_points, cam_points)
        elif (self.backbone == 'latentTF'):
            features, image_features_grid, fused_features = self._model(rgb, lidar_bev, ego_vel)
        else:
            raise (
                "The chosen vision backbone does not exist. The options are: transFuser, late_fusion, geometric_fusion, latentTF")

        pred_wp, _, _, _, _ = self.forward_gru(fused_features, target_point)

        # pred topdown view
        pred_bev = self.pred_bev(features[0])
        pred_bev = F.interpolate(pred_bev, (self.config.bev_resolution_height, self.config.bev_resolution_width),
                                 mode='bilinear', align_corners=True)

        weight = torch.from_numpy(np.array([1., 1., 3.])).to(dtype=torch.float32, device=pred_bev.device)
        loss_bev = F.cross_entropy(pred_bev, bev, weight=weight).mean()

        loss_wp = torch.mean(torch.abs(pred_wp - ego_waypoint))
        loss.update({
            "loss_wp": loss_wp,
            "loss_bev": loss_bev
        })

        preds = self.head([features[0]])

        gt_labels = torch.zeros_like(label[:, :, 0])
        gt_bboxes_ignore = label.sum(dim=-1) == 0.
        loss_bbox = self.head.loss(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6],
                                   [label], gt_labels=[gt_labels], gt_bboxes_ignore=[gt_bboxes_ignore], img_metas=None)

        loss.update(loss_bbox)

        if self.config.multitask:
            pred_semantic = self.seg_decoder(image_features_grid)
            pred_depth = self.depth_decoder(image_features_grid)
            loss_semantic = self.config.ls_seg * F.cross_entropy(pred_semantic, semantic).mean()
            loss_depth = self.config.ls_depth * F.l1_loss(pred_depth, depth).mean()
            loss.update({
                "loss_depth": loss_depth,
                "loss_semantic": loss_semantic
            })
        else:
            loss.update({
                "loss_depth": torch.zeros_like(loss_wp),
                "loss_semantic": torch.zeros_like(loss_wp)
            })

        self.i += 1
        if ((self.config.debug == True) and (self.i % self.config.train_debug_save_freq == 0) and (save_path != None)):
            with torch.no_grad():
                results = self.head.get_bboxes(preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6])
                bboxes, _ = results[0]
                bboxes = bboxes[bboxes[:, -1] > self.config.bb_confidence_threshold]
                self.visualize_model_io(save_path, self.i, self.config, rgb, lidar_bev, target_point,
                                        pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, self.device,
                                        gt_bboxes=label, expert_waypoints=ego_waypoint, stuck_detector=0,
                                        forced_move=False)

        return loss

    # Converts the coordinate system to x front y right, vehicle center at the origin.
    # Units are converted from pixels to meters
    def get_bbox_local_metric(self, bbox):
        x, y, w, h, yaw, speed, brake, confidence = bbox

        w = w / self.config.bounding_box_divisor / self.config.pixels_per_meter  # We multiplied by 2 when collecting the data, and multiplied by 8 when loading the labels.
        h = h / self.config.bounding_box_divisor / self.config.pixels_per_meter  # We multiplied by 2 when collecting the data, and multiplied by 8 when loading the labels.

        T = get_lidar_to_bevimage_transform()
        T_inv = np.linalg.inv(T)

        center = np.array([x, y, 1.0])

        center_old_coordinate_sys = T_inv @ center

        center_old_coordinate_sys = center_old_coordinate_sys + np.array(self.config.lidar_pos)

        # Convert to standard CARLA right hand coordinate system
        center_old_coordinate_sys[1] = -center_old_coordinate_sys[1]

        bbox = np.array([[-h, -w, 1],
                         [-h, w, 1],
                         [h, w, 1],
                         [h, -w, 1],
                         [0, 0, 1],
                         [0, h * speed * 0.5, 1]])

        R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

        for point_index in range(bbox.shape[0]):
            bbox[point_index] = R @ bbox[point_index]
            bbox[point_index] = bbox[point_index] + np.array(
                [center_old_coordinate_sys[0], center_old_coordinate_sys[1], 0])

        return bbox, brake, confidence

    # this is different
    def get_rotated_bbox(self, bbox):
        x, y, w, h, yaw, speed, brake = bbox

        bbox = np.array([[h, w, 1],
                         [h, -w, 1],
                         [-h, -w, 1],
                         [-h, w, 1],
                         [0, 0, 1],
                         [-h * speed * 0.5, 0, 1]])
        bbox[:, :2] /= self.config.bounding_box_divisor
        bbox[:, :2] = bbox[:, [1, 0]]

        c, s = np.cos(yaw), np.sin(yaw)
        # use y x because coordinate is changed
        r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

        bbox = r1_to_world @ bbox.T
        bbox = bbox.T

        return bbox, brake

    def draw_bboxes(self, bboxes, image, color=(255, 255, 255), brake_color=(0, 0, 255)):
        idx = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
        for bbox, brake in bboxes:
            bbox = bbox.astype(np.int32)[:, :2]
            for s, e in idx:
                if brake >= self.config.draw_brake_threshhold:
                    color = brake_color
                else:
                    color = color
                # brake is true while still have high velocity
                cv2.line(image, tuple(bbox[s]), tuple(bbox[e]), color=color, thickness=1)
        return image

    def draw_waypoints(self, label, waypoints, image, color=(255, 255, 255)):
        waypoints = waypoints.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        for bbox, points in zip(label, waypoints):
            x, y, w, h, yaw, speed, brake = bbox
            c, s = np.cos(yaw), np.sin(yaw)
            # use y x because coordinate is changed
            r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

            # convert to image space
            # need to negate y componet as we do for lidar points
            # we directly construct points in the image coordiante
            # for lidar, forward +x, right +y
            #            x
            #            +
            #            |
            #            |
            #            |---------+y
            #
            # for image, ---------> x
            #            |
            #            |
            #            +
            #            y

            points[:, 0] *= -1
            points = points * self.config.pixels_per_meter
            points = points[:, [1, 0]]
            points = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1)

            points = r1_to_world @ points.T
            points = points.T

            points_to_draw = []
            for point in points[:, :2]:
                points_to_draw.append(point.copy())
                point = point.astype(np.int32)
                cv2.circle(image, tuple(point), radius=3, color=color, thickness=3)
        return image

    def draw_target_point(self, target_point, image, color=(255, 255, 255)):
        target_point = target_point.copy()

        target_point[1] += self.config.lidar_pos[0]
        point = target_point * self.config.pixels_per_meter
        point[1] *= -1
        point[1] = self.config.lidar_resolution_width - point[1]  # Might be LiDAR height
        point[0] += int(self.config.lidar_resolution_height / 2.0)  # Might be LiDAR width
        point = point.astype(np.int32)
        point = np.clip(point, 0, 512)
        cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
        return image

    def visualize_model_io(self, save_path, step, config, rgb, lidar_bev, target_point,
                           pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, device,
                           gt_bboxes=None, expert_waypoints=None, stuck_detector=0, forced_move=False, safety_on=False):
        font = ImageFont.load_default()
        i = 0  # We only visualize the first image if there is a batch of them.
        if config.multitask:
            classes_list = config.classes_list
            converter = np.array(classes_list)

            depth_image = pred_depth[i].detach().cpu().numpy()

            indices = np.argmax(pred_semantic.detach().cpu().numpy(), axis=1)
            semantic_image = converter[indices[i, ...], ...].astype('uint8')

            ds_image = np.stack((depth_image, depth_image, depth_image), axis=2)
            ds_image = (ds_image * 255).astype(np.uint8)
            ds_image = np.concatenate((ds_image, semantic_image), axis=0)
            ds_image = cv2.resize(ds_image, (640, 256))
            ds_image = np.concatenate([ds_image, np.zeros_like(ds_image[:50])], axis=0)

        images = np.concatenate(list(lidar_bev.detach().cpu().numpy()[i][:2]), axis=1)
        images = (images * 255).astype(np.uint8)
        images = np.stack([images, images, images], axis=-1)
        images = np.concatenate([images, np.zeros_like(images[:50])], axis=0)

        # draw bbox GT
        if (not (gt_bboxes is None)):
            rotated_bboxes_gt = []
            for bbox in gt_bboxes.detach().cpu().numpy()[i]:
                bbox = self.get_rotated_bbox(bbox)
                rotated_bboxes_gt.append(bbox)
            images = self.draw_bboxes(rotated_bboxes_gt, images, color=(0, 255, 0), brake_color=(0, 255, 128))

        rotated_bboxes = []
        for bbox in bboxes.detach().cpu().numpy():
            bbox = self.get_rotated_bbox(bbox[:7])
            rotated_bboxes.append(bbox)
        images = self.draw_bboxes(rotated_bboxes, images, color=(255, 0, 0), brake_color=(0, 255, 255))

        label = torch.zeros((1, 1, 7)).to(device)
        label[:, -1, 0] = 128.
        label[:, -1, 1] = 256.

        if not expert_waypoints is None:
            images = self.draw_waypoints(label[0], expert_waypoints[i:i + 1], images, color=(0, 0, 255))

        images = self.draw_waypoints(label[0], pred_wp[i:i + 1, 2:], images,
                                     color=(255, 255, 255))  # Auxliary waypoints in white
        images = self.draw_waypoints(label[0], pred_wp[i:i + 1, :2], images,
                                     color=(255, 0, 0))  # First two, relevant waypoints in blue

        # draw target points
        images = self.draw_target_point(target_point[i].detach().cpu().numpy(), images)

        # stuck text
        images = Image.fromarray(images)
        draw = ImageDraw.Draw(images)
        draw.text((10, 0), "stuck detector:   %04d" % (stuck_detector), font=font)
        draw.text((10, 30), "forced move:      %s" % (" True" if forced_move else "False"), font=font,
                  fill=(255, 0, 0, 255) if forced_move else (255, 255, 255, 255))
        images = np.array(images)

        bev = pred_bev[i].detach().cpu().numpy().argmax(axis=0) / 2.
        bev = np.stack([bev, bev, bev], axis=2) * 255.
        bev_image = bev.astype(np.uint8)
        bev_image = cv2.resize(bev_image, (256, 256))
        bev_image = np.concatenate([bev_image, np.zeros_like(bev_image[:50])], axis=0)

        if not expert_waypoints is None:
            bev_image = self.draw_waypoints(label[0], expert_waypoints[i:i + 1], bev_image, color=(0, 0, 255))

        bev_image = self.draw_waypoints(label[0], pred_wp[i:i + 1], bev_image, color=(255, 255, 255))
        bev_image = self.draw_waypoints(label[0], pred_wp[i:i + 1, :2], bev_image, color=(255, 0, 0))

        bev_image = self.draw_target_point(target_point[i].detach().cpu().numpy(), bev_image)

        if (not (expert_waypoints is None)):
            aim = expert_waypoints[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
            expert_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))

            aim = pred_wp[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
            ego_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))
            angle_error = normalize_angle_degree(expert_angle - ego_angle)

            bev_image = Image.fromarray(bev_image)
            draw = ImageDraw.Draw(bev_image)
            draw.text((0, 0), "Angle error:        %.2f°" % (angle_error), font=font)

        bev_image = np.array(bev_image)

        rgb_image = rgb[i].permute(1, 2, 0).detach().cpu().numpy()[:, :, [2, 1, 0]]
        rgb_image = cv2.resize(rgb_image, (1280 + 128, 320 + 32))
        assert (config.multitask)
        images = np.concatenate((bev_image, images, ds_image), axis=1)

        images = np.concatenate((rgb_image, images), axis=0)

        cv2.imwrite(str(save_path + ("/%d.png" % (step // 2))), images)
        if safety_on:
            cv2.imwrite(str(save_path + ("/danger/%d.png" % (step // 2))), images)


# class TFuseEnv(gym.Env):
#     def __init__(self, observer):
#         self.observation_space = gym.spaces.Dict({
#             "steer": gym.spaces.Box(-1, 1), "throttle": gym.spaces.Box(0,1)})
#         self.action_space = gym.spaces.Discrete(44)
#
#         self.observer = observer
#
#         self.prev_observation = None
#         self.current_observation = None
#
#         self.prev_steer = None
#         self.prev_throttle = None
#
#         self.unmodified_steer = None
#         self.unmodified_throttle = None
#
#     def step(self, action):
#         steer = ((action[0] % 11) - 5) / 5
#         throttle = (action[0] % 4) / 3
#
#         reward = (0.5 - (((steer - self.prev_steer)**2) / 8)) + (0.5 - ((throttle - self.prev_throttle)**2) / 8)
#
#         return self.observer.observe(), reward, True, {}
#
#     def reset(self):
#         return self.observer.return_data(), {}


class RlDataLogger:
    def __init__(self, config):
        self.config = config
        self.waypoints = None  # size = 8
        self.hd_map = None
        self.bboxes = None

        self.throttle = None
        self.steer = None

    def return_data(self):
        return {
            "waypoints": self.waypoints,
            "hd_map": self.hd_map,
            "bboxes": self.bboxes
        }

    def return_outdata(self):
        data = {
            "steer": self.steer,
            "throttle": self.throttle
        }
        return data

    def transform_waypoints(self, waypoints):
        waypoints = waypoints.detach().cpu().numpy()
        label = torch.zeros((1, 1, 7))
        label[:, -1, 0] = 128.
        label[:, -1, 1] = 256.
        label = label.detach().cpu().numpy()
        points_transformed = []

        for bbox, points in zip(label[0], waypoints):
            x, y, w, h, yaw, speed, brake = bbox
            c, s = np.cos(yaw), np.sin(yaw)
            # use y x because coordinate is changed
            r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

            points[:, 0] *= -1
            points = points * self.config.pixels_per_meter
            points = points[:, [1, 0]]
            points = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1)

            points = r1_to_world @ points.T
            points = points.T

            for point in points[:, :2]:
                points_transformed.append(point.copy())

        return points_transformed

    def transform_hdmap(self, pred_bev):
        bev = pred_bev[0].detach().cpu().numpy().argmax(axis=0) / 2.
        bev = np.stack([bev, bev, bev], axis=2) * 255.
        bev_image = bev.astype(np.uint8)
        bev_image = cv2.resize(bev_image, (256, 256))


        cv2.imwrite(str("/home/transfuser/hd_map.png"), bev_image)

        return bev_image

    def transform_bboxes(self, bboxes):
        rotated_bboxes = []
        for bbox in bboxes.detach().cpu().numpy():
            bbox = self.get_rotated_bbox(bbox[:7])
            rotated_bboxes.append(bbox)

        idx = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]

        image = np.zeros((256, 512))

        for bbox, brake in bboxes:
            bbox = bbox.astype(np.int32)[:, :2]
            contours = np. array(bbox[idx])
            cv2.fillPoly(image, pts=[contours], color=255)

        cv2.imwrite(str("/home/transfuser/image.png"), image)

        return image

    def get_rotated_bbox(self, bbox):
        x, y, w, h, yaw, speed, brake = bbox

        bbox = np.array([[h, w, 1],
                         [h, -w, 1],
                         [-h, -w, 1],
                         [-h, w, 1],
                         [0, 0, 1],
                         [-h * speed * 0.5, 0, 1]])
        bbox[:, :2] /= self.config.bounding_box_divisor
        bbox[:, :2] = bbox[:, [1, 0]]

        c, s = np.cos(yaw), np.sin(yaw)
        # use y x because coordinate is changed
        r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

        bbox = r1_to_world @ bbox.T
        bbox = bbox.T

        return bbox, brake
# class RlController:
#     def __init__(self, config, observer):
#         self.args = config.args
#         self.observer = observer
#
#     def test_rainbow(self):
#         env = TFuseEnv(self.observer)
#         args = self.args
#         args.state_shape = env.observation_space.shape or env.observation_space.n
#         args.action_shape = env.action_space.shape or env.action_space.n
#         if args.reward_threshold is None:
#             args.reward_threshold = 100
#         train_envs = TFuseEnv(self.observer)
#         # you can also use tianshou.env.SubprocVectorEnv
#         # train_envs = DummyVectorEnv(
#         #     [lambda: gym.make(args.task) for _ in range(args.training_num)]
#         # )
#         test_envs = None
#         # test_envs = DummyVectorEnv(
#         #     [lambda: gym.make(args.task) for _ in range(args.test_num)]
#         # )
#         # seed
#         np.random.seed(args.seed)
#         torch.manual_seed(args.seed)
#
#         # train_envs.seed(args.seed)
#         # test_envs.seed(args.seed)
#
#         # model
#
#         def noisy_linear(x, y):
#             return NoisyLinear(x, y, args.noisy_std)
#
#         net = Net(
#             args.state_shape,
#             args.action_shape,
#             hidden_sizes=args.hidden_sizes,
#             device=args.device,
#             softmax=True,
#             num_atoms=args.num_atoms,
#             dueling_param=({
#                                "linear_layer": noisy_linear
#                            }, {
#                                "linear_layer": noisy_linear
#                            }),
#         )
#         optim = torch.optim.Adam(net.parameters(), lr=args.lr)
#         policy = RainbowPolicy(
#             net,
#             optim,
#             args.gamma,
#             args.num_atoms,
#             args.v_min,
#             args.v_max,
#             args.n_step,
#             target_update_freq=args.target_update_freq,
#         ).to(args.device)
#         # buffer
#         if args.prioritized_replay:
#             buf = PrioritizedVectorReplayBuffer(
#                 args.buffer_size,
#                 # buffer_num=len(train_envs),
#                 buffer_num=1,
#                 alpha=args.alpha,
#                 beta=args.beta,
#                 weight_norm=True,
#             )
#         else:
#             # buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))
#             buf = VectorReplayBuffer(args.buffer_size, buffer_num=1)
#         # collector
#         train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
#         # test_collector = Collector(policy, test_envs, exploration_noise=True)
#         test_collector = None
#         # policy.set_eps(1)
#         train_collector.collect(n_step=args.batch_size * args.training_num)
#         # log
#         log_path = os.path.join(args.logdir, "Tfuse", "rainbow")
#         writer = SummaryWriter(log_path)
#         logger = TensorboardLogger(writer, save_interval=args.save_interval)
#
#         def save_best_fn(policy_):
#             torch.save(policy_.state_dict(), os.path.join(log_path, "policy.pth"))
#
#         def stop_fn(mean_rewards):
#             # return mean_rewards >= args.reward_threshold
#             return False
#
#         def train_fn(epoch, env_step):
#             # eps annealing, just a demo
#             if env_step <= 10000:
#                 policy.set_eps(args.eps_train)
#             elif env_step <= 50000:
#                 eps = args.eps_train - (env_step - 10000) / \
#                       40000 * (0.9 * args.eps_train)
#                 policy.set_eps(eps)
#             else:
#                 policy.set_eps(0.1 * args.eps_train)
#             # beta annealing, just a demo
#             if args.prioritized_replay:
#                 if env_step <= 10000:
#                     beta = args.beta
#                 elif env_step <= 50000:
#                     beta = args.beta - (env_step - 10000) / \
#                            40000 * (args.beta - args.beta_final)
#                 else:
#                     beta = args.beta_final
#                 buf.set_beta(beta)
#
#         def test_fn(epoch, env_step):
#             policy.set_eps(args.eps_test)
#
#         def save_checkpoint_fn(epoch, env_step, gradient_step):
#             # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
#             ckpt_path = os.path.join(log_path, "checkpoint.pth")
#             # Example: saving by epoch num
#             # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
#             torch.save(
#                 {
#                     "model": policy.state_dict(),
#                     "optim": optim.state_dict(),
#                 }, ckpt_path
#             )
#             buffer_path = os.path.join(log_path, "train_buffer.pkl")
#             pickle.dump(train_collector.buffer, open(buffer_path, "wb"))
#             return ckpt_path
#
#         if args.resume:
#             # load from existing checkpoint
#             print(f"Loading agent under {log_path}")
#             ckpt_path = os.path.join(log_path, "checkpoint.pth")
#             if os.path.exists(ckpt_path):
#                 checkpoint = torch.load(ckpt_path, map_location=args.device)
#                 policy.load_state_dict(checkpoint['model'])
#                 policy.optim.load_state_dict(checkpoint['optim'])
#                 print("Successfully restore policy and optim.")
#             else:
#                 print("Fail to restore policy and optim.")
#             buffer_path = os.path.join(log_path, "train_buffer.pkl")
#             if os.path.exists(buffer_path):
#                 train_collector.buffer = pickle.load(open(buffer_path, "rb"))
#                 print("Successfully restore buffer.")
#             else:
#                 print("Fail to restore buffer.")
#
#         # trainer
#         result = offpolicy_trainer(
#             policy,
#             train_collector,
#             test_collector,
#             args.epoch,
#             args.step_per_epoch,
#             args.step_per_collect,
#             args.test_num,
#             args.batch_size,
#             update_per_step=args.update_per_step,
#             train_fn=train_fn,
#             test_fn=test_fn,
#             stop_fn=stop_fn,
#             save_best_fn=save_best_fn,
#             logger=logger,
#             resume_from_log=args.resume,
#             save_checkpoint_fn=save_checkpoint_fn,
#         )

# class SafetyController:
#     def __init__(self, mem_size, config, device, dev=False):
#         self.config = config
#         self.bbox_memory = Queue(maxsize=mem_size)
#         self.mem_size = mem_size
#         self.device = device
#         self.dev = dev
#         self.save_path = "/home/transfuser/autonomous_car/transfuser-afa/my_save"
#
#     def forward(self, pred_wp, bboxes, i):
#         if (self.dev == False):
#             safety_action = False
#             if self.bbox_memory.qsize() < self.mem_size:
#                 rotated_bboxes = []
#                 for bbox in bboxes.detach().cpu().numpy():
#                     bbox = self.get_rotated_bbox(bbox[:7])
#                     rotated_bboxes.append(bbox)
#                 self.bbox_memory.put(rotated_bboxes)
#                 return pred_wp, safety_action
#             else:
#                 rotated_bboxes = []
#                 for bbox in bboxes.detach().cpu().numpy():
#                     bbox = self.get_rotated_bbox(bbox[:7])
#                     rotated_bboxes.append(bbox)
#                 self.bbox_memory.get()
#                 self.bbox_memory.put(rotated_bboxes)
#                 future_bboxes = self.guess_future_bboxes(bboxes)
#
#                 label = torch.zeros((1, 1, 7)).to(self.device)
#                 label[:, -1, 0] = 128.
#                 label[:, -1, 1] = 256.
#                 image = np.zeros(shape=(306, 512, 3))
#                 image = self.draw_bboxes(rotated_bboxes, image, color=(255, 0, 0), brake_color=(0, 0, 255))
#
#                 image, points_in_img = self.draw_waypoints(label[0], pred_wp[0:0 + 1, :2], image, color=(255, 0, 0))
#
#                 new_wp, is_dangerous = self.modify_waypoints(pred_wp, points_in_img, future_bboxes)
#                 cv2.imwrite(str(self.save_path + ("/%d.png" % (i // 2))), image)
#
#                 image_2 = np.zeros(shape=(306, 512, 3))
#                 image_2 = self.draw_bboxes(future_bboxes, image_2, color=(0, 255, 0), brake_color=(0, 0, 255))
#                 image_2, _ = self.draw_waypoints(label[0], new_wp[0:0 + 1, :2], image_2, color=(255, 0, 0))
#
#                 cv2.imwrite(str(self.save_path + ("/%d-2.png" % (i // 2))), image_2)
#                 if is_dangerous:
#                     cv2.imwrite(str(self.save_path + ("/danger/%d.png" % (i // 2))), image)
#                     cv2.imwrite(str(self.save_path + ("/danger/%d-2.png" % (i // 2))), image_2)
#                     safety_action = True
#                 return new_wp, safety_action
#         else:
#             if self.bbox_memory.qsize() < self.mem_size:
#                 rotated_bboxes = []
#                 for bbox in bboxes.detach().cpu().numpy():
#                     bbox = self.get_rotated_bbox(bbox[:7])
#                     rotated_bboxes.append(bbox)
#                 self.bbox_memory.put(rotated_bboxes)
#             else:
#                 if (self.config.debug):
#                     print(pred_wp)
#                     print(bboxes)
#                 rotated_bboxes = []
#                 for bbox in bboxes.detach().cpu().numpy():
#                     bbox = self.get_rotated_bbox(bbox[:7])
#                     rotated_bboxes.append(bbox)
#                 image = np.zeros(shape=(306, 512, 3))
#                 images = self.draw_bboxes(rotated_bboxes, image, color=(255, 0, 0), brake_color=(0, 0, 255))
#                 self.bbox_memory.get()
#                 self.bbox_memory.put(rotated_bboxes)
#                 future_bboxes = self.guess_future_bboxes(bboxes)
#
#                 image_2 = np.zeros(shape=(306, 512, 3))
#                 image_2 = self.draw_bboxes(future_bboxes, image_2, color=(0, 255, 0), brake_color=(0, 255, 0))
#
#                 cv2.imwrite(str(self.save_path + ("/%d.png" % (i // 2))), images)
#                 cv2.imwrite(str(self.save_path + ("/%d-2.png" % (i // 2))), image_2)
#             # print(pred_wp)
#             # print(bboxes)
#
#     def guess_future_bboxes(self, rotated_bboxes):
#         # bbox_center_memory = []
#         bbox_memory_list = copy.copy(list(self.bbox_memory.queue))
#         future_bboxes = bbox_memory_list[-1]
#         # center calculation
#         # for moment in list(self.bbox_memory.queue):
#         #     centers = []
#         #     for bbox in moment:
#         #         x = 0
#         #         y = 0
#         #         x += bbox[0][0]
#         #         y += bbox[0][1]
#         #         centers.append([x/4, y/4])
#         #     bbox_center_memory.append(centers)
#
#         center_diffs = []
#         lowest_center_diffs = []
#
#         for i, bbox_ in enumerate(future_bboxes):
#             bbox = bbox_[0]
#             center_diffs.append([])
#             lowest_center_diff = (None, None)  # diff, j
#             for j, past_bbox_ in enumerate(bbox_memory_list[-2]):
#                 past_bbox = past_bbox_[0]
#
#                 center_diff = ((bbox[5][0] - past_bbox[5][0]) ** 2 + (bbox[5][1] - past_bbox[5][1]) ** 2) ** 0.5
#                 center_diffs[i].append(center_diff)
#
#                 if (lowest_center_diff[0] is None):
#                     lowest_center_diff = (center_diff, j)
#                 elif (lowest_center_diff[0] > center_diff):
#                     lowest_center_diff = (center_diff, j)
#             lowest_center_diffs.append(lowest_center_diff)
#
#         for i, (lowest_diff, j) in enumerate(lowest_center_diffs):
#             if lowest_diff is not None:
#                 if lowest_diff < self.config.bbox_losed_threshold:
#                     for point in range(6):
#                         for x_or_y in range(2):
#                             future_bboxes[i][0][point][x_or_y] += future_bboxes[i][0][point][x_or_y] - \
#                                                                   bbox_memory_list[-2][j][0][point][x_or_y]
#
#         return future_bboxes
#
#     def modify_waypoints(self, pred_wp, wp_converted, bboxes):
#         is_dangerous = False
#         wp_1 = Point(wp_converted[0][0][0], wp_converted[0][0][1])
#         wp_2 = Point(wp_converted[0][1][0], wp_converted[0][1][1])
#         idx = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
#         new_wp = copy.copy(pred_wp)
#         for bbox in bboxes:
#             bbox = bbox[0]
#             for id in idx:
#                 p_1 = Point(bbox[id[0]][0], bbox[id[0]][1])
#                 p_2 = Point(bbox[id[1]][0], bbox[id[1]][1])
#                 if Point.doIntersect(wp_1, wp_2, p_1, p_2):
#                     new_wp[0][1] = pred_wp[0][0]
#                     is_dangerous = True
#                     print("dangerous move detected")
#                     break
#
#         return new_wp, is_dangerous
#
#     def draw_bboxes(self, bboxes, image, color=(255, 0, 0), brake_color=(0, 0, 255)):
#         idx = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
#         for bbox, brake in bboxes:
#             bbox = bbox.astype(np.int32)[:, :2]
#             for s, e in idx:
#                 if brake >= self.config.draw_brake_threshhold:
#                     color = brake_color
#                 else:
#                     color = color
#                 # brake is true while still have high velocity
#                 cv2.line(image, tuple(bbox[s]), tuple(bbox[e]), color=color, thickness=1)
#         return image
#
#     def draw_waypoints(self, label, waypoints, image, color=(255, 255, 255)):
#         waypoints = waypoints.detach().cpu().numpy()
#         label = label.detach().cpu().numpy()
#         points_in_img = []
#
#         for bbox, points in zip(label, waypoints):
#             x, y, w, h, yaw, speed, brake = bbox
#             c, s = np.cos(yaw), np.sin(yaw)
#             # use y x because coordinate is changed
#             r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
#
#             points[:, 0] *= -1
#             points = points * self.config.pixels_per_meter
#             points = points[:, [1, 0]]
#             points = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1)
#
#             points = r1_to_world @ points.T
#             points = points.T
#
#             points_to_draw = []
#             for point in points[:, :2]:
#                 points_to_draw.append(point.copy())
#                 point = point.astype(np.int32)
#                 cv2.circle(image, tuple(point), radius=3, color=color, thickness=3)
#             points_in_img.append(points_to_draw)
#         return image, points_in_img
#
#     def get_rotated_bbox(self, bbox):
#         x, y, w, h, yaw, speed, brake = bbox
#
#         bbox = np.array([[h, w, 1],
#                          [h, -w, 1],
#                          [-h, -w, 1],
#                          [-h, w, 1],
#                          [0, 0, 1],
#                          [-h * speed * 0.5, 0, 1]])
#         bbox[:, :2] /= self.config.bounding_box_divisor
#         bbox[:, :2] = bbox[:, [1, 0]]
#
#         c, s = np.cos(yaw), np.sin(yaw)
#         # use y x because coordinate is changed
#         r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
#
#         bbox = r1_to_world @ bbox.T
#         bbox = bbox.T
#
#         return bbox, brake
#
#
# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#
#     # Given three collinear points p, q, r, the function checks if
#     # point q lies on line segment 'pr'
#     @staticmethod
#     def onSegment(p, q, r):
#         if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
#                 (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
#             return True
#         return False
#
#     @staticmethod
#     def orientation(p, q, r):
#         # to find the orientation of an ordered triplet (p,q,r)
#         # function returns the following values:
#         # 0 : Collinear points
#         # 1 : Clockwise points
#         # 2 : Counterclockwise
#
#         # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
#         # for details of below formula.
#
#         val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
#         if (val > 0):
#
#             # Clockwise orientation
#             return 1
#         elif (val < 0):
#
#             # Counterclockwise orientation
#             return 2
#         else:
#
#             # Collinear orientation
#             return 0
#
#     # The main function that returns true if
#     # the line segment 'p1q1' and 'p2q2' intersect.
#     @staticmethod
#     def doIntersect(p1, q1, p2, q2):
#         # Find the 4 orientations required for
#         # the general and special cases
#         o1 = Point.orientation(p1, q1, p2)
#         o2 = Point.orientation(p1, q1, q2)
#         o3 = Point.orientation(p2, q2, p1)
#         o4 = Point.orientation(p2, q2, q1)
#
#         # General case
#         if ((o1 != o2) and (o3 != o4)):
#             return True
#
#         # Special Cases
#
#         # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
#         if ((o1 == 0) and Point.onSegment(p1, p2, q1)):
#             return True
#
#         # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
#         if ((o2 == 0) and Point.onSegment(p1, q2, q1)):
#             return True
#
#         # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
#         if ((o3 == 0) and Point.onSegment(p2, p1, q2)):
#             return True
#
#         # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
#         if ((o4 == 0) and Point.onSegment(p2, q1, q2)):
#             return True
#
#         # If none of the cases
#         return False

# class SafetyController:
#     def __init__(self, mem_size, config, device, dev=False):
#         self.config = config
#         self.bbox_memory = Queue(maxsize=mem_size)
#         self.mem_size = mem_size
#         self.device = device
#         self.dev=dev
#         self.save_path="/home/transfuser/autonomous_car/transfuser-afa/my_save"
#
#     def forward(self, pred_wp, bboxes, i):
#         if (self.dev==False):
#             if self.bbox_memory.qsize() < self.mem_size:
#                 rotated_bboxes = []
#                 for bbox in bboxes.detach().cpu().numpy():
#                     bbox = self.get_rotated_bbox(bbox[:7])
#                     rotated_bboxes.append(bbox)
#                 self.bbox_memory.put(rotated_bboxes)
#                 return pred_wp
#             else:
#                 if (self.config.debug):
#                     print("-")
#                     # print(pred_wp)
#                     # print(bboxes)
#                 rotated_bboxes = []
#                 for bbox in bboxes.detach().cpu().numpy():
#                     bbox = self.get_rotated_bbox(bbox[:7])
#                     rotated_bboxes.append(bbox)
#                 self.bbox_memory.get()
#                 self.bbox_memory.put(rotated_bboxes)
#
#                 label = torch.zeros((1, 1, 7)).to(self.device)
#                 label[:, -1, 0] = 128.
#                 label[:, -1, 1] = 256.
#                 image = np.zeros(shape=(306, 512, 3))
#
#                 images = self.draw_bboxes(rotated_bboxes, image, color=(255, 0, 0), brake_color=(0, 0, 255))
#                 future_bboxes = self.guess_future_bboxes(bboxes)
#                 images, image_wp, addition = self.draw_waypoints(label[0], pred_wp[0:0 + 1, :2], images,
#                                                        color=(255, 0, 0))
#                 new_wp = self.modify_waypoints(label[0], image_wp, future_bboxes, addition)
#                 cv2.imwrite(str(self.save_path + ("/%d.png" % (i // 2))), images)
#             return new_wp
#         else:
#             if self.bbox_memory.qsize() < self.mem_size:
#                 rotated_bboxes = []
#                 for bbox in bboxes.detach().cpu().numpy():
#                     bbox = self.get_rotated_bbox(bbox[:7])
#                     rotated_bboxes.append(bbox)
#                 self.bbox_memory.put(rotated_bboxes)
#             else:
#                 if (self.config.debug):
#                     print("-")
#                     # print(pred_wp)
#                     # print(bboxes)
#                 rotated_bboxes = []
#                 for bbox in bboxes.detach().cpu().numpy():
#                     bbox = self.get_rotated_bbox(bbox[:7])
#                     rotated_bboxes.append(bbox)
#
#                 label = torch.zeros((1, 1, 7)).to(self.device)
#                 label[:, -1, 0] = 128.
#                 label[:, -1, 1] = 256.
#
#                 image = np.zeros(shape=(306, 512, 3))
#                 images = self.draw_bboxes(rotated_bboxes, image, color=(255, 0, 0), brake_color=(0, 0, 255))
#                 images, image_wp, addition = self.draw_waypoints(label[0], pred_wp[0:0 + 1, :2], images,
#                                              color=(255, 0, 0))  # First two, relevant waypoints in blu
#
#                 self.bbox_memory.get()
#                 self.bbox_memory.put(rotated_bboxes)
#                 future_bboxes = self.guess_future_bboxes(bboxes)
#
#                 image_2  = np.zeros(shape=(306,512,3))
#                 images_2 = self.draw_bboxes(future_bboxes, image_2, color=(0, 255, 0), brake_color=(0, 0, 255))
#                 images_2 = self.draw_waypoints(label[0], pred_wp[0:0 + 1, :2], images_2,
#                                              color=(0, 0, 255))
#
#
#
#                 cv2.imwrite(str(self.save_path + ("/%d.png" % (i // 2))), images)
#                 cv2.imwrite(str(self.save_path + ("/%d-2.png" % (i // 2))), images_2)
#             #print(pred_wp)
#             #print(bboxes)
#         return ["dummy"]
#
#     def guess_future_bboxes(self, rotated_bboxes):
#         #bbox_center_memory = []
#         bbox_memory_list = copy.copy(list(self.bbox_memory.queue))
#         future_bboxes = bbox_memory_list[-1]
#         #center calculation
#         # for moment in list(self.bbox_memory.queue):
#         #     centers = []
#         #     for bbox in moment:
#         #         x = 0
#         #         y = 0
#         #         x += bbox[0][0]
#         #         y += bbox[0][1]
#         #         centers.append([x/4, y/4])
#         #     bbox_center_memory.append(centers)
#
#         center_diffs = []
#         lowest_center_diffs = []
#
#         for i, bbox_ in enumerate(future_bboxes):
#             bbox = bbox_[0]
#             center_diffs.append([])
#             lowest_center_diff = (None, None) #diff, j
#             for j, past_bbox_ in enumerate(bbox_memory_list[-2]):
#                 past_bbox = past_bbox_[0]
#
#                 center_diff = ((bbox[5][0] - past_bbox[5][0]) ** 2 + (bbox[5][1] - past_bbox[5][1]) ** 2) ** 0.5
#                 center_diffs[i].append(center_diff)
#
#                 if (lowest_center_diff[0] is None):
#                     lowest_center_diff = (center_diff, j)
#                 elif (lowest_center_diff[0] > center_diff):
#                     lowest_center_diff = (center_diff, j)
#             lowest_center_diffs.append(lowest_center_diff)
#
#         for i, (lowest_diff, j) in enumerate(lowest_center_diffs):
#             if lowest_diff is not None:
#                 if lowest_diff < self.config.bbox_losed_threshold:
#                     for point in range(6):
#                         for x_or_y in range(2):
#                             future_bboxes[i][0][point][x_or_y] += future_bboxes[i][0][point][x_or_y] - bbox_memory_list[-2][j][0][point][x_or_y]
#
#         # bbox[0][0] += bbox[0][0] - bbox_memory_list[-2][i][0][0][0]
#         # bbox[0][1] += bbox[0][1] - bbox_memory_list[-2][i][0][0][1]
#         return future_bboxes
#
#     def modify_waypoints(self, label, pred_wp, bboxes, addition):
#         waypoints = []
#         for bbox, points in zip(label, pred_wp):
#             x, y, w, h, yaw, speed, brake = bbox
#             c, s = np.cos(yaw), np.sin(yaw)
#
#             r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
#
#             points = points.T
#             points = (np.linalg.inv(r1_to_world) @ points).T
#             points = points[:, :-1]
#             points = np.concatenate((points, addition), axis=1)
#             points = points / self.config.pixels_per_meter
#             points[:, 0] *= -1
#             waypoints.append(points)
#         new_wp = torch.Tensor(np.array(waypoints)).cuda()
#         return new_wp
#
#     def draw_bboxes(self, bboxes, image, color=(255, 0, 0), brake_color=(0, 0, 255)):
#         idx = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
#         for bbox, brake in bboxes:
#             bbox = bbox.astype(np.int32)[:, :2]
#             for s, e in idx:
#                 if brake >= self.config.draw_brake_threshhold:
#                     color = brake_color
#                 else:
#                     color = color
#                 # brake is true while still have high velocity
#                 cv2.line(image, tuple(bbox[s]), tuple(bbox[e]), color=color, thickness=1)
#         return image
#
#     def draw_waypoints_2(self, label, waypoints, image, color=(255, 255, 255)):
#         waypoints = waypoints.detach().cpu().numpy()
#         label = label.detach().cpu().numpy()
#
#         img_points = []
#         for bbox, points in zip(label, waypoints):
#             x, y, w, h, yaw, speed, brake = bbox
#             c, s = np.cos(yaw), np.sin(yaw)
#             # use y x because coordinate is changed
#             r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
#
#             # convert to image space
#             # need to negate y componet as we do for lidar points
#             # we directly construct points in the image coordiante
#             # for lidar, forward +x, right +y
#             #            x
#             #            +
#             #            |
#             #            |
#             #            |---------+y
#             #
#             # for image, ---------> x
#             #            |
#             #            |
#             #            +
#             #            y
#
#             points[:, 0] *= -1
#             points = points * self.config.pixels_per_meter
#             addition = points[:, 2:]
#             points = points[:, [1, 0]]
#             points = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1)
#
#             points = r1_to_world @ points.T
#             points = points.T
#             img_points.append(points)
#             points_to_draw = []
#             for point in points[:, :2]:
#                 points_to_draw.append(point.copy())
#                 point = point.astype(np.int32)
#                 cv2.circle(image, tuple(point), radius=3, color=color, thickness=3)
#         return image, img_points, addition
#
#
#     def get_rotated_bbox(self, bbox):
#         x, y, w, h, yaw, speed, brake = bbox
#
#         bbox = np.array([[h, w, 1],
#                          [h, -w, 1],
#                          [-h, -w, 1],
#                          [-h, w, 1],
#                          [0, 0, 1],
#                          [-h * speed * 0.5, 0, 1]])
#         bbox[:, :2] /= self.config.bounding_box_divisor
#         bbox[:, :2] = bbox[:, [1, 0]]
#
#         c, s = np.cos(yaw), np.sin(yaw)
#         # use y x because coordinate is changed
#         r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
#
#         bbox = r1_to_world @ bbox.T
#         bbox = bbox.T
#
#         return bbox, brake

# class RlController:
#     def __init__(self, config, device, state_size, action_size):
#         self.config = config
#         self.device = device
#         self.agent = Agent(state_size, action_size, self.config.args, self.device)
#         self.writer = SummaryWriter("runs/" + self.config.rl_info)
#
#         self.scores = []
#         self.current_frame = 0
#         self.action_low = -1
#         self.action_high = 1
#         self.old_state = self.get_state()
#         self.old_action = 0
#         self.score = 0
#
#         self.scores_window = deque(maxlen=100)  # last 100 scores
#         self.i_episode = 1
#         self.frames = self.config.args.frames // self.config.args.worker
#         worker = self.config.args.worker
#         self.ERE = self.config.args.ere
#
#         if self.ERE:
#             self.episode_K = 0
#             self.eta_0 = 0.996
#             self.eta_T = 1.0
#             # episodes = 0
#             self.max_ep_len = 500  # original = 1000
#             self.c_k_min = 2500  # original = 5000
#
#         if self.config.args.saved_model is not None:
#             self.agent.actor_local.load_state_dict(torch.load(self.config.args.saved_model))
#
#     def get_reward(self):
#         return 0
#
#     def get_action(self, current_state, done):
#         old_reward = self.get_reward()
#         self.agent.step(self.old_state, self.old_action, old_reward, current_state, done)
#         self.current_frame += 1
#         action = self.agent.act(current_state)
#         action_v = np.clip(action, self.action_low, self.action_high)
#         old_state = current_state
#         old_action = action_v
#         if self.ERE:
#             self.eta_t = self.eta_0 + (self.eta_T - self.eta_0) * (self.current_frame / (self.frames + 1))
#             self.episode_K += 1
#         self.score += np.mean(old_reward)
#         self.if_done(done)
#         return action_v
#
#     def if_done(self, is_done):
#         if is_done:
#             if self.ERE:
#                 for k in range(1, self.episode_K):
#                     c_k = max(int(self.agent.memory.__len__() * self.eta_t ** (k * (self.max_ep_len / self.episode_K))),
#                               self.c_k_min)
#                     self.agent.ere_step(c_k)
#             self.scores_window.append(self.score)  # save most recent score
#             self.scores.append(self.score)  # save most recent score
#             self.writer.add_scalar("Average100", np.mean(self.scores_window), self.current_frame * 1)
#             print('\rEpisode {}\tFrame: [{}/{}]\t Reward: {:.2f} \tAverage100 Score: {:.2f}' \
#                   .format(self.i_episode * 1, self.current_frame * 1, self.frames, self.score,
#                           np.mean(self.scores_window)), end="", flush=True)
#
#             # if i_episode % 100 == 0:
#             #    print('\rEpisode {}\tFrame \tReward: {}\tAverage100 Score: {:.2f}'.format(i_episode*worker, frame*worker, round(eval_reward,2), np.mean(scores_window)), end="", flush=True)
#             self.i_episode += 1
#             state = self.get_state()
#             self.score = 0
#             self.episode_K = 0
#
#     def agent_step(self, state, action, reward, next_state, done):
#         self.agent.step(state, action, reward, next_state, done, self.current_frame, self.ERE)
#
#     def save_policy(self):
#         torch.save(self.agent.actor_local.state_dict(), 'runs/' + self.config.args.info + ".pth")
#
#     def save_parameter(self):
#         with open('runs/' + self.config.args.info + ".json", 'w') as f:
#             json.dump(self.config.args.__dict__, f, indent=2)
