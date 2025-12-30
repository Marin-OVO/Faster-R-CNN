import torch
from torch import nn
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple
from .fasterrcnn import FasterRCNN
from model.modules import PFIM, PGDP, CBAM


class EnhancedFasterRCNN(FasterRCNN):

    def __init__(self, backbone, num_classes=None,
                 pfim_ich=256, pfim_hch=128,
                 pgdp_ich=256, pgdp_hch=256,
                 lambda_ie=1.0,
                 lambda_pred=1.0,

                 **kwargs):

        super().__init__(backbone, num_classes, **kwargs)

        self.pfim = PFIM(ich=pfim_ich, hch=pfim_hch)
        self.pgdp = PGDP(ich=pgdp_ich, hch=pgdp_hch)

        self.cbam1 = CBAM(inc=pfim_ich, kernel_size=7)
        self.cbam2 = CBAM(inc=pfim_ich, kernel_size=7)

        self.lambda_ie = lambda_ie
        self.lambda_pred = lambda_pred

        self.cached_features = None

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor "
                                         "of shape [N, 4], got {:}.".format(boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        feature_list = list(features.values())
        p2 = feature_list[0] if len(feature_list) > 0 else None
        p3 = feature_list[1] if len(feature_list) > 1 else None
        p4 = feature_list[2] if len(feature_list) > 2 else None

        L_IE = torch.tensor(0.0, device=images.tensors.device)
        L_pred = torch.tensor(0.0, device=images.tensors.device)

        if p2 is not None and p3 is not None and p4 is not None:
            pfim_out = self.pfim(p2)
            mu = pfim_out['mu']
            sigma = pfim_out['sigma']
            L_IE = pfim_out['L_IE']

            boxes = [t['boxes'] for t in targets] if (targets and self.training) else None
            img_size = images.tensors.shape[-2:]

            pgdp_out = self.pgdp(p4, p3, p2, sigma, boxes, img_size)
            L_pred = pgdp_out['L_pred']
            Mpd2 = pgdp_out['Mpd2']

            y1 = p2 * (sigma + 1)
            y2 = p2 * (Mpd2 + 1)

            y1_att = self.cbam1(y1)
            y2_att = self.cbam2(y2)

            p2_enhanced = y1_att + y2_att

            # p2_enhanced = p2 + 0.5 * (y1_att + y2_att)

            features['0'] = p2_enhanced

        proposals, proposal_losses = self.rpn(images, features, targets)

        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )

        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            losses['loss_ie'] = self.lambda_ie * L_IE
            losses['loss_pred'] = self.lambda_pred * L_pred

        if torch.jit.is_scripting():
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)