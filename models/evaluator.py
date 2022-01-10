import torch.nn as nn
from opts import *
import numpy as np
import copy
import cv2


def get_mask(x, boxes, img_size):
    """
    :param x: [4, 480, 28, 14, 14]
    :param boxes:
    :param img_size: (H,W) == > (640,360)
    :return:
    """
    # img_size : (H, W)
    # mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3], x.shape[4]))  # (B,T,H,W)
    # boxes  # [B, 103, 8]
    boxes = copy.deepcopy(boxes.numpy())

    t_ratio = 4  # 4
    H_r, W_r = img_size[0] / x.shape[3], img_size[1] / x.shape[4]
    boxes[:, :, 0], boxes[:, :, 2], boxes[:, :, 4], boxes[:, :, 6] = \
        boxes[:, :, 0] / H_r, boxes[:, :, 2] / H_r, boxes[:, :, 4] / H_r, boxes[:, :, 6] / H_r
    boxes[:, :, 1], boxes[:, :, 3], boxes[:, :, 5], boxes[:, :, 7] = \
        boxes[:, :, 1] / W_r, boxes[:, :, 3] / W_r, boxes[:, :, 5] / W_r, boxes[:, :, 7] / W_r

    # Get mask of 103
    mask_b = []  # [B, 103, 14, 14]
    for b in range(x.shape[0]):
        mask_v = []
        for t in range(103):
            img = np.zeros((x.shape[3], x.shape[4]), dtype=np.uint8)
            position = np.array([[boxes[b, t, 0], boxes[b, t, 1]],
                                 [boxes[b, t, 2], boxes[b, t, 3]],
                                 [boxes[b, t, 4], boxes[b, t, 5]],
                                 [boxes[b, t, 6], boxes[b, t, 7]]])
            # if t > 60:  # With big mask
            #     position = np.array([[3, 3], [10, 3], [10, 10], [3, 10]])
            cv2.fillConvexPoly(img, position, 1)
            mask_v.append(np.array(img))
        mask_b.append(mask_v)
    mask_b = np.array(mask_b)  # [4,103,14,14]

    # Temporal pooling of mask  ==>  (B,T,H,W)
    mask_final = []
    for b in range(x.shape[0]):  # Every batch
        mask_crt = []
        for start_point in segment_points:  # [0, 16, 32, 48, 64, 80, 96]
            mask_clip = np.zeros((x.shape[3], x.shape[4]), dtype=np.uint8)
            for i in range(16):  # 16 frames
                mask_clip += mask_b[b, start_point + i]
                if (i + 1) % t_ratio == 0:  # sum up 4 images
                    mask_crt.append(mask_clip)
                    mask_clip = np.zeros((x.shape[3], x.shape[4]), dtype=np.uint8)
        mask_final.append(mask_crt)
    return np.array(mask_final).astype(np.bool).astype(int)  # (B,T,H,W)


class MLP_block(nn.Module):

    def __init__(self, output_dim):
        super(MLP_block, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.layer1 = nn.Linear(feature_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, output_dim)

        self.dp = nn.Dropout(0.5)

        self.getprob = nn.Sigmoid()

    def forward(self, x, dp_flag):
        if dp_flag is False:
            x = self.activation(self.layer1(x))
            x = self.activation(self.layer2(x))
        elif dp_flag is True:
            x = self.activation(self.dp(self.layer1(x)))
            x = self.activation(self.dp(self.layer2(x)))
        output = self.getprob(self.layer3(x))
        return output


class Evaluator(nn.Module):

    def __init__(self, output_dim, model_type='USDL', num_judges=None):
        super(Evaluator, self).__init__()

        self.model_type = model_type

        if model_type == 'USDL':
            self.evaluator = MLP_block(output_dim=output_dim)
        else:
            assert num_judges is not None, 'num_judges is required in MUSDL'
            self.evaluator = nn.ModuleList([MLP_block(output_dim=output_dim) for _ in range(num_judges)])

    def forward(self, feats_avg, args):  # data: NCTHW

        if self.model_type == 'USDL':
            probs = self.evaluator(feats_avg, dp_flag=args.dp)  # Nxoutput_dim
        else:
            probs = [evaluator(feats_avg) for evaluator in self.evaluator]  # len=num_judges
        return probs

