import os
import sys

sys.path.append('../')

import torch
import torch.nn as nn

from scipy import stats
from tqdm import tqdm
import itertools

from models.i3d import InceptionI3d
from models.i3d import TSA_Module, NONLocalBlock3D
from models.evaluator import Evaluator, get_mask

from opts import *
from dataset import VideoDataset
from config import get_parser
from logger import Logger

from utils import *

from thop import profile
from thop import clever_format

def get_models(args):
    """
    Get the i3d backbone and the evaluator with parameters moved to GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    i3d = InceptionI3d().cuda()
    i3d.load_state_dict(torch.load(i3d_pretrained_path))
    
    evaluator = Evaluator(output_dim=output_dim['USDL'], model_type='USDL').cuda()

    if len(args.gpu.split(',')) > 1:
        i3d = nn.DataParallel(i3d)
        evaluator = nn.DataParallel(evaluator)
    return i3d, evaluator


def compute_score(model_type, probs, data):
    if model_type == 'USDL':
        pred = probs.argmax(dim=-1) * (label_max / (output_dim['USDL'] - 1))
    else:
        # calculate expectation & denormalize & sort
        judge_scores_pred = torch.stack([prob.argmax(dim=-1) * judge_max / (output_dim['MUSDL'] - 1)
                                         for prob in probs], dim=1).sort()[0]  # N, 7

        # keep the median 3 scores to get final score according to the rule of diving
        pred = torch.sum(judge_scores_pred[:, 2:5], dim=1) * data['difficulty'].cuda()
    return pred


def compute_loss(model_type, criterion, probs, data):
    if model_type == 'USDL':
        loss = criterion(torch.log(probs), data['soft_label'].cuda())
    else:
        loss = sum([criterion(torch.log(probs[i]), data['soft_judge_scores'][:, i].cuda()) for i in range(num_judges)])
    return loss


def compute_acc(pred_scores, true_scores):
    pred_scores, true_scores = np.array(pred_scores).astype(int), np.array(true_scores).astype(int)
    # TP predict 1 label 1
    TP = sum((pred_scores == 1) & (true_scores == 1))
    # TN predict 0 label 0
    TN = sum((pred_scores == 0) & (true_scores == 0))
    # FN predict 0 label 1
    FN = sum((pred_scores == 0) & (true_scores == 1))
    # FP predict 1 label 0
    FP = sum((pred_scores == 1) & (true_scores == 0))
    # print('TP, TN, FN, FP: ', TP, TN, FN, FP)
    # print(pred_scores, true_scores)

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc


def get_dataloaders(args):
    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', args),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn)

    dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', args),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
    return dataloaders


def main(dataloaders, i3d, evaluator, args):
    # print configuration
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)

    # TSA_Block:
    if args.TSA is True:
        TSA_module = TSA_Module(I3D_ENDPOINTS[args.TSA_pos][1], bn_layer=True).cuda()

    # Load pre-trained weights:
    if args.pt_w is not None:
        weights = torch.load(args.pt_w)
        # param of i3d
        i3d.load_state_dict(weights['i3d'])
        # param of evaluator
        evaluator.load_state_dict(weights['evaluator'])
        # param of TSA module
        if args.TSA is True:
            TSA_module.load_state_dict(weights['TSA_module'])
        print('----- Pre-trained weight loaded from ' + args.pt_w)

    true_scores, pred_scores, keys_list = [], [], []

    # Set train/test mode of the whole model
    i3d.eval()
    evaluator.eval()
    if args.TSA is True:
        TSA_module.eval()
    torch.set_grad_enabled(False)

    # Testing
    for data in tqdm(dataloaders['test']):
        true_scores.extend(data['final_score'].numpy())
        videos = data['video'].cuda()
        boxes = data['boxes']  # [B,103,8]
        keys_list.extend(data['keys'])
        videos.transpose_(1, 2)  # N, C, T, H, W

        batch_size, C, frames, H, W = videos.shape
        clip_feats = torch.empty(batch_size, 7, feature_dim).cuda()

        ### Forward
        if args.TSA is False:
            for i in range(6):  # [4,3,103,224,224]=>[4,1024]
                clip_feats[:, i] = i3d(videos[:, :, 16 * i:16 * i + 16, :, :], args).squeeze(2)
            clip_feats[:, 6] = i3d(videos[:, :, -16:, :, :], args).squeeze(2)
        else:
            # ####
            # Stage 1 of I3D
            # ####
            feats_tsa = []
            for i in range(6):  # [0,TSA_loc]
                feats_tsa.append(i3d(videos[:, :, 16 * i:16 * i + 16, :, :], args, stage=1))  # [B,C,T,H,W]
            feats_tsa.append(i3d(videos[:, :, -16:, :, :], args, stage=1))
            ckpt_C, ckpt_T, ckpt_S = feats_tsa[0].shape[1:4]

            # ####
            # Feature enhancement stage (FLOPS counter)
            # ####
            feats_tsa = torch.cat(feats_tsa, dim=2).cuda()  # [B,C,T*10,H,W] Merge time
            mask = get_mask(x=feats_tsa, boxes=boxes, img_size=(W_img, H_img))  # Get box : [B,T,H,W]
            feats_tsa = TSA_module(feats_tsa, mask)

            feats_tsa = feats_tsa.view(batch_size, ckpt_C, ckpt_T, 7, ckpt_S, ckpt_S)
            feats_tsa = feats_tsa.permute(0, 1, 2, 4, 5, 3).contiguous()  # [4,192,8,28,28,10]

            # ####
            # Stage 2 of I3D
            # ####
            for i in range(6):  # (TSA_loc,-1]    [4,3,103,224,224]=>[4,1024]
                clip_feats[:, i] = i3d(feats_tsa[:, :, :, :, :, i], args, stage=2).squeeze(2)
            clip_feats[:, 6] = i3d(feats_tsa[:, :, :, :, :, 6], args, stage=2).squeeze(2)
            del feats_tsa

        probs = evaluator(clip_feats.mean(1), args)  # [4,1]
        pred_scores.extend([i for i in probs.cpu().detach().numpy().reshape((-1,))])  # probs

    # Test:
    pred_cls = []
    for i in range(len(pred_scores)):
        pred_cls.append(1 if pred_scores[i] > 0.5 else 0)

    acc = compute_acc(pred_cls, true_scores)
    print('\t %s Acc: %.2f ' % ('Test', acc * 100))


if __name__ == '__main__':

    args = get_parser().parse_args()

    # Create Experiments dirs
    if not os.path.exists('./Exp'):
        os.mkdir('./Exp')
    args.model_path = './Exp/' + args.model_path
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    init_seed(args)

    base_logger = get_logger(f'{args.model_path}/{args.type}.log', args.log_info)
    i3d, evaluator = get_models(args)
    dataloaders = get_dataloaders(args)

    main(dataloaders, i3d, evaluator, args)
