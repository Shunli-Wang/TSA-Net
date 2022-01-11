import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='Model_1', help='one model per file')
    parser.add_argument('--gpu', type=str, default='1', help='id of gpu device(s) to be used')
    parser.add_argument('--log_info', type=str, default='Exp1')
    parser.add_argument('--pt_w', type=str, default=None, help='pre-trained models')
    parser.add_argument('--dp', action="store_true", help='Dropout layer')

    # TSA:
    parser.add_argument('--TSA', action="store_true", help='NL Block')
    parser.add_argument('--TSA_pos', type=int, default=7, help='location of NL Block')

    parser.add_argument('--temporal_aug', type=int, default=0, help='the max of rand temporal shift, ranges 0 to 6')
    parser.add_argument('--std', type=float, default=5, help='standard deviation for G distribution learning')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 weight decay')
    parser.add_argument('--seed', type=int, default=1, help='manual seed')
    parser.add_argument('--save', action='store_true', default=True, help='if set true, save the best model')

    # Batchsize and epochs
    parser.add_argument('--num_epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=8, help='number of subprocesses for dataloader')
    parser.add_argument('--train_batch_size', type=int, default=4, help='batch size for training phase')
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size for test phase')

    # Dataset
    parser.add_argument('--dataset_path', type=str, default='./data/FRFS', help='path to FRFS dataset')

    return parser

# 第一步：克隆Repo，配置环境
# 第二步：创建data文件夹，下载pt文件，下载FRFS数据集，添加ln链接，保证格式如下

# scipy 1.7.3
# pandas
# xlrd-2.0.1
# thop

# python train.py --gpu 0 --model_path TSA-USDL --TSA
# python test.py --gpu 0 --pt_w TSA-USDL --TSA

# python train.py --gpu 0 --model_path USDL
# python test.py --gpu 0 --pt_w USDL 

# numpy                   1.21.2
# opencv-python           4.5.3.56
# pandas                  1.3.5
# pytz                    2021.3
# PyYAML                  6.0
# scipy                   1.7.3
# tensorboardX            2.4
# thop                    0.0.31.post2005241907
# torch                   1.9.1
# torchaudio              0.9.0a0+a85b239
# torchvision             0.10.1
# tqdm                    4.62.3
