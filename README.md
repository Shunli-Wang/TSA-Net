# Tube Self-Attention Network (TSA-Net)
This repository contains the PyTorch implementation for paper __TSA-Net: Tube Self-Attention Network for Action Quality Assessment__ (ACM-MM'21 Oral) 

\[[arXiv](???)\]
\[[supp](https://drive.google.com/file/d/1JeAgMqOwnoxStbmAd6sQLg9SCrrSxRkZ/view?usp=sharing)\]
\[[slides](https://drive.google.com/file/d/1l5mgKpINXCwxKR5jwgjCtNlFC2tsbRZ_/view?usp=sharing)\]
\[[poster](https://drive.google.com/file/d/1DVWinfSI4OQwHwXpDTqOk4cWR7Pa5g7n/view?usp=sharing)\]
\[[video](https://youtu.be/WdAVw1r-fno)\]

<img src="https://github.com/Shunli-Wang/TSA-Net/blob/main/fig/TSA-Net.jpg"/>

If this repository is helpful to you, please star it. If you find our work useful in your research, please consider citing:
```
@inproceedings{TSA-Net,
  title={TSA-Net: Tube Self-Attention Network for Action Quality Assessment},
  author={Wang, Shunli and Yang, Dingkang and Zhai, Peng and Chen, Chixiao and Zhang, Lihua},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  year={2021},
  pages={4902–4910},
  numpages={9}
}
```

## User Guide
In this repository, we open source the code of TSA-Net on FRFS dataset. The initialization process is as follows:
```bash
# 1.Clone this repository
git clone https://github.com/Shunli-Wang/TSA-Net.git ./TSA-Net
cd ./TSA-Net

# 2.Create conda env
conda create -n TSA-Net python
pip install -r requirments.txt

# 3.Download pre-trained model and FRFS dataset. All download links are listed as follow.
# PATH/TO/rgb_i3d_pretrained.pt 
# PATH/TO/FRFS

# 4.Create data dir
mkdir ./data && cd ./data
mv PATH/TO/rgb_i3d_pretrained.pt ./
ln -s PATH/TO/FRFS ./FRFS
```
After initialization, please check the data structure:
```bash
.
├── data
│   ├── FRFS -> PATH/TO/FRFS
│   └── rgb_i3d_pretrained.pt
├── dataset.py
├── train.py
├── test.py
...
```
Download links:
- __FRFS Dataset__: You can download the FRFS dataset (About 2.5 G) from [BaiduNetDisk](https://pan.baidu.com/s/1Nkl6FlM2PcvbofegNjCIGA) \[star\] or [Google Drive](https://drive.google.com/file/d/1wmMUtMx5eqOFMa8vtM_pA6S9Psxwq3_l/view?usp=sharing)
- __rgb_i3d_pretrained.pt__: I3D backbone pretrained on Kinetics ([BaiduNetDisk](https://pan.baidu.com/s/1L1MqzlTDFtbOKLYm1b1GpQ ) \[i3dm\] or [Google Drive](https://drive.google.com/file/d/1M_4hN-beZpa-eiYCvIE7hsORjF18LEYU)) is used in our work, which is referenced from [Gated-Spatio-Temporal-Energy-Graph](https://github.com/yaohungt/Gated-Spatio-Temporal-Energy-Graph).

## Training & Evaluation

## Acknowledgement
Our code is adapted from [MUSDL](https://github.com/nzl-thu/MUSDL). We are very grateful for their wonderful implementation and selfless contributions.

## Contact
If you have any questions about our work, please contact <slwang19@fudan.edu.cn>.
