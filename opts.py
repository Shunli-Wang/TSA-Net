# directory containing frames
frames_dir = './data/frames'
# directory containing labels and annotations
info_dir = './data/info'

# i3d model pretrained on Kinetics, https://github.com/yaohungt/Gated-Spatio-Temporal-Energy-Graph
i3d_pretrained_path = './data/rgb_i3d_pretrained.pt'

# num of frames in a single video
num_frames = 103

# beginning frames of the 10 segments
# segment_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 87]
segment_points = [0, 16, 32, 48, 64, 80, 87]
seg_number = 7

# input data dims;
C, H, W = 3, 224, 224
# image resizing dims;
input_resize = 455, 256

# statistics of dataset
label_max = 104.5
label_min = 0.
judge_max = 10.
judge_min = 0.

# output dimension of I3D backbone
feature_dim = 1024

# For USDL, normalized score is chosen from [0, 1, ..., 100].
# For MUSDL, since each judge choose a score from [0, 0.5, ..., 9.5, 10], so the output dim is 21.
output_dim = {'USDL': 1, 'MUSDL': 21}

# num of judges in MUSDL method
num_judges = 7

H_img, W_img = 360, 640

sheet_names = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
# score norm for AQA-7: [min, max]
score_norm = [[21.6, 102.6], [12.3, 16.87], [8, 50], [8, 50], [46.2, 104.88], [49.8, 99.36]]  #
# score_norm = [[0, 102.6], [0, 16.87], [0, 50], [0, 50], [0, 104.88], [0, 99.36]]  #
score_std = [14.4912, 0.8608, 11.3202, 12.8210, 14.7112, 15.2136]

I3D_ENDPOINTS = {  # [name,channel,T,size]
    0: ['Conv3d_1a_7x7', 64, 8, 112],
    1: ['MaxPool3d_2a_3x3', 64, 8, 56],
    2: ['Conv3d_2b_1x1', 64, 8, 56],
    3: ['Conv3d_2c_3x3', 12, 8, 56],
    4: ['MaxPool3d_3a_3x3', 192, 8, 28],
    5: ['Mixed_3b', 256, 8, 28],
    6: ['Mixed_3c', 480, 8, 28],
    7: ['MaxPool3d_4a_3x3', 480, 4, 14],  # exp_25
    8: ['Mixed_4b', 512, 4, 14],  # exp_26
    9: ['Mixed_4c', 512, 4, 14],  # exp_27
    10: ['Mixed_4d', 512, 4, 14],  # exp_28
    11: ['Mixed_4e', 528, 4, 14],  # exp_29
    12: ['Mixed_4f', 832, 4, 14],  # exp_30
    13: ['MaxPool3d_5a_2x2', 832, 2, 7],
    14: ['Mixed_5b', 832, 2, 7],  # exp_31
    15: ['Mixed_5c', 1024, 2, 7],  # exp_32
    16: ['Logits'],
    17: ['Predictions'],
}
