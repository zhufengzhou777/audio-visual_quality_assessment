import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from audioload_utils import load_audio
from videoload_utils import load_video

sys.path.append(os.getcwd())
from model.vit_feature_extractor import vit_feature_extractor

namelist = ['Goose', 'RedKayak', 'Stream', 'Boxing', 'Fountain', 'BigGreenRabbit', 'CrowdRun', 'PowerDig', 'Speech',
            'FootMusic', 'Town', 'Drama', 'Sparks', 'Car']

frame_dict = {}

transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_cat(file_name):
    cat = file_name.split('_')[0]
    return cat


def get_feature(file_path, output_path):
    """
    assert /audio and /video in filepath, the same with output_path
    """
    if not file_path.endswith('/'):
        file_path += '/'
    if not output_path.endswith('/'):
        output_path += '/'
    video_path = file_path + 'video/'
    audio_path = file_path + 'audio/'
    video_file_list = os.listdir(video_path)
    audio_file_list = os.listdir(audio_path)
    print('loading video ...')
    for fp in tqdm(video_file_list):
        cat_name = get_cat(fp)
        assert cat_name in namelist
        video = load_video(video_path + fp)
        frame_num = video.shape[0]
        if cat_name not in frame_dict.keys():
            frame_dict[cat_name] = int(frame_num)
        video = video[::2]
        video = np.array([cv2.resize(x, (224, 224), cv2.INTER_LANCZOS4) for x in video])
        # video.shape (192, 224, 224, 3) video.type <class 'numpy.ndarray'>
        video = torch.from_numpy(np.float32((np.transpose(video, (0, 3, 1, 2)))))
        # video.shape torch.Size([192, 3, 224, 224]) video.type <class 'torch.Tensor'>
        video = transform(video)
        video = vit_feature_extractor(video)
        video_file_name = fp.split('.')[0]
        video_save_name = output_path + 'video/' + video_file_name
        np.save(video_save_name + '_0', video[0:24])
        np.save(video_save_name + '_1', video[24:48])
        np.save(video_save_name + '_2', video[48:72])
        np.save(video_save_name + '_3', video[72:96])
    print('loading audio ...')
    for ap in tqdm(audio_file_list):
        cat_name = get_cat(ap)
        assert cat_name in namelist
        frame_num = frame_dict[cat_name]
        audio = load_audio(audio_path + ap, frame_num, fps=24)[::2]
        audio = transform(audio)
        # audio.shape torch.Size([192, 3, 224, 224]) video.type <class 'torch.Tensor'>
        audio = vit_feature_extractor(audio)
        audio_file_name = ap.split('.')[0]
        audio_save_name = output_path + 'audio/' + audio_file_name  # [F,197,768]
        np.save(audio_save_name + '_0', audio[0:24])
        np.save(audio_save_name + '_1', audio[24:48])
        np.save(audio_save_name + '_2', audio[48:72])
        np.save(audio_save_name + '_3', audio[72:96])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str, default='./data')
    args = parser.parse_args()
    input = args.input
    output = args.output
    get_feature(input, output)
