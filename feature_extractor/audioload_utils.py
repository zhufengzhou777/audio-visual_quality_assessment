from warnings import simplefilter

import cv2
import librosa
import librosa.display
import numpy as np
import torch

simplefilter(action='ignore', category=FutureWarning)


def load_one_video_audio(path, final_frame, fps, input_dim, hop_len=20, hop_sample_num=512, audio_sample_rate=22050):
    y, sr = librosa.load(path)  # y: the audio data;  sr: 采样率.  sr(default)=22050
    # len(y):sample_rate*duration
    melspec = librosa.feature.melspectrogram(y, sr)
    logmelspec = librosa.power_to_db(melspec, ref=np.max)  # log_mel ref=max即根据最大值缩放
    logmelspec = logmelspec - np.min(logmelspec)
    logmelspec = logmelspec / np.max(logmelspec) * 255  # max-min归一化

    # convert to frames
    one_video_audio = []
    one_frame_time = 1000.0 / fps  # 单帧时间(ms) (40+)
    one_hop_time = hop_sample_num * 1000 / audio_sample_rate  # 帧移时间ms
    mel_spectr_len = np.shape(logmelspec)[1]  # 时域维度
    for i_frame in range(final_frame):  # 帧总数(240)
        frame_point = int(one_frame_time * i_frame / one_hop_time)
        one_frame_start = frame_point - int(hop_len / 2)
        one_frame_end = frame_point + int(hop_len / 2)
        if one_frame_start < 0:
            off_set = 0 - one_frame_start
        elif one_frame_end > mel_spectr_len:
            off_set = mel_spectr_len - one_frame_end
        else:
            off_set = 0
        one_frame_start += off_set
        one_frame_end += off_set

        one_frame_audio = logmelspec[:, one_frame_start:one_frame_end]

        one_frame_audio = cv2.resize(one_frame_audio, (input_dim, input_dim))  # (128*20)
        one_video_audio.append(one_frame_audio)  # 240*(in*in)
    return one_video_audio


def load_one_batch_audio_melspectr(one_video_audio, i_batch, train_batch_size, frame_offset=0,
                                   final_frame=192, one_visual_frame_3D_frames=16, debug=False):
    one_batch_audio = []

    one_batch_start = frame_offset + i_batch * train_batch_size
    one_batch_end = frame_offset + (i_batch + 1) * train_batch_size

    if not final_frame == None:
        if one_batch_end > final_frame:
            return -1, -1

    for i_frame in range(one_batch_start, one_batch_end):
        one_frame_start = i_frame - int(one_visual_frame_3D_frames / 2)
        one_frame_end = i_frame + int(one_visual_frame_3D_frames / 2)

        if one_frame_start < 0:
            off_set = 0 - one_frame_start
        elif one_frame_end > final_frame:
            off_set = final_frame - one_frame_end
        else:
            off_set = 0
        one_frame_start += off_set
        one_frame_end += off_set

        one_frame_audio = []
        for i_step in range(one_frame_start, one_frame_end):
            one_step_audio = one_video_audio[i_step]
            one_step_audio = np.expand_dims(one_step_audio, axis=0)
            one_frame_audio.append(one_step_audio)

        one_batch_audio.append(one_frame_audio)
    one_batch_audio = np.array(one_batch_audio).transpose(0, 2, 1, 3, 4)
    one_batch_audio = torch.from_numpy(np.array(one_batch_audio))

    return one_batch_audio


def load_audio(audio_dir, actual_video_frames, length=48, input_dim=224, final_frame=192, fps=24):
    f1 = []
    with torch.no_grad():
        one_video_audio = load_one_video_audio(audio_dir, actual_video_frames, fps, input_dim)
        for ova in one_video_audio:
            ova = torch.from_numpy(ova)
            ovas = torch.cat((ova, ova, ova), 0)
            ovast = ovas.view(3, ova.shape[0], ova.shape[1]).float()
            f1.append(ovast)
        f1 = torch.stack(f1)
        # batch_num = int(final_frame / length)
        # f1 = rearrange(f1, '(b c) d h w->b c d h w', b=batch_num)
    return f1
