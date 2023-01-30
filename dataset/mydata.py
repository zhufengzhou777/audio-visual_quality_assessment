import numpy as np
from torch.utils.data import Dataset


class Traindata(Dataset):
    def __init__(self, root_dir, name_file):
        self.root_dir = root_dir
        self.name_file = name_file
        self.size = 0
        self.name_list = []
        file = open(self.name_file)
        for f in file:
            self.name_list.append(f)
            self.size += 1

    def __getitem__(self, idx):
        flag = int(self.name_list[idx].split(' ')[0])
        video_path = self.root_dir + '/video/' + self.name_list[idx].split(' ')[1].split('.')[0] + '_' + str(flag) + '.npy'
        audio_path = self.root_dir + '/audio/' + self.name_list[idx].split(' ')[2].split('.')[0] + '_' + str(flag) + '.npy'
        score = float(self.name_list[idx].split(' ')[3])
        video = np.load(video_path)
        audio = np.load(audio_path)
        sample = {'video': video, 'audio': audio, 'score': score}
        return sample

    def __len__(self):
        return self.size


class Testdata(Dataset):
    def __init__(self, root_dir, name_file):
        self.root_dir = root_dir
        self.name_file = name_file
        self.size = 0
        self.name_list = []
        file = open(self.name_file)
        for f in file:
            self.name_list.append(f)
            self.size += 1

    def __getitem__(self, idx):
        # video_path = self.root_dir + '/video/' + self.name_list[idx].split(' ')[0].split('.')[0] + '.npy'
        # audio_path = self.root_dir + '/audio/' + self.name_list[idx].split(' ')[1].split('.')[0] + '.npy'
        # score = float(self.name_list[idx].split(' ')[2])
        # video = np.load(video_path)
        # audio = np.load(audio_path)
        # sample = {'video': video, 'audio': audio, 'score': score}
        # return sample
        flag = int(self.name_list[idx].split(' ')[0])
        video_path = self.root_dir + '/video/' + self.name_list[idx].split(' ')[1].split('.')[0] + '_' + str(
            flag) + '.npy'
        audio_path = self.root_dir + '/audio/' + self.name_list[idx].split(' ')[2].split('.')[0] + '_' + str(
            flag) + '.npy'
        score = float(self.name_list[idx].split(' ')[3])
        video = np.load(video_path)
        audio = np.load(audio_path)
        sample = {'video': video, 'audio': audio, 'score': score}
        return sample

    def __len__(self):
        return self.size