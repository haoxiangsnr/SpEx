import os
import random

import librosa
import numpy as np
import torchaudio as torchaudio
from joblib import Parallel, delayed
from torch.utils import data

from util.utils import sample_fixed_length_data_aligned


class Dataset(data.Dataset):
    def __init__(self, dataset_list, limit=None, offset=0, n_samples=32000, sr=8000,
                 reference_length=5):
        """
        验证数据集

        dataset_list(*.txt):
            <mixture_path> <target_path> <reference_path>\n
        e.g:
            mixture_1.wav target_1.wav reference_1.wav
            mixture_2.wav target_2.wav reference_2.wav
            ...
            mixture_n.wav target_n.wav reference_n.wav
        """
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset_list)), "r")]
        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        # extract target wav
        target_wav_path_list = [item.split(" ")[1] for item in dataset_list]
        self.speaker_dict = self.build_speaker_dict(target_wav_path_list)
        self.speaker_id_list = list(self.speaker_dict.keys())
        print("Number of the speakers is: ", len(self.speaker_id_list))

        self.dataset_list = dataset_list
        self.length = len(self.dataset_list)
        self.n_samples = n_samples
        self.sr = sr
        self.reference_length = reference_length

    def build_speaker_dict(self, wav_path_list):
        """
        Args:
            wav_path_list: ["~/Datasets/SpeakerBeam/online_first_wavdata/3110FB112self.reference_length321_speech_13_238self.reference_length167_0.wav", ...]

        Returns:
            {"3110FB112self.reference_length321": [abs_path_1, abs_path_2, ...], ...}
        """
        random.shuffle(wav_path_list)

        fully_speaker_dict = {}
        for file_path in wav_path_list:
            speaker_id = self.get_speaker_id(self.get_filename(file_path))
            if speaker_id in fully_speaker_dict:
                fully_speaker_dict[speaker_id].append(file_path)
            else:
                fully_speaker_dict[speaker_id] = [file_path]

        return fully_speaker_dict

    @staticmethod
    def get_filename(file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        return filename

    @staticmethod
    def get_speaker_id(filename):
        speaker_id = filename.split("_")[0]
        return speaker_id

    def __len__(self):
        return self.length

    def load_wav(self, file_path):
        return librosa.load(os.path.abspath(os.path.expanduser(file_path)), sr=self.sr)[0]

    def __getitem__(self, item):
        mixture_path, target_path = self.dataset_list[item].split(" ")

        target_filename = self.get_filename(target_path)
        target_speaker_id = self.get_speaker_id(target_filename)

        reference_wav_path_list = self.speaker_dict[target_speaker_id]

        reference_wav_path = random.choice(reference_wav_path_list)
        reference_filename = self.get_filename(reference_wav_path)
        while target_filename == reference_filename:
            reference_wav_path = random.choice(reference_wav_path_list)
            reference_filename = self.get_filename(reference_wav_path)

        mixture_y = self.load_wav(mixture_path)
        target_y = self.load_wav(target_path)
        reference_y = self.load_wav(reference_wav_path)

        if len(reference_y) > (self.sr * self.reference_length):
            start = np.random.randint(len(reference_y) - self.sr * self.reference_length + 1)
            end = start + self.sr * self.reference_length
            reference_y = reference_y[start:end]
        else:
            reference_y = np.pad(reference_y, (0, self.sr * self.reference_length - len(reference_y)))

        mixture_y, target_y = sample_fixed_length_data_aligned(mixture_y, target_y, self.n_samples)

        return mixture_y.astype(np.float32), target_y.astype(np.float32), reference_y, reference_filename,
