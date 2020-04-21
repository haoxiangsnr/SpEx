import os
import random

import librosa
import numpy as np
import torchaudio as torchaudio
from joblib import Parallel, delayed
from torch.utils import data

from util.utils import sample_fixed_length_data_aligned


class Dataset(data.Dataset):
    def __init__(self, dataset_list, limit=None, offset=0, sr=8000):
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
        dataset_list = [line.rstrip('\n') for line in
                        open(os.path.abspath(os.path.expanduser(dataset_list)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.dataset_list = dataset_list
        self.length = len(self.dataset_list)
        self.sr = sr

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

    @staticmethod
    def get_reference(filename):
        namelist = filename.split('.wav')[0].split('_')
        if len(namelist) <= 4:
            rname = namelist[0] + '_adaptation_' + namelist[-1] + '.wav'
        else:
            rname = namelist[0] + '_adaptation_' + namelist[-5] + "_" + namelist[-4] + "_" + namelist[-3] + "_" + \
                    namelist[-2] + "_" + namelist[-1] + '.wav'
        return rname

    def __getitem__(self, item):
        mixture_path, target_path = self.dataset_list[item].split(" ")

        target_filename = self.get_filename(target_path)
        reference_filename = self.get_reference(target_filename)
        reference_wav_path = f"~/Datasets/SpeakerBeam/test_S1_real_ALL/test_S1_real_ALL_15s_ref/{reference_filename}"

        mixture_y = self.load_wav(mixture_path)
        target_y = self.load_wav(target_path)
        reference_y = self.load_wav(reference_wav_path)

        return mixture_y.astype(np.float32), target_y.astype(np.float32), reference_y, target_filename
