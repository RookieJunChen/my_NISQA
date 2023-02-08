#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Time : 2023/2/8 16:01
# @Author : Jun Chen
# @File : preprocess_wavfile.py

import librosa
import argparse
import os
import glob
import numpy as np
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str,
                    default="/apdcephfs/share_976139/users/thujunchen/data/dns5_blind/V5_BlindTestSet_mono/Track1_Headset/noisy")
parser.add_argument('--preprocess_dir', type=str,
                    default="/apdcephfs/share_976139/users/thujunchen/data/prepross_temp/BlindTestSet_Track1_Headset_noisy")
parser.add_argument('--sr', type=int, default=48000)
parser.add_argument('--chunk_lentgh', type=int, default=50)
args = parser.parse_args()


def cut_wav(wav_vec, chunk_length):
    cut_wav_vec_lst = []
    part_num = int(wav_vec.shape[-1] / chunk_length)
    for i in range(0, part_num):
        begin_p = chunk_length * i
        end_p = chunk_length * (i + 1)
        cut_wav_vec_lst.append(wav_vec[begin_p:end_p])

    return cut_wav_vec_lst


if __name__ == "__main__":
    input_dir = args.input_dir
    preprocess_dir = args.preprocess_dir
    os.makedirs(preprocess_dir, exist_ok=True)
    input_files = [fp for fp in glob.glob(os.path.join(input_dir, '*.wav'))]

    sr = args.sr
    max_wav_length = sr * args.chunk_lentgh

    for i, filename in enumerate(input_files):
        wav_name = os.path.basename(filename)
        audio = librosa.core.load(filename, sr)[0]

        if audio.shape[-1] > max_wav_length:
            audio_lst = cut_wav(audio, max_wav_length)
            print("{} is too long ({}s), cut it to {} chunks.".
                  format(wav_name, audio.shape[-1] / sr, len(audio_lst)))
            for j, audio_cut in enumerate(audio_lst):
                prepross_wav_pth = os.path.join(preprocess_dir, os.path.splitext(wav_name)[0] + '_part{}.wav'.format(j))
                sf.write(prepross_wav_pth, audio_cut, sr, subtype='PCM_16')

        else:
            print("{} is normal ({}s).".format(wav_name, audio.shape[-1] / sr))
            prepross_wav_pth = os.path.join(preprocess_dir, wav_name)
            sf.write(prepross_wav_pth, audio, sr, subtype='PCM_16')
