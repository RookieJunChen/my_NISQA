#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Time : 2023/1/17 15:45
# @Author : Jun Chen
# @File : eval_csv.py
import argparse
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--csv_pth', type=str, default="/apdcephfs_cq3/share_2906397/users/thujunchen/enhanced_audio/nisqa/blind_data/NISQA_results.csv")
args = parser.parse_args()


def read_xlsx(csv_dir):
    df = pd.read_csv(csv_dir)
    # print(df)
    # print(df.columns)
    return df


if __name__ == "__main__":
    csv_pth = args.csv_pth
    df = read_xlsx(csv_pth)
    mos_pred = df['mos_pred']
    noi_pred = df['noi_pred']
    dis_pred = df['dis_pred']
    col_pred = df['col_pred']
    loud_pred = df['loud_pred']
    print("mos_pred: {}, noi_pred: {}, dis_pred: {}, col_pred: {}, loud_pred: {}"
          .format(mos_pred.mean(), noi_pred.mean(), dis_pred.mean(), col_pred.mean(), loud_pred.mean()))


