#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 23:18:05 2025; @author: sylwia

nie wiem czemu ale nie pokazuje nie robi tego ostatniego niepełnego batcha 
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from d2_kluski import KluskiConfig


X_for_keras = np.array([
    [1.0, 0.1, 0],   # wiersz 0 - shutdown=0
    [2.0, 0.2, 0],   # wiersz 1 - shutdown=0
    [3.0, 0.3, 1],   # wiersz 2 - shutdown=1 (start shutdownu)
    [4.0, 0.4, 1],   # wiersz 3 - shutdown=1
    [5.0, 0.5, 1],   # wiersz 4 - shutdown=1
    [6.0, 0.6, 0],   # wiersz 5 - shutdown=0 (koniec shutdownu)
    [7.0, 0.7, 0],   # wiersz 6 - shutdown=0
    [8.0, 0.8, 1],   # wiersz 7 - shutdown=1 (kolejny shutdown)
    [9.0, 0.9, 1],   # wiersz 8 - shutdown=1
    [1.9, 9.1, 1],   # wiersz 9  - shutdown=1
    [2.8, 8.2, 0],   # wiersz 10 - shutdown=0
    [3.7, 7.3, 0],   # wiersz 11 - shutdown=0
    [4.6, 6.4, 0],   # wiersz 12 - shutdown=0
    [5.5, 5.5, 1],   # wiersz 13 - shutdown=1
    [6.4, 4.6, 1],   # wiersz 14 - shutdown=1
    [7.3, 3.7, 0],   # wiersz 15 - shutdown=0
    [8.2, 2.8, 0],   # wiersz 16 - shutdown=0
    [9.1, 1.9, 0],   # wiersz 17 - shutdown=0
])
y_for_keras = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90,  # train (0-8)
                        19, 28, 37, 46, 55, 64, 73, 82, 91]) # val (9-16)
df_for_keras = pd.DataFrame({
    't_blade': X_for_keras[:, 0],
    'shut_minutes': X_for_keras[:, 1],
    'shutdown': X_for_keras[:, 2].astype(int),  # boolean kolumna
    'target': y_for_keras
}, index=pd.date_range('2023-01-01', periods=18, freq='h'))

# Parametry:
SEQLEN = 3  # każdy klusek ma 3 timestepy
delay = SEQLEN
BATCHSIZE = 2  # 2 kluski w batchu


qty_train, qty_val, qty_test = 4, 5, 9
assert sum([qty_train, qty_val, qty_test])==len(df_for_keras)

#                      t_blade  shut_minutes  shutdown  target
# 2023-01-01 00:00:00      1.0           0.1         0      10
# 2023-01-01 01:00:00      2.0           0.2         0      20
# 2023-01-01 02:00:00      3.0           0.3         1      30 #
# 2023-01-01 03:00:00      4.0           0.4         1      40

# 2023-01-01 04:00:00      5.0           0.5         1      50
# 2023-01-01 05:00:00      6.0           0.6         0      60
# 2023-01-01 06:00:00      7.0           0.7         0      70 #
# 2023-01-01 07:00:00      8.0           0.8         1      80
# 2023-01-01 08:00:00      9.0           0.9         1      90

# 2023-01-01 09:00:00      1.9           9.1         1      19
# 2023-01-01 10:00:00      2.8           8.2         0      28
# 2023-01-01 11:00:00      3.7           7.3         0      37 #
# 2023-01-01 12:00:00      4.6           6.4         0      46
# 2023-01-01 13:00:00      5.5           5.5         1      55
# 2023-01-01 14:00:00      6.4           4.6         1      64 #
# 2023-01-01 15:00:00      7.3           3.7         0      73
# 2023-01-01 16:00:00      8.2           2.8         0      82
# 2023-01-01 17:00:00      9.1           1.9         0      91


#### KDS
EPOCHS = 1
BATCHSIZE = 2
SEQLEN = 5

k2_zrozum = KluskiConfig(df_for_keras,
                           qty_train, qty_val, qty_test, 
                           'shutdown',
                            EPOCHS, BATCHSIZE, SEQLEN,
                            True)

# (kds_A, kds_B, rezimA_mask, idx_A, idx_B
#      ) = k2_zrozum.rozdziel_kluski_na_rezimy_A_i_B(train_kds, "train_kds")

print(f"\n\nqty_train, qty_val, qty_test = {qty_train, qty_val, qty_test}")



(k2_zrozum.train_kds_dict, 
 k2_zrozum.val_kds_dict, 
 k2_zrozum.test_kds_dict
 ) = k2_zrozum.stworz_uzupelnione_slowniki()





#%%
#### PDS
train_pds = df_for_keras.iloc[0:qty_train].copy()   
val_pds = df_for_keras.iloc[qty_train:qty_train+qty_val].copy()    
test_pds = df_for_keras.iloc[qty_train+qty_val:].copy()    

X_train = train_pds.iloc[:, :-1]
y_train = train_pds.iloc[:, -1]
X_val = val_pds.iloc[:, :-1] 
y_val = val_pds.iloc[:, -1]
X_test = test_pds.iloc[:, :-1]
y_test = test_pds.iloc[:, -1]

df_for_keras = pd.concat([pd.concat([X_train, y_train], axis=1), 
                          pd.concat([X_val, y_val], axis=1), 
                          pd.concat([X_test, y_test], axis=1)], axis=0)

X_for_keras = np.round(df_for_keras.drop('target', axis=1).values, 3)
y_for_keras = np.round(df_for_keras['target'].values, 3)

## indeksy pds
# =============================================================================
# print("\nTRAIN_PDS:")
# # print(train_pds)
# pd_train_idx = train_pds.index
# print(f"pd_train_idx: {pd_train_idx}")
# 
# print("\nVAL_PDS:")
# # print(val_pds)
# pd_val_idx = val_pds.index
# print(f"pd_val_idx: {pd_val_idx}")
# 
# print("\nTEST_PDS:")
# # print(test_pds)
# pd_test_idx = test_pds.index
# print(f"pd_test_idx: {pd_test_idx}")
# =============================================================================



