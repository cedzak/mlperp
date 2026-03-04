#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adam 1 - Eksperyment z shallow architecture, single behavior
"""
from datetime import datetime
from inputs.i_columns import *
from inputs.i_mdeep_hps import *
from klasy_data.d0_data import DataConfig
from klasy_jesien.runner import run_experiment


#### constants
arabian_wav = '../mixkit-arabian.wav'
game_wav = '../mixkit-game-notification.wav'
KOLOR_ACTUAL, KOLOR_PRED, KOLOR_RES = "limegreen", "navy", "purple"


#### ====== PARAMETRY EKSPERYMENTU ======
TESTSETATLAST = True
ILERUNOW = 2
SPRKOD = True

ISDEEP = False
ISDUAL = False 

SMTYPE = "rfr"  #"lr" #  "xgbr" # 
DMTYPE = "bigruta"   # "lstm"

WYBORKOLS_MR = "mr_testowe" if SPRKOD else "mr_niedokonczony_i_shutdown"

EPOCHS = 2 if SPRKOD else 500 
BATCHSIZE = 512 
SEQLEN = 5 if SPRKOD else 30

DMHPS_MONO = wynik_naj411
DMHPS_REZIMA = wynik_5020_rezimA
DMHPS_REZIMB = wynik_naj411

ARCHIT_NAME = "deep" if ISDEEP else "shallow"
APPRO_NAME = "dual" if ISDUAL else "single"
COROBIE = f'__maskA_OST__bs{BATCHSIZE}_sl{SEQLEN}__{ILERUNOW}runs__{ARCHIT_NAME}_{APPRO_NAME}'

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
TIMESTAMPPLUS = TIMESTAMP + COROBIE


#### ====== KONFIGURACJA DANYCH ======
DATACFG_MR = DataConfig(
        projekt_akronim = 'mr',
        plik_dane       = '../mojecsvy/mr/maj__20250521_expert_included.csv',
        kols_map        = kols_map_mr,
        kol_target      = 't_casing',
        kol_flagA       = 'shutdown', ######
        intkols_to_scale= ['RPM_diff_binned', 'shut_minutes'],
        #
        date_start = '2023-03-31 11:00:00', date_end = '2024-02-27 18:00:00',
        out1_start = '2023-12-19 15:30:00', out1_end = '2023-12-28 12:30:00',
        out2_start = '2024-01-19 19:15:00', out2_end = '2024-01-22 11:00:00'
        )


#### ====== URUCHOMIENIE ======
if __name__ == "__main__":
    archit, behav, dfres_handler = run_experiment(
        
        # Konfiguracja
        datacfg=DATACFG_MR,
        wyborkols=WYBORKOLS_MR,
        timestampplus=TIMESTAMPPLUS,
        corobie=COROBIE,
        
        # Parametry główne
        isdeep=ISDEEP,
        isdual=ISDUAL,
        ilerunow=ILERUNOW,
        smtype=SMTYPE,
        dmtype=DMTYPE,
        
        # Parametry deep model (None dla shallow)
        epochs=EPOCHS,
        batchsize=BATCHSIZE,
        seqlen=SEQLEN,
        
        # Hiperparametry deep model (None dla shallow)
        dmhps_Mono=DMHPS_MONO,
        dmhps_rezimA=DMHPS_REZIMA,
        dmhps_rezimB=DMHPS_REZIMB,
        
        # Flagi
        sprkod=SPRKOD,
        testsetatlast=TESTSETATLAST,
        
        # Oprawa artystyczna
        kolor_actual=KOLOR_ACTUAL,
        kolor_pred=KOLOR_PRED,
        kolor_res=KOLOR_RES,
        arabian_wav=arabian_wav,
        game_wav=game_wav
    )
