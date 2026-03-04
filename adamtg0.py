#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adam 1 - Eksperyment z shallow architecture, single behavior
"""
import os
print("Current katalog:", os.getcwd())
#
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
TESTSETATLAST = False ############### zostaw w spokoju
SPRKOD = False


# ISDEEP=False; ISDUAL=False ## 1 Shallow Single
ISDEEP=True;  ISDUAL=False ## 2 Deep Single
# ISDEEP=False; ISDUAL=True  ## 3 Shallow Dual
# ISDEEP=True;  ISDUAL=True  ## 4 Deep Dual
  

ILERUNOW = 2 if (ISDEEP or not SPRKOD) else 1

SMTYPE = "rfr"  #"lr" #  "xgbr" # 
DMTYPE = "bigruta"   # "lstm"

WYBORKOLS_TG5 = "tg_rfr_important"

EPOCHS = 2 if SPRKOD else 500 
BATCHSIZE = 512 
SEQLEN = 5 if SPRKOD else 30

DMHPS_MONO = wynik_naj411
DMHPS_REZIMA = wynik_5020_rezimA
DMHPS_REZIMB = wynik_naj411

ARCHIT_NAME = "deep" if ISDEEP else "shallow"
APPRO_NAME = "dual" if ISDUAL else "single"
COROBIE = f'__maskAOST__bs{BATCHSIZE}_sl{SEQLEN}__{ILERUNOW}runs__{ARCHIT_NAME}_{APPRO_NAME}'

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
TIMESTAMPPLUS = TIMESTAMP + COROBIE


#### ====== KONFIGURACJA DANYCH ======
DATACFG_TG5 = DataConfig(
        projekt_akronim='tg',
        plik_dane = '../mojecsvy/tg5NEW/2022__sg_kols__interpolated.csv',
        kols_map=kols_map_tg5,
        kol_target      = 'MW_moc',
        kol_flagA       = 'shutdown', ######
        intkols_to_scale= []
        )







#### ====== URUCHOMIENIE ======
if __name__ == "__main__":
    archit, behav, dfres_handler = run_experiment(
        
        # Konfiguracja
        datacfg=DATACFG_TG5,
        wyborkols=WYBORKOLS_TG5,
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
