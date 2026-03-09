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


import sys
print(sys.executable)
print(sys.version)



#### constants
arabian_wav = '../mixkit-arabian.wav'
game_wav = '../mixkit-game-notification.wav'
KOLOR_ACTUAL, KOLOR_PRED, KOLOR_RES = "limegreen", "navy", "purple"


#### ====== PARAMETRY EKSPERYMENTU ======
TESTSETATLAST = False ############### zostaw w spokoju
SPRKOD = False
WYBORKOLS = 'top_gdaciep_kols' # 'gdaciep_base_kols' #
CIEPLO = True



#######################################################
# ISDEEP=False; ISDUAL=False ## 1 Shallow Single
ISDEEP=True;  ISDUAL=False ## 2 Deep Single
# ISDEEP=False; ISDUAL=True  ## 3 Shallow Dual
# ISDEEP=True;  ISDUAL=True  ## 4 Deep Dual

## shallow 
SMTYPE = "rfr"  #"lr" #  "xgbr" #

## deep
ALIGNED_KROTNOSC_24H = True #### BEZ SENSU, SHALLOWOWI TO ZWISA: False if ISDEEP==False else True #### CZY FALSE --> TU WYBIERASZ
# True: X[i..i+k-1] → y[i..i+k-1]; wymaga ISDEEP=True, SEQLEN=wielokrotność 24, TASK='seq2seq'
k=1
SEQLEN = k*24 if ALIGNED_KROTNOSC_24H else (5 if SPRKOD else 72)
DMTYPE = "enc_dec" # "bigruta"  # "lstm" 
TASK  = "seq2seq" if ALIGNED_KROTNOSC_24H else "seq2one"
#
EPOCHS = 2 if SPRKOD else 500
BATCHSIZE = 128
#
#
# DMHPS_MONO = wynik_naj_gdaciep_enc_dec
DMHPS_MONO = wynik_naj_gdaciep_bigruta
#
DMHPS_REZIMA = wynik_5020_rezimA
DMHPS_REZIMB = wynik_naj411





ILERUNOW = 1 if (SPRKOD or not ISDEEP) else 2





ARCHIT_NAME = "deep" if ISDEEP else "shallow"
APPRO_NAME = "dual-maskost" if ISDUAL else "single" ## tu moge zmienić maske
COROBIE = f'__{ARCHIT_NAME}_{APPRO_NAME}__'
corobieDEEP = f'__bs{BATCHSIZE}_sl{SEQLEN}__{ILERUNOW}runs__{DMTYPE}'
corobieSHALLOW = f'__{SMTYPE}'
corobie_end = corobieDEEP if ISDEEP else corobieSHALLOW 
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
TIMESTAMPPLUS = COROBIE + TIMESTAMP + corobie_end


#### ====== KONFIGURACJA DANYCH ======
DATACFG_CIEPLO = DataConfig(
        projekt_akronim='gdaciep',
        plik_dane = '../mojecsvy/gdaciep/cieplo_pogoda_czas__sg.csv', 
        kols_map=kols_map_gdaciep,
        kol_target      = 'MW_th',
        kol_flagA       = 'is_weekend', ######
        intkols_to_scale= [],
        frac_val        = 0.15, ##!!
        frac_test       = 0.15,  ##!!
        aligned_krotnosc_24h = ALIGNED_KROTNOSC_24H,
        )

DATACFG_TGE = DataConfig(
        projekt_akronim='gdatge',
        plik_dane = '../mojecsvy/gdatge/tge__sg.csv', 
        kols_map=kols_map_gdatge,
        kol_target      = 'fix_I_pln',
        kol_flagA       = 'is_weekend', ######
        intkols_to_scale= [],
        aligned_krotnosc_24h = ALIGNED_KROTNOSC_24H,
        )




#### ====== URUCHOMIENIE ======
if __name__ == "__main__":
    print('\n\n\n\n\n\n\n\n\n\n')
    print(COROBIE)
    archit, behav, dfres_handler = run_experiment(
        
        # Konfiguracja
        datacfg = DATACFG_CIEPLO if CIEPLO else DATACFG_TGE,
        wyborkols=WYBORKOLS,
        timestampplus=TIMESTAMPPLUS,
        corobie=COROBIE,

        # Parametry główne
        isdeep=ISDEEP,
        isdual=ISDUAL,
        ilerunow=ILERUNOW,
        smtype=SMTYPE,
        dmtype=DMTYPE,
        task=TASK,

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



#%% wyniki ciep

# all kols
# mse=8.511, rmse=2.917,   mae=1.715

# top10_features_permut_importance__valset
# mse=9.67, rmse=3.11,   mae=1.859

# top8_features_feat_importance
    # mse=9.296, rmse=3.049, mae=1.824
# enc_dec:
    # loss=61.746, mae=5.284
# po naprawie CLCO: 
    # loss=923.017, mae=27.156... -- ok, on jest niestabilny, trzeba wtedy probowac dalej
# enc_dec po hp tuning:
    # loss=29.632, mae=3.809
# bigruta po hp tuningu:
    # loss=52.307, mae=5.005




# 1) rfr wg napale:
# frac_val        = 0.15, ##!!
# frac_test       = 0.05  ##!!
    # mse=7.845, rmse=2.801,   mae=1.715
    # to jest liczone w mojej metodzie evaluate
    # mse=7.874, rmse=2.806,   mae=1.723
    # to jest liczone w mojej metodzie evaluate
    
# 2) rfr wg napale:
# frac_val        = 0.15, ##!!
# frac_test       = 0.15  ##!!
    # mse=9.551, rmse=3.09,   mae=1.79
    # to jest liczone w mojej metodzie evaluate
    # mse=9.513, rmse=3.084,   mae=1.8
    # to jest liczone w mojej metodzie evaluate
    
# 3) rfr wg napale:
# frac_val        = 0.15, ##!!
# frac_test       = 0.15  ##!!
# WYRZUCILAM t_zasilania I t_powrotu
    # mse=9.039, rmse=3.006,   mae=1.777
    # to jest liczone w mojej metodzie evaluate
    # mse=9.578, rmse=3.095,   mae=1.794
    # to jest liczone w mojej metodzie evaluate

# 3) rfr wg soft sensor:
# frac_val        = 0.15, ##!!
# frac_test       = 0.15  ##!!
# WYRZUCILAM t_zasilania I t_powrotu
    # mse=8.505, rmse=2.916,   mae=1.714
    # to jest liczone w mojej metodzie evaluate
    
    # mse=8.488, rmse=2.914,   mae=1.712
    # to jest liczone w mojej metodzie evaluate



