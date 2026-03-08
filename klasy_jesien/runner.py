#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 00:53:26 2025; @author: sylwia
Funkcja główna do uruchamiania eksperymentów
"""
#### kawa i wozetka
import os; import warnings; import logging
import time; from datetime import datetime
from pathlib import Path
import numpy as np; import pandas as pd
import matplotlib.pyplot as plt; import seaborn as sns

logger = logging.getLogger(__name__)

pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.options.mode.chained_assignment = None  # wyłącza SettingWithCopyWarning (specyficzne warningi pandas)

# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ustawienia plots:
plt.style.use("seaborn-v0_8-poster")
print(plt.rcParams["figure.facecolor"])  # powinno być 'white'
print(plt.rcParams["axes.titlesize"])    # powinna być np. 18.0 (duża czcionka)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import subprocess
def playsound(path): #### bo to lepsze niż biblioteka playsound
    try:
        subprocess.run(['aplay', path], check=True, capture_output=True)
    except Exception:
        print(f"(Nie można odtworzyć dźwięku: {path})")
import tensorflow as tf
pd.options.display.max_columns = 10
#
from klasy_data.d1_pds import PdsSetup
from klasy_data.d3_kds import KdsSetup
from klasy_jesien.archit1_shallow import ShallowArchitecture
from klasy_jesien.archit2_deep import DeepArchitecture
from klasy_jesien.behav1_single import SingleBehavior
from klasy_jesien.behav2_dual import DualBehavior
from klasy_jesien.handler_dfres import DfResHandler

def setup_logging():
    """Konfiguracja loggera"""
    logging.basicConfig(
        level=logging.DEBUG,
        filename='zz_project_log.log',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def setup_gpu():
    """Konfiguracja GPU - wzrost pamięci na żądanie"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def run_experiment(
    # Konfiguracja
    datacfg,
    wyborkols,
    timestampplus,
    corobie,

    # Parametry główne
    isdeep=False,
    isdual=False,
    ilerunow=1,
    smtype="rfr",
    dmtype="bigruta",
    
    # Parametry deep model (None dla shallow)
    epochs=None,
    batchsize=None,
    seqlen=None,
    
    # Hiperparametry deep model (None dla shallow)
    dmhps_Mono=None,
    dmhps_rezimA=None,
    dmhps_rezimB=None,
    
    # Flagi
    sprkod=True,
    testsetatlast=False,
    
    # Oprawa artystyczna
    kolor_actual="limegreen",
    kolor_pred="navy",
    kolor_res="purple",
    arabian_wav='../mixkit-arabian.wav',
    game_wav='../mixkit-game-notification.wav'
    ):
    """
    Uruchamia pełny eksperyment treningowy.
    
    Parameters
    ----------
    datacfg : DataConfig
        Konfiguracja danych
    wyborkols : str
        Wybór kolumn
    timestampplus : str
        Timestamp z dodatkowym opisem
    corobie : str
        Opis co robisz
    arabian_wav : str
        Ścieżka do pliku audio na koniec
    isdeep : bool
        Czy architektura deep
    isdual : bool
        Czy podejście dual-regime
    ilerunow : int
        Liczba runów
    SMTYPE : str, optional
        Typ modelu shallow
    epochs, batchsize, seqlen : int, optional
        Parametry dla deep architecture
    dmhps_Mono, dmhps_rezimA, dmhps_rezimB : dict, optional
        Hiperparametry modeli
    sprkod, testsetatlast : bool, optional
        Flagi
    kolor_actual, kolor_pred, kolor_res : str, optional
        Kolory do wykresów
    
    Returns
    -------
    tuple
        (archit, behav, dfres_handler) - obiekty z wynikami
    """
    # Setup
    logger = setup_logging()
    setup_gpu()
    start_time = time.time()
    
    try:
        print("\n" + "~"*70)
        print("RUNNER zaczyna robić robotę")
        print("~"*70)
        logger.info("\n"*10 + "#"*70)
        
        # ~~~~ Data and Architecture and Behavior ~~~~ 
        
        
        ## Deep
        if isdeep:  
            if epochs is None or batchsize is None or seqlen is None:
                raise ValueError("isdeep=True wymaga parametrów: epochs, batchsize, seqlen")
            d3_kds = KdsSetup(
                datacfg_instance=datacfg,
                wybor_kols=wyborkols,
                epochs=epochs,
                batchsize=batchsize,
                seqlen=seqlen,
                timestampplus=timestampplus,
                sprkod=sprkod,
                testsetatlast=testsetatlast
            )
            archit = DeepArchitecture(d3_kds, ilerunow, dmtype)
            
            if isdual: ## Deep Dual
                if dmhps_rezimA is None or dmhps_rezimB is None:
                    raise ValueError("isdual=True dla deep wymaga: dmhps_rezimA, dmhps_rezimB")
                behav = DualBehavior(archit)
                results_for_dfres = behav.run_runs_and_get_results_for_dfres(dmhps_rezimA, dmhps_rezimB)
            else: ## Deep Single
                if dmhps_Mono is None:
                    raise ValueError("isdual=False dla deep wymaga: dmhps_Mono")
                behav = SingleBehavior(archit)
                results_for_dfres = behav.run_runs_and_get_results_for_dfres(dmhps_Mono)
            
            sciezka_play = d3_kds.d1_pds.k1_path
            
            
        ## Shallow
        else:  
            d1_pds = PdsSetup(
                datacfg=datacfg,
                wybor_kols=wyborkols,
                timestampplus=timestampplus,
                sprkod=sprkod,
                testsetatlast=testsetatlast
            )
            archit = ShallowArchitecture(d1_pds, ilerunow, smtype)
            
            if isdual: ## Shallow Dual
                behav = DualBehavior(archit)
                results_for_dfres = behav.run_runs_and_get_results_for_dfres()
            else: ## Shallow Single
                behav = SingleBehavior(archit)
                results_for_dfres = behav.run_runs_and_get_results_for_dfres()
                
                # tylko tu można to zrobić, ale dual nie zakodowane
                print(archit.get_feature_importance())
                print(archit.get_permut_importance(behav.data_for_tvt))
            
            sciezka_play = d1_pds.k1_path
        
        
        
        
        # ~~~~ DfRes Handler ~~~~ 
        print("\n\njestem w klasy_jesien.runner, sciezka_play to:")
        print(sciezka_play)
        
        dfres_handler = DfResHandler(
            *results_for_dfres,
            kolor_actual, kolor_pred, kolor_res,
            sciezka_play, 
            timestampplus
            )
        
        # ~~~~ Przetwarzanie wyników ~~~~ 
        dfres_handler.plot_all_in_one(behav.pred_best) 
        
           
        all_runs_metrics_dicts_list = (dfres_handler
                                       .calculate_metrics_dicts_list_from_all_runs_predictions()
                                       )
        archit.calc_and_save_all_runs_summary_stats_from_predictions(
                                                    all_runs_metrics_dicts_list
                                                    )
        
        best_metrics_dict = dfres_handler.calculate_metrics_from_one_predictions(behav.pred_best)
        archit.save_best_run_results_dict(best_metrics_dict)
        
        
        # ~~~~ Zakończenie ~~~~ 
        elapsed_time = (time.time() - start_time) / 60
        timestamp_end = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        logger.info(f'\n\n{timestamp_end} --- SKONCZYL LICZYC')
        logger.info(f"\n\nCzas robienia: {elapsed_time:.6f} minut")
        
        playsound(arabian_wav)
        
        print("\n\n" + "~"*70)
        print("ZAKOŃCZONE")
        print(f"CO ROBIĘ: {corobie}")
        print(f'{timestamp_end} --- SKONCZYL LICZYC')
        print(f"Czas robienia: {elapsed_time:.6f} minut")
        print("~"*70 + "\n")
        
        return archit, behav, dfres_handler

    except Exception as e:
        # Zakończenie BŁĄD
        elapsed_time = (time.time() - start_time) / 60
        timestamp_end = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        logger.error(f'\n\n{timestamp_end} --- BŁĄD PODCZAS LICZENIA')
        logger.error(f"Błąd: {str(e)}", exc_info=True)
        logger.error(f"\n\nCzas do błędu: {elapsed_time:.6f} minut")
        
        # Odtwórz inną muzykę przy błędzie
        try:
            playsound(game_wav)
        except:
            print("(Nie można odtworzyć dźwięku błędu)")
        
        print("\n\n" + "~"*70)
        print("❌ BŁĄD - PRZERWANO OBLICZENIA ❌")
        print(f"CO ROBIŁEM: {corobie}")
        print(f'{timestamp_end} --- BŁĄD')
        print(f"Czas do błędu: {elapsed_time:.6f} minut")
        print(f"\nBłąd: {str(e)}")
        print("~"*70 + "\n")
        
        # Ponownie rzuć wyjątek żeby zobaczyć pełny traceback
        raise



