#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Tuner Run - uproszczony skrypt do uruchomienia tuningu hiperparametrów

JAK UŻYWAĆ:
-----------

### Opcja 1: Gotowa konfiguracja (ŁATWE! ⭐)
1. Ustaw: USE_PRESET_CONFIG = True
2. Wybierz: PRESET_NAME = "overnight"  # lub inna (patrz niżej)
3. Uruchom: python run_tuner.py

### Opcja 2: Własna konfiguracja
1. Ustaw: USE_PRESET_CONFIG = False
2. Edytuj parametry poniżej
3. Uruchom: python run_tuner.py

DOSTĘPNE PRESET:
----------------
"quick_test"   → 30 min   - test czy działa
"exploration"  → 2-4h     - szybkie rozpoznanie
"standard"     → 4-8h     - solidny tuning
"overnight" ⭐ → 8-12h    - REKOMENDOWANE na noc!
"intensive"    → 24-48h   - bardzo dokładny
"lstm"         → 4-8h     - dla modeli LSTM
"hyperband"    → 8-16h    - adaptacyjne
"low_memory"   → 4-8h     - dla słabszych GPU

"""
import os
print("Current katalog:", os.getcwd())

from datetime import datetime
from inputs.i_columns import kols_map_gda
from klasy_data.d0_data import DataConfig
from klasy_tuner.tuner_main import DeepModelTuner, setup_logging, setup_gpu
from klasy_tuner.tuner_configs import (CONFIG_QUICK_TEST, CONFIG_STANDARD, CONFIG_OVERNIGHT,
                            CONFIG_INTENSIVE, CONFIG_LSTM, CONFIG_HYPERBAND,
                            CONFIG_EXPLORATION, CONFIG_LOW_MEMORY)


#### ====== WYBIERZ KONFIGURACJĘ ======

# Opcja 1: Użyj gotowej konfiguracji (ŁATWE!)
USE_PRESET_CONFIG = True  # True = użyj gotowej, False = własna konfiguracja

# Wybierz którą:
PRESET_NAME = "overnight"  # "quick_test", "standard", "overnight", "intensive", "lstm", "hyperband"

# Mapowanie nazw na konfiguracje
PRESET_CONFIGS = {
    "quick_test": CONFIG_QUICK_TEST,
    "exploration": CONFIG_EXPLORATION,
    "standard": CONFIG_STANDARD,
    "overnight": CONFIG_OVERNIGHT,  # ⭐ REKOMENDOWANE NA NOC
    "intensive": CONFIG_INTENSIVE,
    "lstm": CONFIG_LSTM,
    "hyperband": CONFIG_HYPERBAND,
    "low_memory": CONFIG_LOW_MEMORY
}


#### ====== KONFIGURACJA TUNINGU ======

if USE_PRESET_CONFIG:
    # Użyj gotowej konfiguracji
    config = PRESET_CONFIGS[PRESET_NAME]
    TUNER_STRATEGY = config['tuner_strategy']
    MAX_TRIALS = config['max_trials']
    ILERUNOW_PER_TRIAL = config['ilerunow_per_trial']
    EPOCHS = config['epochs']
    BATCHSIZE = config['batchsize']
    SEQLEN = config['seqlen']
    DMTYPE = config['dmtype']
    
else:
    # Opcja 2: Własna konfiguracja (RĘCZNE USTAWIENIA)
    TUNER_STRATEGY = "bayesian"  # "bayesian", "random", "hyperband"
    DMTYPE = "bigruta"  # "bigruta" lub "lstm"
    EPOCHS = 200
    BATCHSIZE = 128
    SEQLEN = 72
    MAX_TRIALS = 30
    ILERUNOW_PER_TRIAL = 3

# Parametry wspólne (niezależnie od wyboru)
WYBORKOLS_GDA = "gda_deepheat_bezsincos"
SPRKOD = False
TESTSETATLAST = False

# Timestamp
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
TIMESTAMPPLUS = f"tuner_{DMTYPE}_{TUNER_STRATEGY}_{TIMESTAMP}"


#### ====== KONFIGURACJA DANYCH ======
DATACFG_GDA = DataConfig(
    projekt_akronim='gda',
    plik_dane='../mojecsvy/gda/cieplo_withlags__sg.csv',
    kols_map=kols_map_gda,
    kol_target='GJ_heat',
    kol_flagA='is_weekend',
    intkols_to_scale=[]
)


#### ====== URUCHOMIENIE ======
if __name__ == "__main__":
    print('\n\n\n' + "="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)
    
    if USE_PRESET_CONFIG:
        print(f"\nUżywam gotowej konfiguracji: {PRESET_NAME.upper()}")
    else:
        print(f"\nUżywam własnej konfiguracji")
    
    print(f"\nParametry:")
    print(f"  Strategia:         {TUNER_STRATEGY}")
    print(f"  Model:             {DMTYPE}")
    print(f"  Max trials:        {MAX_TRIALS}")
    print(f"  Runs per trial:    {ILERUNOW_PER_TRIAL}")
    print(f"  Epochs per run:    {EPOCHS}")
    print(f"  Batch size:        {BATCHSIZE}")
    print(f"  Sequence length:   {SEQLEN}")
    
    # Szacowany czas
    estimated_hours = (MAX_TRIALS * ILERUNOW_PER_TRIAL * EPOCHS * 0.5) / 3600  # grube oszacowanie
    print(f"\n  Szacowany czas:    ~{estimated_hours:.1f}h")
    print("="*70 + "\n")
    
    # Setup
    setup_logging()
    setup_gpu()
    
    # Stwórz tuner
    tuner = DeepModelTuner(
        datacfg=DATACFG_GDA,
        wyborkols=WYBORKOLS_GDA,
        timestampplus=TIMESTAMPPLUS,
        
        # Parametry stałe
        epochs=EPOCHS,
        batchsize=BATCHSIZE,
        seqlen=SEQLEN,
        dmtype=DMTYPE,
        
        # Parametry tuningu
        ilerunow_per_trial=ILERUNOW_PER_TRIAL,
        max_trials=MAX_TRIALS,
        tuner_strategy=TUNER_STRATEGY,
        
        # Opcje
        sprkod=SPRKOD,
        testsetatlast=TESTSETATLAST
    )
    
    # Uruchom tuning
    best_hps = tuner.run_tuning()
    
    # Wyświetl wyniki
    print("\n\n" + "="*70)
    print("NAJLEPSZE HIPERPARAMETRY:")
    print("="*70)
    for key, value in best_hps.items():
        print(f"  {key:<25} {value}")
    print("="*70)
    print("\nSkopiuj te wartości do inputs/i_mdeep_hps.py")
    print("="*70 + "\n\n")
