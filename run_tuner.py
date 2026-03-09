#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HP Tuning dla projektu gdaciep — bigruta i enc_dec, seq2seq (aligned 24h).

DOSTĘPNE PRESETY:
-----------------
"quick_test"   → ~30 min  - test czy działa
"exploration"  → ~2h      - szybkie rozpoznanie ← TERAZ
"standard"     → 4-8h     - solidny tuning
"overnight" ⭐ → 8-12h    - REKOMENDOWANE na noc!
"intensive"    → 24-48h   - bardzo dokładny
"hyperband"    → 8-16h    - adaptacyjne

JAK UŻYWAĆ:
-----------
1. Wybierz PRESET_NAME poniżej
2. Uruchom: python run_tuner.py
3. Wyniki lądują w mojerys/gdaciep/tuner_*/
"""
import os
print("Current katalog:", os.getcwd())

from datetime import datetime
from inputs.i_columns import kols_map_gdaciep
from klasy_data.d0_data import DataConfig
from klasy_tuner.tuner_main import DeepModelTuner, setup_logging, setup_gpu
from klasy_tuner.tuner_configs import (CONFIG_QUICK_TEST, CONFIG_STANDARD, CONFIG_OVERNIGHT,
                            CONFIG_INTENSIVE, CONFIG_LSTM, CONFIG_HYPERBAND,
                            CONFIG_EXPLORATION, CONFIG_LOW_MEMORY)


#### ====== WYBIERZ PRESET ======
PRESET_NAME = "exploration"  # "quick_test", "standard", "overnight", "intensive", "hyperband"

PRESET_CONFIGS = {
    "quick_test":  CONFIG_QUICK_TEST,
    "exploration": CONFIG_EXPLORATION,
    "standard":    CONFIG_STANDARD,
    "overnight":   CONFIG_OVERNIGHT,
    "intensive":   CONFIG_INTENSIVE,
    "hyperband":   CONFIG_HYPERBAND,
    "low_memory":  CONFIG_LOW_MEMORY,
}

config = PRESET_CONFIGS[PRESET_NAME]

#### ====== PARAMETRY TUNINGU (z presetu) ======
TUNER_STRATEGY     = config['tuner_strategy']
MAX_TRIALS         = config['max_trials']
ILERUNOW_PER_TRIAL = config['ilerunow_per_trial']
EPOCHS             = config['epochs']
BATCHSIZE          = config['batchsize']

#### ====== PARAMETRY PROJEKTU (nadpisują preset) ======
SEQLEN  = 24           # aligned = 1 doba
TASK    = "seq2seq"    # aligned mode zawsze seq2seq
DMTYPES = ["enc_dec"]  # bigruta już gotowy

WYBORKOLS   = 'top10_features_permut_importance'
SPRKOD      = False
TESTSETATLAST = False


#### ====== KONFIGURACJA DANYCH ======
DATACFG_CIEPLO = DataConfig(
    projekt_akronim = 'gdaciep',
    plik_dane       = '../mojecsvy/gdaciep/cieplo_pogoda_czas__sg.csv',
    kols_map        = kols_map_gdaciep,
    kol_target      = 'MW_th',
    kol_flagA       = 'is_weekend',
    intkols_to_scale= [],
    frac_val        = 0.15,
    frac_test       = 0.15,
    aligned_krotnosc_24h = True,
)


#### ====== URUCHOMIENIE ======
if __name__ == "__main__":
    setup_logging('zz_tuner_run.log')
    setup_gpu()
    # - setup_logging() — konfiguruje logger Pythona (poziom INFO, format z timestampem) żeby komunikaty z tunera ładnie lądowały w logu zamiast jako surowy print                   
    # - setup_gpu() — konfiguruje TensorFlow żeby nie zajmowało od razu całej pamięci GPU (memory_growth=True), tylko przydzielało ją stopniowo w miarę potrzeb            

    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")

    print('\n\n' + "="*70)
    print(f"HP TUNING — {PRESET_NAME.upper()}")
    print(f"  Strategia:      {TUNER_STRATEGY}")
    print(f"  Task:           {TASK}")
    print(f"  Architektury:   {DMTYPES}")
    print(f"  Max trials:     {MAX_TRIALS} × {len(DMTYPES)} architektur")
    print(f"  Runs/trial:     {ILERUNOW_PER_TRIAL}")
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Seqlen:         {SEQLEN}")
    estimated = (MAX_TRIALS * ILERUNOW_PER_TRIAL * EPOCHS * 0.5) / 3600
    print(f"  Szacowany czas: ~{estimated:.1f}h × {len(DMTYPES)} = ~{estimated*len(DMTYPES):.1f}h łącznie")
    print("="*70 + "\n")

    for dmtype in DMTYPES:
        print(f"\n{'='*70}")
        print(f"ARCHITEKTURA: {dmtype.upper()}")
        print(f"{'='*70}\n")

        timestampplus = f"tuner_{dmtype}_{TUNER_STRATEGY}_{TIMESTAMP}"

        tuner = DeepModelTuner(
            datacfg            = DATACFG_CIEPLO,
            wyborkols          = WYBORKOLS,
            timestampplus      = timestampplus,
            epochs             = EPOCHS,
            batchsize          = BATCHSIZE,
            seqlen             = SEQLEN,
            dmtype             = dmtype,
            task               = TASK,
            ilerunow_per_trial = ILERUNOW_PER_TRIAL,
            max_trials         = MAX_TRIALS,
            tuner_strategy     = TUNER_STRATEGY,
            sprkod             = SPRKOD,
            testsetatlast      = TESTSETATLAST,
        )

        best_hps = tuner.run_tuning()

        print(f"\nNAJLEPSZE HP dla {dmtype}:")
        for key, value in best_hps.items():
            print(f"  {key:<25} {value}")
        print(f"\nSkopiuj do inputs/i_mdeep_hps.py")
        print("="*70 + "\n")
