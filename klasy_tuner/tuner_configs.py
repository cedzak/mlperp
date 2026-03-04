#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
HYPERPARAMETER TUNING - DOKUMENTACJA UŻYCIA
=============================================================================

Ten moduł zawiera narzędzia do automatycznego tuningu hiperparametrów 
dla deep learning models przy użyciu Keras Tuner.

PLIKI W MODULE:
---------------
1. tuner_main.py              - Główna klasa DeepModelTuner
2. run_tuner.py               - Prosty skrypt do uruchomienia tuningu
3. analyze_tuner_results.py   - Analiza wyników tuningu
4. tuner_configs.py           - Przykładowe konfiguracje (ten plik)


PODSTAWOWE UŻYCIE:
------------------

### Krok 1: Wybierz konfigurację w run_tuner.py

```python
# Opcja A: Użyj gotowej konfiguracji (ŁATWE! ⭐)
USE_PRESET_CONFIG = True
PRESET_NAME = "overnight"  # Wybierz: quick_test, standard, overnight, intensive, itp.

# Opcja B: Własna konfiguracja
USE_PRESET_CONFIG = False
TUNER_STRATEGY = "bayesian"
MAX_TRIALS = 30
# ... reszta parametrów
```

### Krok 2: Uruchom tuning

```python
python run_tuner.py
```

### Krok 3: Analizuj wyniki

```python
python analyze_tuner_results.py <ścieżka_do_katalogu_tunera>
```

### Krok 4: Użyj najlepszych hiperparametrów

Skopiuj najlepsze hiperparametry z pliku:
  `best_hps_code__<timestamp>.py`
  
Do pliku:
  `inputs/i_mdeep_hps.py`


DOSTĘPNE PRESET KONFIGURACJE:
------------------------------

Nazwa w PRESET_NAME     │ Czas      │ Opis
────────────────────────┼───────────┼─────────────────────────────────
"quick_test"            │ 30 min    │ Test czy działa
"exploration"           │ 2-4h      │ Szybkie rozpoznanie przestrzeni
"standard"              │ 4-8h      │ Solidny tuning
"overnight" ⭐          │ 8-12h     │ REKOMENDOWANE na noc!
"intensive"             │ 24-48h    │ Bardzo dokładny tuning
"lstm"                  │ 4-8h      │ Dla modeli LSTM
"hyperband"             │ 8-16h     │ Adaptacyjne przeszukiwanie
"low_memory"            │ 4-8h      │ Dla słabszych GPU


ZAAWANSOWANE UŻYCIE:
--------------------

### Programowe użycie w własnym skrypcie:

```python
from tuner_main import DeepModelTuner, setup_logging, setup_gpu
from klasy_data.d0_data import DataConfig
from inputs.i_columns import kols_map_gda

# Setup
setup_logging()
setup_gpu()

# Konfiguracja
datacfg = DataConfig(
    projekt_akronim='gda',
    plik_dane='../mojecsvy/gda/cieplo_withlags__sg.csv',
    kols_map=kols_map_gda,
    kol_target='GJ_heat',
    kol_flagA='is_weekend',
    intkols_to_scale=[]
)

# Stwórz tuner
tuner = DeepModelTuner(
    datacfg=datacfg,
    wyborkols="gda_deepheat_bezsincos",
    timestampplus="tuner_custom_2025",
    epochs=200,
    batchsize=128,
    seqlen=72,
    dmtype="bigruta",
    ilerunow_per_trial=3,
    max_trials=50,
    tuner_strategy="bayesian"
)

# Uruchom
best_hps = tuner.run_tuning()
```


STRATEGIE TUNINGU:
------------------

1. BAYESIAN OPTIMIZATION (rekomendowane)
   - Używa modelu probabilistycznego do przewidywania obiecujących regionów
   - Najefektywniejszy dla drogich funkcji (deep learning)
   - Dobry dla 20-100 trials
   
   ```python
   tuner_strategy="bayesian"
   max_trials=50
   ```

2. RANDOM SEARCH
   - Losowe próbkowanie przestrzeni hiperparametrów
   - Prosty, czasem zaskakująco efektywny
   - Dobry dla wstępnego rozpoznania
   
   ```python
   tuner_strategy="random"
   max_trials=100
   ```

3. HYPERBAND
   - Adaptacyjny algorytm z wczesnym zatrzymywaniem
   - Testuje wiele konfiguracji z mniejszą liczbą epok
   - Dobry dla bardzo dużej przestrzeni przeszukiwania
   
   ```python
   tuner_strategy="hyperband"
   max_trials=200  # Może przetestować więcej konfiguracji
   ```


TUNOWANE HIPERPARAMETRY:
-------------------------

### Dla modelu BIGRUTA (BiGRU + Temporal Attention):

1. **initial_learning_rate**: 0.0001 - 0.1 (log scale)
   - Learning rate dla optimizera
   
2. **momentum**: [0.5, 0.8, 0.9]
   - Momentum dla RMSprop
   
3. **dropout**: 0.0 - 0.5 (step 0.1)
   - Dropout rate
   
4. **regl2**: [0.0, 1e-5, 1e-4, 1e-3]
   - L2 regularization
   
5. **patience_lr**: [10, 20, 50, 80]
   - Patience dla ReduceLROnPlateau
   
6. **patience_es**: [10, 20, 50, 80]
   - Patience dla EarlyStopping
   
7. **min_delta**: [0.01, 0.1, 0.5]
   - Minimalna zmiana dla callbacks
   
8. **variant_gru**: ["1xGRU", "2xGRU"]
   - 1 lub 2 warstwy BiGRU
   
9. **variant_ta**: ["simple_TA", "directional_TA"]
   - Typ temporal attention
   
10. **variant_ile_nodow**: [32, 64, 128, 256]
    - Liczba jednostek w warstwach


### Dla modelu LSTM:

1-7: Jak wyżej
8. **variant_ile_nodow**: [16, 32, 64, 128]


PARAMETRY NIE-TUNOWANE (stałe):
--------------------------------

- epochs: Liczba epok treningowych (np. 200-500)
- batchsize: Rozmiar batcha (np. 128)
- seqlen: Długość sekwencji (np. 72)
- dmtype: Typ modelu ("bigruta" lub "lstm")


PARAMETRY TUNINGU:
------------------

**ilerunow_per_trial** (domyślnie: 3)
  - Ile runów wykonać dla każdej konfiguracji hiperparametrów
  - Wynik = średnia RMSE z tych runów
  - Większa wartość = stabilniejsze wyniki, ale dłuższy czas
  - Rekomendacja: 3-5
  
**max_trials** (domyślnie: 50)
  - Maksymalna liczba prób (konfiguracji hiperparametrów)
  - Bayesian: 30-100
  - Random: 100-500
  - Hyperband: 200-1000
  
**executions_per_trial** (domyślnie: 1)
  - Wewnętrzny parametr Keras Tuner
  - Zostaw na 1 (używamy własnego ilerunow_per_trial)


INTERPRETACJA WYNIKÓW:
-----------------------

Po zakończeniu tuningu otrzymasz:

1. **best_hyperparameters__<timestamp>.json**
   - Najlepsze hiperparametry w formacie JSON
   
2. **best_hps_code__<timestamp>.py**
   - Gotowy kod Python do skopiowania
   
3. **trials_results__<timestamp>.csv**
   - Wszystkie trials z wynikami
   
4. **tuning_summary__<timestamp>.txt**
   - Podsumowanie statystyczne
   
5. **tuning_progress__<timestamp>.png**
   - Wykres postępu tuningu
   
6. **hyperparameter_importance__<timestamp>.png**
   - Znaczenie każdego hiperparametru


BEST PRACTICES:
---------------

1. **Zacznij od Random Search**
   - Zrób 20-30 random trials żeby poznać przestrzeń
   
2. **Przejdź do Bayesian**
   - Użyj Bayesian dla dokładniejszej optymalizacji
   
3. **Zmniejsz epochs dla tuningu**
   - Użyj 150-200 epok zamiast 500
   - Przyspiesza tuning, EarlyStopping i tak zatrzyma wcześniej
   
4. **Użyj ilerunow_per_trial=3**
   - Dobre balance między stabilnością a czasem
   
5. **Monitoruj GPU**
   - Użyj `nvidia-smi -l 1` żeby sprawdzić wykorzystanie
   
6. **Zapisuj checkpointy**
   - Keras Tuner automatycznie zapisuje postęp
   - Możesz wznowić przerwany tuning


PRZYKŁADOWE SCENARIUSZE:
-------------------------

### Scenariusz 1: Szybkie rozpoznanie (2-4 godziny)
```python
tuner_strategy="random"
max_trials=20
ilerunow_per_trial=1
epochs=150
```

### Scenariusz 2: Dokładny tuning (8-12 godzin)
```python
tuner_strategy="bayesian"
max_trials=50
ilerunow_per_trial=3
epochs=200
```

### Scenariusz 3: Intensywne przeszukiwanie (24-48 godzin)
```python
tuner_strategy="hyperband"
max_trials=200
ilerunow_per_trial=1
epochs=500
```


TROUBLESHOOTING:
----------------

**Problem: Out of Memory (OOM)**
Rozwiązanie:
  - Zmniejsz batchsize (np. 128 → 64)
  - Zmniejsz seqlen (np. 72 → 48)
  - Zmniejsz maksymalną liczbę jednostek w warstwach

**Problem: Tuning trwa za długo**
Rozwiązanie:
  - Zmniejsz max_trials
  - Zmniejsz epochs
  - Zmniejsz ilerunow_per_trial
  - Użyj "hyperband" zamiast "bayesian"

**Problem: Wyniki nie poprawiają baseline**
Rozwiązanie:
  - Zwiększ max_trials (może nie znalazł dobrych regionów)
  - Sprawdź czy przestrzeń przeszukiwania jest sensowna
  - Rozważ tunowanie innych parametrów (batchsize, seqlen)

**Problem: Kernel died / GPU crash**
Rozwiązanie:
  - Dodaj tf.keras.backend.clear_session() przed każdym trial
  - Zmniejsz batchsize
  - Sprawdź temperatury GPU


INTEGRACJA Z ISTNIEJĄCYM KODEM:
--------------------------------

Po zakończeniu tuningu, użyj najlepszych hiperparametrów w `adamgda0.py`:

```python
# W pliku inputs/i_mdeep_hps.py dodaj:

from inputs.i_mdeep_hps import DeepModelHps

wynik_tuned_2025 = DeepModelHps(
    cfg_id="tuned_2025",
    initial_learning_rate=0.005,  # z tuningu
    momentum=0.8,                  # z tuningu
    dropout=0.2,                   # z tuningu
    regl2=0.0,                     # z tuningu
    patience_lr=20,                # z tuningu
    patience_es=50,                # z tuningu
    min_delta=0.1,                 # z tuningu
    warstwy=["1xGRU", "simple_TA", 64]  # z tuningu
)

# W pliku adamgda0.py użyj:
DMHPS_MONO = wynik_tuned_2025
```


ZAAWANSOWANE CUSTOMIZACJE:
--------------------------

### Dodaj własne hiperparametry do tunowania

Edytuj metodę `build_model()` w `tuner_main.py`:

```python
# Przykład: dodaj batch normalization jako parametr do tunowania
use_batch_norm = hp.Boolean('use_batch_norm', default=False)

if use_batch_norm:
    x = tf.keras.layers.BatchNormalization()(x)
```

### Zmień metrykę optymalizacji

Domyślnie używamy validation RMSE. Możesz zmienić na MAE:

```python
# W objective_function():
mae = mean_absolute_error(self.val_actuals, val_predictions)
return mae  # zamiast rmse
```

### Dodaj własne callbacks podczas tuningu

```python
# W objective_function():
custom_callback = MyCustomCallback()
callbacks_list.append(custom_callback)
```


CONTACT & SUPPORT:
------------------

Pytania? Problemy? Sugestie?
- Sprawdź logi: tuner_log.log
- Sprawdź Keras Tuner docs: https://keras.io/keras_tuner/
- Zadaj pytanie Claude! 😊


=============================================================================
Wersja: 1.0
Data: 2025-12-09
Autor: Sylwia (z pomocą Claude)
=============================================================================
"""

# =============================================================================
# PRZYKŁADOWE KONFIGURACJE
# =============================================================================

# Szybka konfiguracja do testowania (30 min)
CONFIG_QUICK_TEST = {
    'tuner_strategy': 'random',
    'max_trials': 5,
    'ilerunow_per_trial': 1,
    'epochs': 50,
    'batchsize': 128,
    'seqlen': 72,
    'dmtype': 'bigruta'
}

# Konfiguracja standardowa (4-8h)
CONFIG_STANDARD = {
    'tuner_strategy': 'bayesian',
    'max_trials': 30,
    'ilerunow_per_trial': 3,
    'epochs': 200,
    'batchsize': 128,
    'seqlen': 72,
    'dmtype': 'bigruta'
}

# Konfiguracja na noc (8-12h) - REKOMENDOWANA! ⭐
CONFIG_OVERNIGHT = {
    'tuner_strategy': 'bayesian',
    'max_trials': 60,
    'ilerunow_per_trial': 3,
    'epochs': 250,
    'batchsize': 128,
    'seqlen': 72,
    'dmtype': 'bigruta'
}

# Konfiguracja intensywna (24-48h)
CONFIG_INTENSIVE = {
    'tuner_strategy': 'bayesian',
    'max_trials': 100,
    'ilerunow_per_trial': 5,
    'epochs': 300,
    'batchsize': 128,
    'seqlen': 72,
    'dmtype': 'bigruta'
}

# Konfiguracja dla LSTM (4-8h)
CONFIG_LSTM = {
    'tuner_strategy': 'bayesian',
    'max_trials': 50,
    'ilerunow_per_trial': 3,
    'epochs': 200,
    'batchsize': 128,
    'seqlen': 72,
    'dmtype': 'lstm'
}

# Konfiguracja Hyperband - szybkie przeszukiwanie (8-16h)
CONFIG_HYPERBAND = {
    'tuner_strategy': 'hyperband',
    'max_trials': 200,
    'ilerunow_per_trial': 1,
    'epochs': 500,
    'batchsize': 128,
    'seqlen': 72,
    'dmtype': 'bigruta'
}

# Konfiguracja konserwatywna - mało pamięci GPU (4-8h)
CONFIG_LOW_MEMORY = {
    'tuner_strategy': 'bayesian',
    'max_trials': 40,
    'ilerunow_per_trial': 3,
    'epochs': 200,
    'batchsize': 64,    # ← mniejszy batch
    'seqlen': 48,       # ← krótsza sekwencja
    'dmtype': 'bigruta'
}

# Konfiguracja eksploracyjna - poznaj przestrzeń (2-4h)
CONFIG_EXPLORATION = {
    'tuner_strategy': 'random',
    'max_trials': 50,
    'ilerunow_per_trial': 1,  # szybko, bez uśredniania
    'epochs': 150,
    'batchsize': 128,
    'seqlen': 72,
    'dmtype': 'bigruta'
}


def print_config_recommendations():
    """Drukuje rekomendacje konfiguracji"""
    print("\n" + "="*70)
    print("REKOMENDOWANE KONFIGURACJE")
    print("="*70)
    
    configs = {
        "Quick Test (30 min)": CONFIG_QUICK_TEST,
        "Exploration (2-4h)": CONFIG_EXPLORATION,
        "Standard (4-8h)": CONFIG_STANDARD,
        "Overnight (8-12h) ⭐": CONFIG_OVERNIGHT,
        "Intensive (24-48h)": CONFIG_INTENSIVE,
        "LSTM Model (4-8h)": CONFIG_LSTM,
        "Hyperband Fast (8-16h)": CONFIG_HYPERBAND,
        "Low Memory GPU (4-8h)": CONFIG_LOW_MEMORY,
    }
    
    for name, config in configs.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key:<20} {value}")
    
    print("\n" + "="*70)
    print("\nJAK UŻYWAĆ:")
    print("1. W run_tuner.py ustaw: USE_PRESET_CONFIG = True")
    print("2. Wybierz: PRESET_NAME = 'overnight'  # lub inna nazwa")
    print("3. Uruchom: python run_tuner.py")
    print("="*70)


if __name__ == "__main__":
    print_config_recommendations()
