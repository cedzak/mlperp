# Hyperparameter Tuning Module

Automatyczny tuning hiperparametrów dla deep learning models przy użyciu Keras Tuner.

## 📦 Pliki w module

```
├── tuner_main.py              # Główna klasa DeepModelTuner
├── run_tuner.py               # Prosty skrypt do uruchomienia
├── analyze_tuner_results.py   # Analiza wyników
├── tuner_configs.py           # Przykładowe konfiguracje
└── README_TUNER.md            # Ten plik
```

## 🚀 Quick Start

### 1. Instalacja zależności

```bash
pip install keras-tuner
```

### 2. Uruchom tuning

Edytuj parametry w `run_tuner.py`:

```python
# Strategia tuningu
TUNER_STRATEGY = "bayesian"  # "bayesian", "random", "hyperband"

# Typ modelu
DMTYPE = "bigruta"  # "bigruta" lub "lstm"

# Parametry tuningu
MAX_TRIALS = 30  # Ile różnych konfiguracji przetestować
ILERUNOW_PER_TRIAL = 3  # Ile runów dla każdej konfiguracji
```

Uruchom:

```bash
python run_tuner.py
```

### 3. Analizuj wyniki

```bash
python analyze_tuner_results.py <ścieżka_do_katalogu_tunera>
```

Lub automatycznie znajdzie najnowszy:

```bash
python analyze_tuner_results.py
```

### 4. Użyj najlepszych hiperparametrów

Skopiuj kod z pliku `best_hps_code__<timestamp>.py` do `inputs/i_mdeep_hps.py`:

```python
wynik_tuned_2025 = DeepModelHps(
    cfg_id="tuned_2025",
    initial_learning_rate=0.005,
    momentum=0.8,
    dropout=0.2,
    # ... reszta parametrów z tuningu
)
```

Użyj w `adamgda0.py`:

```python
DMHPS_MONO = wynik_tuned_2025
```

## 📊 Wyniki tuningu

Po zakończeniu tuningu w katalogu `mojerys/gda/tuner_<timestamp>/` znajdziesz:

- `best_hyperparameters__<timestamp>.json` - najlepsze hiperparametry (JSON)
- `best_hps_code__<timestamp>.py` - gotowy kod do skopiowania
- `trials_results__<timestamp>.csv` - wszystkie trials z wynikami
- `tuning_summary__<timestamp>.txt` - podsumowanie statystyczne
- `tuning_progress__<timestamp>.png` - wykres postępu
- `hyperparameter_importance__<timestamp>.png` - znaczenie hiperparametrów

## 🎯 Strategie tuningu

### Bayesian Optimization (rekomendowane ⭐)
```python
TUNER_STRATEGY = "bayesian"
MAX_TRIALS = 30-50
```
- Używa modelu probabilistycznego
- Najefektywniejszy dla deep learning
- Czas: 4-12 godzin (zależnie od MAX_TRIALS)

### Random Search
```python
TUNER_STRATEGY = "random"
MAX_TRIALS = 50-100
```
- Losowe próbkowanie
- Dobry dla wstępnego rozpoznania
- Czas: 2-8 godzin

### Hyperband
```python
TUNER_STRATEGY = "hyperband"
MAX_TRIALS = 100-200
```
- Adaptacyjny z wczesnym zatrzymywaniem
- Testuje wiele konfiguracji
- Czas: 8-24 godziny

## 🔧 Tunowane hiperparametry

### Model BIGRUTA
- `initial_learning_rate`: 0.0001 - 0.1 (log scale)
- `momentum`: [0.5, 0.8, 0.9]
- `dropout`: 0.0 - 0.5 (step 0.1)
- `regl2`: [0.0, 1e-5, 1e-4, 1e-3]
- `patience_lr`: [10, 20, 50, 80]
- `patience_es`: [10, 20, 50, 80]
- `min_delta`: [0.01, 0.1, 0.5]
- `variant_gru`: ["1xGRU", "2xGRU"]
- `variant_ta`: ["simple_TA", "directional_TA"]
- `variant_ile_nodow`: [32, 64, 128, 256]

### Model LSTM
- Podobnie jak wyżej
- `variant_ile_nodow`: [16, 32, 64, 128]

## 📈 Przykładowe scenariusze

### Szybki test (30 min)
```python
TUNER_STRATEGY = "random"
MAX_TRIALS = 5
ILERUNOW_PER_TRIAL = 1
EPOCHS = 50
```

### Standardowy tuning (8 godzin)
```python
TUNER_STRATEGY = "bayesian"
MAX_TRIALS = 30
ILERUNOW_PER_TRIAL = 3
EPOCHS = 200
```

### Intensywny tuning (24 godziny)
```python
TUNER_STRATEGY = "bayesian"
MAX_TRIALS = 100
ILERUNOW_PER_TRIAL = 5
EPOCHS = 300
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# Zmniejsz:
BATCHSIZE = 64  # zamiast 128
SEQLEN = 48     # zamiast 72
```

### Tuning trwa za długo
```python
# Zmniejsz:
MAX_TRIALS = 20         # zamiast 50
ILERUNOW_PER_TRIAL = 1  # zamiast 3
EPOCHS = 150            # zamiast 200
```

### Wyniki nie poprawiają baseline
```python
# Zwiększ:
MAX_TRIALS = 100  # więcej prób
# Lub zmień strategię:
TUNER_STRATEGY = "hyperband"
```

## 💡 Best Practices

1. **Zacznij od małych testów**
   - 5-10 trials żeby sprawdzić czy wszystko działa

2. **Zmniejsz epochs dla tuningu**
   - 150-200 zamiast 500
   - EarlyStopping i tak zatrzyma wcześniej

3. **Użyj ilerunow_per_trial=3**
   - Dobre balance między stabilnością a czasem

4. **Monitoruj postęp**
   - Sprawdzaj `tuner_log.log`
   - Użyj `nvidia-smi -l 1` dla GPU

5. **Zapisuj wyniki**
   - Keras Tuner automatycznie zapisuje
   - Możesz wznowić przerwany tuning

## 📖 Pełna dokumentacja

Zobacz `tuner_configs.py` dla:
- Szczegółowej dokumentacji
- Zaawansowanych przykładów
- Customizacji

## 🤝 Integracja z istniejącym projektem

Moduł w pełni zintegrowany z Twoją architekturą:
- Używa `DeepArchitecture`, `SingleBehavior`
- Kompatybilny z `DataConfig`, `KdsSetup`
- Zapisuje wyniki w tym samym formacie
- Działa tylko dla **deep models** i **single behavior**

## 🎓 Przykład użycia w kodzie

```python
from tuner_main import DeepModelTuner, setup_logging, setup_gpu
from klasy_data.d0_data import DataConfig
from inputs.i_columns import kols_map_gda

# Setup
setup_logging()
setup_gpu()

# Konfiguracja danych
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
    timestampplus="my_tuning_2025",
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

# Użyj wyników
print("Najlepsze hiperparametry:", best_hps)
```

## 📞 Support

Problemy? Pytania?
- Sprawdź `tuner_log.log`
- Zobacz `tuner_configs.py` dla pełnej dokumentacji
- Zadaj pytanie Claude! 😊

---

**Wersja**: 1.0  
**Data**: 2025-12-09  
**Autor**: Sylwia (z pomocą Claude)  
**Licencja**: Do użytku wewnętrznego
