#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Tuning Module dla Deep Models
Używa Keras Tuner do optymalizacji hiperparametrów
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from playsound import playsound

import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns

from klasy_data.d0_data import DataConfig
from klasy_data.d3_kds import KdsSetup
from klasy_jesien.archit2_deep import DeepArchitecture
from klasy_jesien.behav1_single import SingleBehavior

logger = logging.getLogger(__name__)


class DeepModelTuner:
    """
    Klasa do tunowania hiperparametrów dla deep models.
    
    Wspiera:
    - Bayesian Optimization (rekomendowane)
    - Random Search
    - Hyperband
    """
    
    def __init__(
        self,
        datacfg: DataConfig,
        wyborkols: str,
        timestampplus: str,
        
        # Parametry stałe (nie tunowane)
        epochs: int = 500,
        batchsize: int = 128,
        seqlen: int = 72,
        dmtype: str = "bigruta",
        
        # Parametry tuningu
        ilerunow_per_trial: int = 3,  # Ile runów na każdą konfigurację
        max_trials: int = 50,         # Ile konfiguracji przetestować
        executions_per_trial: int = 1, # Keras Tuner internal (zostaw 1)
        
        # Strategie
        tuner_strategy: str = "bayesian",  # "bayesian", "random", "hyperband"
        
        # Opcje
        sprkod: bool = False,
        testsetatlast: bool = False,
        
        # Ścieżki
        tuner_dir: str = None,
        arabian_wav: str = '../mixkit-arabian.wav',
        game_wav: str = '../mixkit-game-notification.wav'
    ):
        """
        Args:
            datacfg: Konfiguracja danych
            wyborkols: Wybór kolumn
            timestampplus: Timestamp eksperymentu
            epochs: Liczba epok treningowych
            batchsize: Rozmiar batcha
            seqlen: Długość sekwencji
            dmtype: Typ modelu deep ("bigruta" lub "lstm")
            ilerunow_per_trial: Ile runów wykonać dla każdej konfiguracji (uśredniamy RMSE)
            max_trials: Maksymalna liczba prób (konfiguracji hiperparametrów)
            executions_per_trial: Wewnętrzny parametr Keras Tuner (zostaw 1)
            tuner_strategy: Strategia tuningu ("bayesian", "random", "hyperband")
            sprkod: Czy tryb szybkiego sprawdzania kodu
            testsetatlast: Czy używać test setu zamiast validation
            tuner_dir: Katalog do zapisu wyników tunera
        """
        self.datacfg = datacfg
        self.wyborkols = wyborkols
        self.timestampplus = timestampplus
        
        self.epochs = epochs
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.dmtype = dmtype
        
        self.ilerunow_per_trial = ilerunow_per_trial
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.tuner_strategy = tuner_strategy.lower()
        
        self.sprkod = sprkod
        self.testsetatlast = testsetatlast
        
        self.arabian_wav = arabian_wav
        self.game_wav = game_wav
        
        # Setup directories
        if tuner_dir is None:
            base_dir = Path(datacfg.mojerys_path).parent
            self.tuner_dir = base_dir / "mojerys" / datacfg.projekt_akronim / f"tuner_{timestampplus}"
        else:
            self.tuner_dir = Path(tuner_dir)
        
        self.tuner_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize data pipeline
        logger.info("Inicjalizacja pipeline'u danych...")
        self.d3_kds = KdsSetup(
            datacfg_instance=datacfg,
            wybor_kols=wyborkols,
            epochs=epochs,
            batchsize=batchsize,
            seqlen=seqlen,
            timestampplus=timestampplus,
            sprkod=sprkod,
            testsetatlast=testsetatlast
        )
        
        # Przygotuj dane do treningu
        self.train_kds = (self.d3_kds.train_kdict["kds"]
                         .shuffle(buffer_size=self.d3_kds.train_kdict["ilosc_kluskow"])
                         .cache()
                         .prefetch(tf.data.AUTOTUNE))
        
        self.val_kds = (self.d3_kds.val_kdict["kds"]
                       .cache()
                       .prefetch(tf.data.AUTOTUNE))
        
        self.val_actuals = self.d3_kds.val_kdict["y_actuals"]
        
        logger.info("Pipeline danych gotowy!")
        
    
    def _setup_logging(self):
        """Konfiguracja loggera dla tunera"""
        log_file = self.tuner_dir / "tuner_log.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    
    def build_model(self, hp):
        """
        Buduje model z hiperparametrami do tunowania.
        
        Ta funkcja jest wywoływana przez Keras Tuner dla każdej próby.
        
        Args:
            hp: HyperParameters object from Keras Tuner
            
        Returns:
            Compiled Keras model
        """
        # Zbuduj model używając architektury
        # Tworzymy tymczasową instancję DeepArchitecture tylko dla budowy modelu
        archit = DeepArchitecture(
            kds_setup_instance=self.d3_kds,
            ilerunow=1,  # Nie ma znaczenia dla build()
            dm_type=self.dmtype
        )
        
        # Definiuj przestrzeń hiperparametrów
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Learning rate - log scale
        initial_learning_rate = hp.Float(
            'initial_learning_rate',
            min_value=1e-4,
            max_value=1e-1,
            sampling='log',
            default=1e-2
        )
        
        # Momentum
        momentum = hp.Choice(
            'momentum',
            values=[0.5, 0.8, 0.9],
            default=0.8
        )
        
        # Dropout
        dropout = hp.Float(
            'dropout',
            min_value=0.0,
            max_value=0.5,
            step=0.1,
            default=0.2
        )
        
        # L2 regularization
        regl2 = hp.Choice(
            'regl2',
            values=[0.0, 1e-5, 1e-4, 1e-3],
            default=0.0
        )
        
        # Patience for LR reduction
        patience_lr = hp.Choice(
            'patience_lr',
            values=[10, 20, 50, 80],
            default=20
        )
        
        # Patience for Early Stopping
        patience_es = hp.Choice(
            'patience_es',
            values=[10, 20, 50, 80],
            default=50
        )
        
        # Min delta
        min_delta = hp.Choice(
            'min_delta',
            values=[0.01, 0.1, 0.5],
            default=0.1
        )
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Architektura warstw
        
        if self.dmtype == "bigruta":
            # Typ GRU
            variant_gru = hp.Choice(
                'variant_gru',
                values=["1xGRU", "2xGRU"],
                default="1xGRU"
            )
            
            # Typ Temporal Attention
            variant_ta = hp.Choice(
                'variant_ta',
                values=["simple_TA", "directional_TA"],
                default="simple_TA"
            )
            
            # Liczba jednostek
            variant_ile_nodow = hp.Choice(
                'variant_ile_nodow',
                values=[32, 64, 128, 256],
                default=64
            )
            
            warstwy = [variant_gru, variant_ta, variant_ile_nodow]
            
        else:  # lstm
            # Typ LSTM (na razie tylko jeden)
            variant_lstm = "lstm"
            
            # Duplikacja (czy 2 warstwy LSTM)
            variant_duplicate = hp.Choice(
                'variant_duplicate',
                values=["lstm", "lstm"],  # Placeholder - możesz rozszerzyć
                default="lstm"
            )
            
            # Liczba jednostek
            variant_ile_nodow = hp.Choice(
                'variant_ile_nodow',
                values=[16, 32, 64, 128],
                default=32
            )
            
            warstwy = [variant_lstm, variant_duplicate, variant_ile_nodow]
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Stwórz obiekt DeepModelHps z wytunowanymi parametrami
        from inputs.i_mdeep_hps import DeepModelHps
        
        dmhps = DeepModelHps(
            cfg_id=f"tuner_trial_{hp.values}",
            initial_learning_rate=initial_learning_rate,
            momentum=momentum,
            dropout=dropout,
            regl2=regl2,
            patience_lr=patience_lr,
            patience_es=patience_es,
            min_delta=min_delta,
            warstwy=warstwy
        )
        
        # Zbuduj model
        model = archit.build(dmhps)
        
        return model
    
    
    def objective_function(self, hp):
        """
        Funkcja celu do optymalizacji.
        
        Dla każdej konfiguracji hiperparametrów:
        1. Buduje model
        2. Wykonuje `ilerunow_per_trial` runów
        3. Zwraca średnie RMSE z validation set
        
        Args:
            hp: HyperParameters object
            
        Returns:
            float: Średnie validation RMSE (do minimalizacji)
        """
        # Zbuduj model
        model = self.build_model(hp)
        
        # Przygotuj hiperparametry
        from inputs.i_mdeep_hps import DeepModelHps
        
        if self.dmtype == "bigruta":
            warstwy = [
                hp.get('variant_gru'),
                hp.get('variant_ta'),
                hp.get('variant_ile_nodow')
            ]
        else:
            warstwy = [
                "lstm",
                hp.get('variant_duplicate'),
                hp.get('variant_ile_nodow')
            ]
        
        dmhps = DeepModelHps(
            cfg_id=f"tuner_trial",
            initial_learning_rate=hp.get('initial_learning_rate'),
            momentum=hp.get('momentum'),
            dropout=hp.get('dropout'),
            regl2=hp.get('regl2'),
            patience_lr=hp.get('patience_lr'),
            patience_es=hp.get('patience_es'),
            min_delta=hp.get('min_delta'),
            warstwy=warstwy
        )
        
        # Wykonaj multiple runs i uśrednij wyniki
        rmse_list = []
        
        for run_idx in range(self.ilerunow_per_trial):
            logger.info(f"Trial run {run_idx + 1}/{self.ilerunow_per_trial}")
            
            # Clear memory przed każdym runem
            tf.keras.backend.clear_session()
            
            # Rebuild model dla każdego runu (świeża inicjalizacja wag)
            archit = DeepArchitecture(
                kds_setup_instance=self.d3_kds,
                ilerunow=1,
                dm_type=self.dmtype
            )
            model = archit.build(dmhps)
            
            # Callbacks
            callbacks_list = archit._get_callbacks(
                run_str_id=f"tuner_trial_run{run_idx}",
                dmhps=dmhps,
                val_kds=self.val_kds
            )
            
            # Trenuj
            history = model.fit(
                self.train_kds,
                validation_data=self.val_kds,
                epochs=self.epochs,
                callbacks=callbacks_list,
                verbose=0
            )
            
            # Predykcja na validation set
            val_predictions = model.predict(self.val_kds, verbose=0).flatten()
            
            # Oblicz RMSE
            mse = np.mean((self.val_actuals - val_predictions) ** 2)
            rmse = np.sqrt(mse)
            rmse_list.append(rmse)
            
            logger.info(f"Run {run_idx}: RMSE = {rmse:.4f}")
        
        # Średnie RMSE
        mean_rmse = np.mean(rmse_list)
        std_rmse = np.std(rmse_list)
        
        logger.info(f"Trial result: RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")
        
        return mean_rmse
    
    
    def create_tuner(self):
        """
        Tworzy obiekt Keras Tuner zgodnie z wybraną strategią.
        
        Returns:
            Keras Tuner object
        """
        if self.tuner_strategy == "bayesian":
            tuner = kt.BayesianOptimization(
                hypermodel=self.build_model,
                objective=kt.Objective('val_loss', direction='min'),
                max_trials=self.max_trials,
                executions_per_trial=self.executions_per_trial,
                directory=self.tuner_dir,
                project_name=f'bayesian_{self.timestampplus}',
                overwrite=False
            )
            
        elif self.tuner_strategy == "random":
            tuner = kt.RandomSearch(
                hypermodel=self.build_model,
                objective=kt.Objective('val_loss', direction='min'),
                max_trials=self.max_trials,
                executions_per_trial=self.executions_per_trial,
                directory=self.tuner_dir,
                project_name=f'random_{self.timestampplus}',
                overwrite=False
            )
            
        elif self.tuner_strategy == "hyperband":
            tuner = kt.Hyperband(
                hypermodel=self.build_model,
                objective=kt.Objective('val_loss', direction='min'),
                max_epochs=self.epochs,
                factor=3,
                directory=self.tuner_dir,
                project_name=f'hyperband_{self.timestampplus}',
                overwrite=False
            )
            
        else:
            raise ValueError(f"Unknown tuner strategy: {self.tuner_strategy}")
        
        return tuner
    
    
    def run_tuning(self):
        """
        Główna metoda - uruchamia proces tuningu.
        
        Returns:
            dict: Najlepsze hiperparametry
        """
        import time
        start_time = time.time()
        
        try:
            logger.info("="*70)
            logger.info("ROZPOCZYNAM HYPERPARAMETER TUNING")
            logger.info(f"Strategia: {self.tuner_strategy}")
            logger.info(f"Max trials: {self.max_trials}")
            logger.info(f"Runs per trial: {self.ilerunow_per_trial}")
            logger.info("="*70)
            
            # Stwórz tuner
            tuner = self.create_tuner()
            
            # Pokaż przestrzeń przeszukiwania
            logger.info("\nPrzestrzeń hiperparametrów:")
            tuner.search_space_summary()
            
            # Uruchom tuning
            tuner.search(
                self.train_kds,
                validation_data=self.val_kds,
                epochs=self.epochs,
                verbose=2
            )
            
            # Pobierz najlepsze wyniki
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            logger.info("\n" + "="*70)
            logger.info("NAJLEPSZE HIPERPARAMETRY:")
            logger.info("="*70)
            
            best_hps_dict = {}
            for param_name in best_hps.values.keys():
                value = best_hps.get(param_name)
                best_hps_dict[param_name] = value
                logger.info(f"{param_name}: {value}")
            
            # Zapisz najlepsze hiperparametry
            self._save_best_hyperparameters(best_hps_dict, tuner)
            
            # Wygeneruj raporty
            self._generate_reports(tuner)
            
            # Zakończenie SUCCESS
            elapsed_time = (time.time() - start_time) / 60
            logger.info(f"\nCzas tuningu: {elapsed_time:.2f} minut")
            logger.info("TUNING ZAKOŃCZONY SUKCESEM!")
            
            try:
                playsound(self.arabian_wav)
            except:
                print("(Nie można odtworzyć dźwięku sukcesu)")
            
            return best_hps_dict
            
        except Exception as e:
            # Zakończenie ERROR
            elapsed_time = (time.time() - start_time) / 60
            logger.error(f"BŁĄD podczas tuningu: {str(e)}", exc_info=True)
            logger.error(f"Czas do błędu: {elapsed_time:.2f} minut")
            
            try:
                playsound(self.game_wav)
            except:
                print("(Nie można odtworzyć dźwięku błędu)")
            
            raise
    
    
    def _save_best_hyperparameters(self, best_hps_dict, tuner):
        """Zapisuje najlepsze hiperparametry do pliku"""
        # JSON
        json_file = self.tuner_dir / f"best_hyperparameters__{self.timestampplus}.json"
        with open(json_file, 'w') as f:
            json.dump(best_hps_dict, f, indent=4)
        
        logger.info(f"\nZapisano najlepsze hiperparametry do: {json_file}")
        
        # Python code snippet (do łatwego skopiowania)
        python_file = self.tuner_dir / f"best_hps_code__{self.timestampplus}.py"
        
        with open(python_file, 'w') as f:
            f.write("# Najlepsze hiperparametry z tuningu\n")
            f.write(f"# Timestamp: {self.timestampplus}\n")
            f.write(f"# Strategy: {self.tuner_strategy}\n\n")
            f.write("from inputs.i_mdeep_hps import DeepModelHps\n\n")
            
            if self.dmtype == "bigruta":
                warstwy_str = f'["{best_hps_dict["variant_gru"]}", "{best_hps_dict["variant_ta"]}", {best_hps_dict["variant_ile_nodow"]}]'
            else:
                warstwy_str = f'["lstm", "{best_hps_dict["variant_duplicate"]}", {best_hps_dict["variant_ile_nodow"]}]'
            
            f.write(f"best_hps_tuned = DeepModelHps(\n")
            f.write(f"    cfg_id='tuned_{self.timestampplus}',\n")
            f.write(f"    initial_learning_rate={best_hps_dict['initial_learning_rate']},\n")
            f.write(f"    momentum={best_hps_dict['momentum']},\n")
            f.write(f"    dropout={best_hps_dict['dropout']},\n")
            f.write(f"    regl2={best_hps_dict['regl2']},\n")
            f.write(f"    patience_lr={best_hps_dict['patience_lr']},\n")
            f.write(f"    patience_es={best_hps_dict['patience_es']},\n")
            f.write(f"    min_delta={best_hps_dict['min_delta']},\n")
            f.write(f"    warstwy={warstwy_str}\n")
            f.write(f")\n")
        
        logger.info(f"Zapisano kod Python do: {python_file}")
        
        # Zapisz też do CSV (do śledzenia historii)
        csv_file = self.tuner_dir.parent / "tuning_history.csv"
        
        df_row = pd.DataFrame([{
            'timestamp': self.timestampplus,
            'strategy': self.tuner_strategy,
            'max_trials': self.max_trials,
            'runs_per_trial': self.ilerunow_per_trial,
            **best_hps_dict
        }])
        
        if csv_file.exists():
            df_row.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df_row.to_csv(csv_file, mode='w', header=True, index=False)
        
        logger.info(f"Dopisano do historii tuningu: {csv_file}")
    
    
    def _generate_reports(self, tuner):
        """Generuje raporty z tuningu"""
        # Pobierz wszystkie trials
        trials = tuner.oracle.get_best_trials(num_trials=min(self.max_trials, 20))
        
        # DataFrame z wynikami
        results = []
        for trial in trials:
            trial_dict = {
                'trial_id': trial.trial_id,
                'score': trial.score,
                **trial.hyperparameters.values
            }
            results.append(trial_dict)
        
        df_results = pd.DataFrame(results)
        
        # Zapisz do CSV
        csv_file = self.tuner_dir / f"trials_results__{self.timestampplus}.csv"
        df_results.to_csv(csv_file, index=False)
        logger.info(f"Zapisano wyniki wszystkich trials do: {csv_file}")
        
        # Wykres: Score vs Trial
        self._plot_score_progression(df_results)
        
        # Wykres: Importance of hyperparameters
        self._plot_hyperparameter_importance(df_results)
        
        # Summary statistics
        self._save_summary_statistics(df_results)
    
    
    def _plot_score_progression(self, df_results):
        """Wykres postępu tuningu"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(df_results['trial_id'], df_results['score'], 
                marker='o', linestyle='-', alpha=0.7)
        plt.xlabel('Trial ID')
        plt.ylabel('Validation Loss (RMSE)')
        plt.title('Tuning Progress')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        cummin = df_results['score'].cummin()
        plt.plot(df_results['trial_id'], cummin, 
                marker='o', linestyle='-', color='green', alpha=0.7)
        plt.xlabel('Trial ID')
        plt.ylabel('Best Score So Far')
        plt.title('Best Score Progression')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.tuner_dir / f"tuning_progress__{self.timestampplus}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        
        logger.info(f"Zapisano wykres postępu do: {plot_file}")
    
    
    def _plot_hyperparameter_importance(self, df_results):
        """Wykres znaczenia hiperparametrów"""
        # Wybierz tylko kolumny numeryczne (hiperparametry)
        numeric_cols = df_results.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('trial_id')
        numeric_cols.remove('score')
        
        if len(numeric_cols) == 0:
            return
        
        # Oblicz korelacje z score
        correlations = df_results[numeric_cols + ['score']].corr()['score'].drop('score').abs().sort_values(ascending=False)
        
        # Wykres
        plt.figure(figsize=(10, 6))
        correlations.plot(kind='barh', color='skyblue', edgecolor='navy')
        plt.xlabel('Absolute Correlation with Score')
        plt.title('Hyperparameter Importance (Correlation with Validation Loss)')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        plot_file = self.tuner_dir / f"hyperparameter_importance__{self.timestampplus}.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        
        logger.info(f"Zapisano wykres znaczenia hiperparametrów do: {plot_file}")
    
    
    def _save_summary_statistics(self, df_results):
        """Zapisuje statystyki podsumowujące"""
        summary_file = self.tuner_dir / f"tuning_summary__{self.timestampplus}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HYPERPARAMETER TUNING SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Timestamp: {self.timestampplus}\n")
            f.write(f"Strategy: {self.tuner_strategy}\n")
            f.write(f"Max trials: {self.max_trials}\n")
            f.write(f"Runs per trial: {self.ilerunow_per_trial}\n\n")
            
            f.write("Score Statistics:\n")
            f.write(f"  Best:   {df_results['score'].min():.4f}\n")
            f.write(f"  Worst:  {df_results['score'].max():.4f}\n")
            f.write(f"  Mean:   {df_results['score'].mean():.4f}\n")
            f.write(f"  Median: {df_results['score'].median():.4f}\n")
            f.write(f"  Std:    {df_results['score'].std():.4f}\n\n")
            
            f.write("Top 5 Trials:\n")
            top5 = df_results.nsmallest(5, 'score')
            for idx, row in top5.iterrows():
                f.write(f"  Trial {row['trial_id']}: score = {row['score']:.4f}\n")
        
        logger.info(f"Zapisano podsumowanie do: {summary_file}")


def setup_logging():
    """Konfiguracja loggera głównego"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def setup_gpu():
    """Konfiguracja GPU"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU dostępne: {len(gpus)}")
        except RuntimeError as e:
            logger.error(f"Błąd konfiguracji GPU: {e}")


# =============================================================================
# PRZYKŁAD UŻYCIA
# =============================================================================
if __name__ == "__main__":
    """
    Przykład użycia tunera
    """
    # Setup
    setup_logging()
    setup_gpu()
    
    # Import konfiguracji
    from inputs.i_columns import kols_map_gda
    
    # Konfiguracja danych
    datacfg = DataConfig(
        projekt_akronim='gda',
        plik_dane='../mojecsvy/gda/cieplo_withlags__sg.csv',
        kols_map=kols_map_gda,
        kol_target='GJ_heat',
        kol_flagA='is_weekend',
        intkols_to_scale=[]
    )
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    timestampplus = f"tuner_bigruta_{timestamp}"
    
    # Stwórz tuner
    tuner = DeepModelTuner(
        datacfg=datacfg,
        wyborkols="gda_deepheat_bezsincos",
        timestampplus=timestampplus,
        epochs=200,  # Krócej dla tuningu
        batchsize=128,
        seqlen=72,
        dmtype="bigruta",
        ilerunow_per_trial=3,  # 3 runy na trial
        max_trials=50,         # 50 prób
        tuner_strategy="bayesian",
        sprkod=False,
        testsetatlast=False
    )
    
    # Uruchom tuning
    best_hps = tuner.run_tuning()
    
    print("\n" + "="*70)
    print("TUNING ZAKOŃCZONY!")
    print("="*70)
    print("\nNajlepsze hiperparametry:")
    for key, value in best_hps.items():
        print(f"  {key}: {value}")
