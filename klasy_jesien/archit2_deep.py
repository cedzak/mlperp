#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Architecture - modele głębokie (LSTM/GRU + Attention)

## nauka: Kiedy _setup() MA SENS:
Gdy ma >10-20 linii skomplikowanej logiki
Gdy chcesz go wywołać ponownie (re-inicjalizacja)
Gdy podklasy nadpisują go własną logiką
Gdy ładujesz zasoby/pliki/łączysz z bazą
# =============================================================================
#         self._setup()
#         
#     def _setup(self):
#         self.self.sciezka_play = self.d3_kds.d1_pds.k1_path
#         self.timestampplus = self.d3_kds.timestampplus
#         self.testsetatlast = self.d3_kds.testsetatlast
# =============================================================================

## niestety ten jit nie działa:
model.compile(optimizer=optimizer, 
              loss=dmhps.loss, 
              metrics=dmhps.metrics,
              jit_compile=True  ### Włącz XLA compilation, Claude JAK WYCISNAC GPU
              )
# UserWarning: Model doesn't support jit_compile=True. Proceeding with jit_compile=False.
#   warnings.warn(
# Użytkownik dostaje warning że model nie wspiera jit_compile=True. To nie jest błąd, tylko ostrzeżenie. Prawdopodobnie dodała jit_compile=True w model.compile() jak sugerowałem wcześniej dla optymalizacji GPU.
# To ostrzeżenie pojawia się gdy:
# Model ma warstwy które nie są kompatybilne z XLA compilation (np. niektóre custom layers, RaggedTensors, niektóre operacje)
# Niektóre typy modeli functional/sequential mogą mieć problemy

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
import tensorflow as tf

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from klasy_jesien.archit0_base import BaseArchitecture
from klasy_data.d3_kds import KdsSetup
from klasy_jesien.ta_warstwy import *



class DeepArchitecture(BaseArchitecture):
    """
    Architektura modeli głębokich: GRU/LSTM z mechanizmem Attention.
    
    TU NIE MOGĘ DAĆ GOTOWYCH DATASETÓW Z KLASY k1 BO W PRZYPADKU DUAL APPROACH BĘDĄ TO INNE DATASETY !!
    A TO KLASĘ ARCHIT WKŁADAM DO KLASY APPROACH A NIE ODWROTNIE.
    """
    
    def __init__(self, kds_setup_instance: KdsSetup, 
                       ilerunow: int, 
                       dm_type: str):
        """
        Args:
            dm_type: 'bigruta' dla BiGRU+TA, 'lstm' dla zwykłego LSTM
        """
        super().__init__(
            ilerunow=ilerunow,  # Przekazujesz to, co dostałeś jako argument
            timestampplus  =kds_setup_instance.timestampplus,
            batchsize      =kds_setup_instance.batchsize,
            seqlen         =kds_setup_instance.seqlen
            )
        
        # NIE POTRZEBUJESZ: self.ilerunow = ilerunow
        # Bo klasa bazowa już to zrobiła!
        
        self.d3_kds = kds_setup_instance
        self.dm_type = dm_type
        self.output_steps = self.d3_kds.datacfg.output_steps

        self.timestampplus = self.d3_kds.timestampplus
        self.testsetatlast = self.d3_kds.testsetatlast
            
        self.epochs = self.d3_kds.epochs
 
        # input_shape: tuple (seqlen, n_features)        
        ile_kols_ma_X = self.d3_kds.d2_kluski.X_for_keras.shape[-1]
        self.input_shape = (self.seqlen, ile_kols_ma_X)
       
        # Bezpośrednie przypisania zamiast _setup()
        self.sciezka_play = self.d3_kds.d1_pds.k1_path
        self.results_path = self.d3_kds.d1_pds.k1_path.parent
        self.model_params_file = self.d3_kds.d1_pds.k1_path / f"model_params_info__{self.timestampplus}.txt"
        

        print(f"\n\njestem w klasy_jesien.archit2_deep, results_path to: {self.results_path}")


    def plot_keras_history(self, run_str_id):
        """        
           PLAN: ZRÓB ŻEBY NA OSI X ZAWSZE BYŁO NP. 200 EPOK
        """
        ile_pomijam = 0 if self.d3_kds.sprkod else 5 
        
        loss = self.history.history["loss"][ile_pomijam:]
        val_loss = self.history.history["val_loss"][ile_pomijam:]
        
        epochs_in_fig = range(ile_pomijam, len(loss)+ile_pomijam)
        
        plt.figure(figsize=(10, 7))
        plt.plot(epochs_in_fig, loss, label="Training", color="red", alpha=0.8, marker='o', markersize=3, linewidth=1.5)
        plt.plot(epochs_in_fig, val_loss, label="Validation", color="navy", alpha=0.8, marker='o', markersize=2, linewidth=1.0)
        plt.legend()
        plt.grid(True)
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    
        #*<
        plt.title(f"Learning curve without first {ile_pomijam} epochs")
        nazwa_pliku = run_str_id + f"__wofirst{ile_pomijam}e"
        path = self.sciezka_play / f"{nazwa_pliku}.png"
        plt.tight_layout(); plt.savefig(path, dpi=200)
        plt.close()
        #*>
        


        
    def build(self, dmhps):
        """
        Buduje model deep learning.

        Args:
            dmhps: Obiekt z hiperparametrami modelu
        """
        if self.dm_type == 'lstm':
            self.model = self._build_zwykly_lstm(dmhps)
        elif self.dm_type == 'bigruta_seq2seq':
            self.model = self._build_gru_attention(dmhps, output_steps=self.output_steps)
        elif self.dm_type == 'enc_dec':
            self.model = self._build_encoder_decoder(dmhps, output_steps=self.output_steps)
        else:  # 'bigruta' (default)
            self.model = self._build_gru_attention(dmhps)

        return self.model
    
    
    
    def _build_zwykly_lstm(self, dmhps):
        """Prosty LSTM - z oryginalnego define_and_compile_mdeep_zwyklyLSTM"""
        variant_ile_nodow = dmhps.warstwy[2]

        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.LSTM(
            variant_ile_nodow,
            return_sequences=False,
            kernel_regularizer=tf.keras.regularizers.l2(dmhps.regl2)
        )(inputs)

        x = tf.keras.layers.Dropout(dmhps.dropout)(x)

        outputs = tf.keras.layers.Dense(
            1, activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(dmhps.regl2)
        )(x)

        model = tf.keras.Model(inputs, outputs)

        optimizer = tf.keras.optimizers.RMSprop(
                            learning_rate=dmhps.initial_learning_rate,
                            momentum=dmhps.momentum,
                            rho=0.85
                            )

        model.compile(optimizer=optimizer, loss=dmhps.loss, metrics=dmhps.metrics)
        return model
    
    
    
    def _build_gru_attention(self, dmhps, output_steps=1):
        """GRU + Attention - z oryginalnego define_and_compile_mdeep.
        output_steps=1: seq2one; output_steps>1: seq2seq Dense (bigruta_seq2seq)
        """
        variant_gru, variant_ta, variant_ile_nodow = dmhps.warstwy
        pol_nodow = int(variant_ile_nodow / 2)

        # ***************
        inputs = tf.keras.Input(shape=self.input_shape)

        # Warstwy GRU
        if variant_gru == "1xGRU":
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    variant_ile_nodow,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(dmhps.regl2)
                )
            )(inputs)

        elif variant_gru == "2xGRU":
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    variant_ile_nodow,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(dmhps.regl2)
                )
            )(inputs)
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    pol_nodow,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(dmhps.regl2)
                )
            )(x)
        else:
            raise ValueError(f"Invalid variant_gru: {variant_gru}")

        x = tf.keras.layers.Dropout(dmhps.dropout)(x)

        # Warstwa Attention
        return_seq = False
        if variant_ta == "simple_TA":
            x = TimeAttention(return_sequences=return_seq)(x)
        elif variant_ta == "directional_TA":
            x = DirectionalTimeAttention(return_sequences=return_seq)(x)
        elif variant_ta == "long_directional_TA":
            x = LongDirectionalTimeAttention(return_sequences=return_seq)(x)
        else:
            raise ValueError(f"Invalid variant_ta: {variant_ta}")

        # ***************
        outputs = tf.keras.layers.Dense(
            output_steps,
            activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(dmhps.regl2)
        )(x)

        # ***************
        model = tf.keras.Model(inputs, outputs)

        # ***************
        optimizer = tf.keras.optimizers.RMSprop(
                            learning_rate=dmhps.initial_learning_rate,
                            momentum=dmhps.momentum,
                            rho=0.85
                            )

        # ***************
        model.compile(optimizer=optimizer, loss=dmhps.loss, metrics=dmhps.metrics)
        return model


    def _build_encoder_decoder(self, dmhps):
        """Encoder-Decoder nieautoregresyjny: Encoder BiGRU → state → Decoder GRU → Dense.
        Działa dla output_steps=1 i output_steps>1.
        """
        ile_nodow = dmhps.warstwy[2]

        # Encoder
        inputs = tf.keras.Input(shape=self.input_shape)
        _, encoder_state = tf.keras.layers.GRU(
            ile_nodow,
            return_sequences=False,
            return_state=True,
            kernel_regularizer=tf.keras.regularizers.l2(dmhps.regl2)
        )(inputs)

        # Decoder (nieautoregresyjny: state → repeat → GRU → Dense)
        x = tf.keras.layers.RepeatVector(self.output_steps)(encoder_state)
        x = tf.keras.layers.GRU(
            ile_nodow,
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(dmhps.regl2)
        )(x, initial_state=encoder_state)
        x = tf.keras.layers.Dropout(dmhps.dropout)(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation='linear')
        )(x)
        # reshape do (batch, output_steps)
        outputs = tf.keras.layers.Reshape((self.output_steps,))(x)

        model = tf.keras.Model(inputs, outputs)
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=dmhps.initial_learning_rate,
            momentum=dmhps.momentum,
            rho=0.85
        )
        model.compile(optimizer=optimizer, loss=dmhps.loss, metrics=dmhps.metrics)
        return model
    
    
    
    

    def stworz_callbacks(self, run_str_id, dmhps):
                
        return [
            tf.keras.callbacks.ModelCheckpoint(
                str(self.sciezka_play / f"model_{run_str_id}.keras"), save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", 
                                             patience=dmhps.patience_es, 
                                             min_delta=dmhps.min_delta, 
                                             restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, 
                                                 patience=dmhps.patience_lr,
                                                 min_lr=0.00001, verbose=1),
            DebugCallback(),  # <- TO JEST instancja (obiekt stworzony z klasy)
            GradientCallback()
            ]


    
    def fit(self, callbacks, train_kds, val_kds=None, verbose=2):
        """
        Trenuje model Keras.
        
        Args:
            train_kds: tf.data.Dataset z danymi treningowymi
            val_kds: tf.data.Dataset z danymi walidacyjnymi
            callbacks: Lista callbacków Keras
            verbose: Poziom szczegółowości
        """
        if self.model is None:
            raise ValueError("Model nie został zbudowany. Wywołaj build() najpierw.")

        self.history = self.model.fit(
                                train_kds, 
                                epochs=self.epochs, 
                                validation_data=val_kds, 
                                callbacks=callbacks, 
                                verbose=verbose)
            
        # pobranie info z historii treningu
        used_epochs = len(self.history.history['loss'])
        
        return self.history, used_epochs



    def wiesz_co_masz_robic(self, run_str_id, dmhps, train_kds, val_kds=None):
        
        self.clear_memory() ################################ czy to ok?
        t_train_start = time.time() # ⏱️ 
        
        self.model= self.build(dmhps)
        callbacks = self.stworz_callbacks(run_str_id, dmhps)
        self.history, used_epochs = self.fit(callbacks, train_kds, val_kds)
        
        self.plot_keras_history(run_str_id)
        
        t_train_minutes = round((time.time()-t_train_start)/60) # czas treningu (minutes) ⏱️ 
        
        return self.model, t_train_minutes, used_epochs



    def evaluate(self, kds, verbose=0):
        """
        Obliczenie podstawowych metryk dla dataset.
        
        Args:
            kds: tf.data.Dataset
            verbose: Poziom szczegółowości
        Returns:
            tuple (loss, metric_value)
        """
        loss, mae = self.model.evaluate(kds, verbose=0)
        loss, mae = round(loss, 3), round(mae, 3) 
        print(f"loss={loss}, mae={mae}")
        print("to jest liczone w mojej metodzie evaluate\n")
        
        return loss, mae


    
    def predict(self, kds):
        """
        Predykcja dla dataset val lub test.
        
        Args:
            kds: tf.data.Dataset lub numpy array
            
        Returns:
            numpy array z predykcjami (spłaszczony do 1D)
        """
        t_pred_start = time.time() # ⏱️         
        predictions = (self.model.predict(kds, verbose=1)
                       .round(2)
                       .ravel()
                       .astype("float64")
                       )
        t_pred_seconds = round(time.time()-t_pred_start)  # czas predykcji (s) ⏱️ 
        return predictions, t_pred_seconds
    

    
    def save(self, filepath):
        """Zapisuje model Keras"""
        if self.model is not None:
            self.model.save(filepath)
    
    def load(self, filepath):
        """Wczytuje model Keras"""
        self.model = tf.keras.models.load_model(filepath)
        
    def get_name(self):
        """dla deep jest to po prostu string 'deep',
        dla shallow string 'shallow'
        """
        return 'deep'


        
    def _stworz_all_runs_params_dict(self, dmhps, model_count_params=None):
        
        v_gru, v_ta, v_ile_nodow = dmhps.warstwy
        
        # Słownik z d3_kds (bez 'datacfg')
        kds_params = {k: v for k, v in vars(self.d3_kds).items() 
                      if k not in ['datacfg', 'sprkod', 'testsetatlast',
                                   "d1_pds", "d2_kluski", 
                                   "train_kdict", "val_kdict", "test_kdict"]}

        # Słownik z dmhps (bez 'params_deep_model' i 'warstwy')
        dmhps_params = {k: v for k, v in vars(dmhps).items() 
                        if k not in ['params_deep_model', 'warstwy']}

        # Złączenie wszystkiego
        return {
            **kds_params,
            'model_params_num': model_count_params,
            'dmhps_id': dmhps.cfg_id,
            'v_gru': v_gru, 'v_ta': v_ta, 'v_ile_nodow': v_ile_nodow,
            'optimizer': "RMSprop",
            **dmhps_params
            }
    

    def calc_and_save_all_runs_summary_stats(self, all_runs_str_id, 
                                              all_runs_results_dicts_list,
                                              all_runs_params_dict=None):
        """ 
        może być single, może być dual
        jak dual to możliwe że plik już istnieje, więc sprawdzam
        
        all_runs_results_dicts_list
        to jest lista slownikow __stworz_one_run_results_dict
        
        all_runs_params_dict - opcjonalny drugi słownik z parametrami
        """
        summary_stats = {'all_runs_str_id': all_runs_str_id}
        
        # Pobierz i iteruj przez wszystkie klucze Z PIERWSZEGO Z BRZEGU słownika
        all_keys_of_results_dicts = all_runs_results_dicts_list[0].keys()
        
        for key in all_keys_of_results_dicts:
            values = [r[key] for r in all_runs_results_dicts_list]
            # Sprawdź czy wartości są numeryczne
            if isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
                # Oblicz mean i std dla wszystkich kluczy numerycznych
                summary_stats[f"{key}_mean"] = round(np.mean(values), 3)
                summary_stats[f"{key}_std"] = round(np.std(values), 3)
        
        # Dodaj liczbę runów
        summary_stats['n_runs'] = len(all_runs_results_dicts_list)
        
        # Połącz: najpierw params A, potem B (jeśli jest), potem stats
        if all_runs_params_dict is not None:
            final_dict = {**summary_stats, **all_runs_params_dict}
        else:
            final_dict = {**summary_stats}
        
        df_new = pd.DataFrame([final_dict])
        
        output_path = self.sciezka_play / f"summary_stats__{self.timestampplus}.csv"
        
        # Sprawdź, czy plik istnieje
        file_exists = os.path.isfile(output_path)
        
        if file_exists: ########## TO NIE ZADZIAŁAŁO, NOWE KOLS NIE MAJĄ NAGŁÓWKÓW
            # Wczytaj istniejący plik, żeby poznać kolejność kolumn
            df_existing = pd.read_csv(output_path)
            existing_columns = df_existing.columns.tolist()
            
            # Dodaj nowe kolumny (jeśli są) na końcu
            new_columns = [col for col in df_new.columns if col not in existing_columns]
            final_columns = existing_columns + new_columns
            
            # Uporządkuj nowy wiersz zgodnie z kolejnością kolumn
            df_new = df_new.reindex(columns=final_columns)
        
        # Zapisz, dopisując jeśli trzeba
        df_new.to_csv(output_path, mode='a', index=False, header=not file_exists)






    
    
    def run_runs_elaborate( self, data_name, krotka_of_3tvtdatadicts, dmhps):
        """RUN fit_and_evaluate_mdeep RUNS
        tutaj jest robiony i val i test sets bo chcę zobaczyć czy mam dobrze zbalansowane te zbiory
        """
        lines = [
                f"\n\n==================Data {data_name}=================="
                ]
        
        for k, v in dmhps.params_deep_model.items():
            lines.append(f"  {k:<15} {v}")
        with open(self.model_params_file, "a") as file:
            file.write("\n".join(lines))
                
        
        logger.info(f"\n\n==================Data {data_name}==================")
                    
        (train_kdict, 
         val_kdict, 
         test_kdict) = krotka_of_3tvtdatadicts

        # Zamiast usuwać prefetch (zapytałam o to), dodaj więcej optymalizacji pipeline'u danych, Claude JAK WYCISNAC GPU
        train_kds = (train_kdict["kds"]
                    .cache()  # Jeśli dane mieszczą się w pamięci
                    .prefetch(tf.data.AUTOTUNE)  # ZACHOWAJ!
                    )
        val_kds   = (val_kdict["kds"]
                    .cache()  # Jeśli dane mieszczą się w pamięci
                    .prefetch(tf.data.AUTOTUNE)  # ZACHOWAJ!
                    )
        val_actuals = val_kdict["y_actuals"]
        val_indices = val_kdict["y_indices"]


        if self.d3_kds.testsetatlast:
            test_kds  = (test_kdict["kds"]
                        .cache()  # Jeśli dane mieszczą się w pamięci
                        .prefetch(tf.data.AUTOTUNE)  # ZACHOWAJ!
                        )
            test_actuals = test_kdict["y_actuals"]
            test_indices = test_kdict["y_indices"]
        
            
        
        ################## WSZYSTKO GOTOWE! MOŻNA ROBIĆ COMBO DLA KAŻDEGO RUN !        
        val_predictions_dct = {}
        test_predictions_dct = {}
        
        
        best_val_mse = float('inf')
        best_run_idx = None
        all_runs_results_dicts_list = []
        all_runs_str_id = f"{self.timestampplus}_{dmhps.cfg_id}_{data_name}"
    
    
        for run_idx in range(self.ilerunow):
            logger.info(f"\n\n=========Run {run_idx}=========")
            run_str_id = all_runs_str_id + f"__run{run_idx}"
            
            
            # Fit i  Obliczenie podstawowych metryk dla val
            # tu model nie musi być self. bo ja go nie używam dalej w kodzie
            # w ogóle to mogłaby być kreska (dla symetrii z shallow), ale niech tu jeszcze tkwi
            model, t_train_minutes, used_epochs= self.wiesz_co_masz_robic(
                                                            run_str_id, 
                                                            dmhps,
                                                            train_kds, val_kds, 
                                                            )

            val_mse, val_mae = self.evaluate(val_kds, verbose=0)
            val_predictions_dct[run_idx], t_pred_seconds = self.predict(val_kds)
            
            logger.info(f"Czas treningu dla Run {run_idx}: {t_train_minutes} minut \n"
                        f"nval_mse, val_mae = {val_mse}, {val_mae}")
           
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # obliczenia dla test set
            if self.testsetatlast:
                test_mse, test_mae = self.evaluate(test_kds, verbose=0)
                test_predictions_dct[run_idx], t_pred_seconds = self.predict(test_kds)
                # na wszelki wypadek zostawiam:
                # test_predictions = model.predict(test_kds).round(2) ## [:-1] ## 2025.10.30 SZTUCZNA CHAMSKA ZMIANA
                # logger.info("\n############# 2025.10.30 SZTUCZNA CHAMSKA ZMIANA")
                # logger.info(f"len TEST actual: {len(test_actuals)}, predicted: {len(test_predictions)}")
            else:
                test_predictions_dct[run_idx], test_mse, test_mae = None, None, None
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            one_run_results_dict = self._stworz_one_run_results_dict(
                                                    run_idx, data_name, 
                                                    t_train_minutes, t_pred_seconds,
                                                    val_mse, val_mae, 
                                                    test_mse, test_mae,
                                                    used_epochs)
            # Dodanie do listy wyników
            all_runs_results_dicts_list.append(one_run_results_dict)
            
            # Polowanie na najlepszy model
            if (len(all_runs_results_dicts_list) == 0) or (val_mse < best_val_mse):
                best_val_mse = val_mse
                # best_model = model  # obiekt Keras ## mi to nie jest potrzebne; zapisują się wszystkie - po best_run_idx wiem który model jest best
                best_run_idx = run_idx
    
        print()
        print()
        # Zapis wyników do CSV
        results_df = pd.DataFrame(all_runs_results_dicts_list)
        results_df.to_csv(self.sciezka_play / f"all_runs_results__{all_runs_str_id}.csv", index=False)
        print(f"Zapisano csv-a all_runs_results dla {data_name}")
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        all_runs_params_dict = self._stworz_all_runs_params_dict(dmhps, model.count_params())
        self.calc_and_save_all_runs_summary_stats( all_runs_str_id, 
                                          all_runs_results_dicts_list,
                                          all_runs_params_dict )
        print(f"Dopisano summary stats dla {data_name} do summary_stats__{self.timestampplus}.csv")
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ##### y_pred dla wielu runów TO JEST y_pred_best_run !!!
        val_kdict["y_pred"] = val_predictions_dct[best_run_idx]
        if self.testsetatlast:
            test_kdict["y_pred"] = test_predictions_dct[best_run_idx]
        
        krotka_of_3tvtdatadicts = (train_kdict, 
                                     val_kdict, 
                                     test_kdict) 
    
        predictions_dct_val_or_test = test_predictions_dct if self.testsetatlast else val_predictions_dct
    
        return (krotka_of_3tvtdatadicts, predictions_dct_val_or_test, best_run_idx, all_runs_params_dict)
    
    
    
    
    
    
    
            
    
    @staticmethod
    def clear_memory(): # BEZ self!
        """Funkcja do czyszczenia pamięci (Wywołaj przed treningiem)"""    
        import gc
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
    
    
    



















###### DODANE DNIA PAŃSKIEGO 2025.11.15
# To jest DEFINICJA klasy (szablon)
class DebugCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f"Batch {batch}: loss={logs.get('loss')}")      
        
class GradientCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 500 == 0:
            print(f"Batch {batch}: loss={logs.get('loss'):.4f}")
            if np.isnan(logs.get('loss')):
                print("!!! NaN detected !!!")
                self.model.stop_training = True        
        
# To jest INSTANCJA klasy (konkretny obiekt)
# DebugCallback()  # <- Nawiasy () tworzą instancję!
