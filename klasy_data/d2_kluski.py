#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kluski_ciurkiem = kluski od-batch-owane, czyli powyjmowane z batchy i w jednym wielkim batchu !!!!!!!!!!

#### zrób logger.debug zamiast print !!!!!!

WROC: ja mam zle robione indices, i jak stride i rate inne niz 1 to nie dziala;
Claude mi zrobił metodę printdebug_kds, może to coś pomoże.

WRÓĆ: wyczaruj_keras_folds, train set:
shuffle=False, #### CZEMU? RACZEJ  NIE BĘDĘ ROBIĆ I DUAL I KFOLD AND ONCE....

"""
import logging
logger = logging.getLogger(__name__)
#
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
#
from sklearn.model_selection import KFold


class KluskiConfig:
    def __init__(self, df_for_keras,
                 qty_train, qty_val, qty_test,
                 lp_kolumny_flagi,
                 epochs, batchsize, seqlen,
                 data_prep_file,
                 testsetatlast=False,
                 output_steps=1,
                 aligned_krotnosc_24h=False):
        """Konfiguracja i tworzenie datasetów z sekwencjami czasowymi.
        NIKT TEGO NIE WIE CZY TEN KOD CIĄGLE DZIAŁA BEZ ZARZUTU GDYBY sequence_stride != 1

        aligned_krotnosc_24h=True: X[i..i+k-1] → y[i..i+k-1] (te same godziny co okno).
        Okna nie nakładają się (stride=seqlen) i startują od pierwszej północy w danych.
        Wymaga danych godzinowych i seqlen będącego wielokrotnością 24.
        aligned_krotnosc_24h=False: stare zachowanie (X[i..i+k-1] → y[i+k], delay=seqlen).
        """
        self.testsetatlast = testsetatlast
        self.data_prep_file = data_prep_file
        self.output_steps = output_steps
        self.aligned_krotnosc_24h = aligned_krotnosc_24h

        self.epochs = epochs
        self.batchsize = batchsize
        self.seqlen = seqlen

        self.sampling_rate = 1
        self.df_for_keras = df_for_keras
        # zaokrąglam bo on w konwersji z PANDAS nd NUMPY traci info ze ma być zaokrąglone, i dodaje mi cyfry
        # np. 0.007 --> 0.00699999975040555000
        self.X_for_keras = np.round(df_for_keras.drop('target', axis=1).values, 3)
        y_1d = np.round(df_for_keras['target'].values, 3)

        if aligned_krotnosc_24h:
            # ============================================================
            # ALIGNED MODE: X[i..i+k-1] → y[i..i+k-1]
            # Okna nie nakładają się (stride=seqlen), startują od północy.
            # ============================================================
            assert seqlen % 24 == 0, (
                f"aligned_krotnosc_24h wymaga seqlen będącego wielokrotnością 24, podano: {seqlen}"
            )
            median_diff = pd.Series(df_for_keras.index).diff().dropna().dt.total_seconds().median()
            assert median_diff == 3600, (
                f"aligned_krotnosc_24h wymaga danych godzinowych (mediana różnic = 3600s), wykryto: {median_diff}s"
            )
            self.delay = 0
            self.sequence_stride = seqlen  # non-overlapping

            # Znajdź pierwszy wiersz z godziną 0:00 (północ)
            self.midnight_offset = 0
            if hasattr(df_for_keras.index, 'hour'):
                for i, ts in enumerate(df_for_keras.index):
                    if ts.hour == 0 and ts.minute == 0:
                        self.midnight_offset = i
                        break

            # Buduj okna od północy, bez nakładania
            n_rows = len(y_1d)
            n_windows = (n_rows - self.midnight_offset) // seqlen
            starts = [self.midnight_offset + i * seqlen for i in range(n_windows)]
            self.X_windows_aligned = np.stack(
                [self.X_for_keras[s : s + seqlen] for s in starts]
            )  # shape: (n_windows, seqlen, n_features)
            self.y_windows_aligned = np.stack(
                [y_1d[s : s + seqlen] for s in starts]
            )  # shape: (n_windows, seqlen)

            # y_for_keras zostaje 1D — używane w _add_info_to_slownik_bazowy do wyciągania actuals
            self.y_for_keras = y_1d

            # Przelicz qty z wierszy na okna (okna = całe doby)
            # Pierwsze midnight_offset wierszy przepada; reszta dzielona proporcjonalnie
            self.qty_train = (qty_train - self.midnight_offset) // seqlen
            self.qty_val   = qty_val   // seqlen
            self.qty_test  = qty_test  // seqlen

        elif output_steps > 1:
            # ============================================================
            # STARY KOD: multi-step (y dla k+1, k+2, ..., k+n)
            # ============================================================
            self.sequence_stride = 1
            self.delay = self.sampling_rate * self.seqlen
            # self.delay = self.sampling_rate * (self.seqlen +k) # gdzie k ile chcesz przeskoczyć
            # Czyli delay powinien być wielokrotnością sampling_rate (czy seqlen) żeby y lądowało na tej samej siatce czasowej co okno. Inaczej trafiasz między próbki.
            #
            # Czy delay zależy od seqlen i sampling_rate?
            # Parametrycznie — nie, delay to osobna liczba. Ale semantycznie — tak,
            # bo okno X zajmuje seqlen * sampling_rate wierszy w oryginalnych danych.
            # Żeby y zaczął się dokładnie po oknie, delay powinien być równy seqlen * sampling_rate.
            # Jeśli masz seqlen=10, sampling_rate=2, to okno zajmuje 20 wierszy, więc delay=20 żeby y był "zaraz po".
            # Multi-step: y[i] = [y_1d[i], y_1d[i+1], ..., y_1d[i+output_steps-1]]
            n = len(y_1d) - output_steps + 1
            self.y_for_keras = np.stack([y_1d[i:i+n] for i in range(output_steps)], axis=1)
            # shape: (n, output_steps); trim X do tej samej długości
            self.X_for_keras = self.X_for_keras[:n]
            # Dopasuj qty (ostatni set absorbuje przycięcie)
            trim = len(y_1d) - n
            qty_test = max(0, qty_test - trim)
            self.qty_train = qty_train
            self.qty_val   = qty_val
            self.qty_test  = qty_test

        else:
            # ============================================================
            # STARY KOD: seq2one (X[i..i+k-1] → y[i+k])
            # ============================================================
            self.sequence_stride = 1
            self.delay = self.sampling_rate * self.seqlen
            # self.delay = self.sampling_rate * (self.seqlen +k) # gdzie k ile chcesz przeskoczyć
            # Czyli delay powinien być wielokrotnością sampling_rate (czy seqlen??) żeby y lądowało na tej samej siatce czasowej co okno. Inaczej trafiasz między próbki.
            #
            # Czy delay zależy od seqlen i sampling_rate?
            # Parametrycznie — nie, delay to osobna liczba. Ale semantycznie — tak,
            # bo okno X zajmuje seqlen * sampling_rate wierszy w oryginalnych danych.
            # Żeby y zaczął się dokładnie po oknie, delay powinien być równy seqlen * sampling_rate.
            # Jeśli masz seqlen=10, sampling_rate=2, to okno zajmuje 20 wierszy, więc delay=20 żeby y był "zaraz po".
            self.y_for_keras = y_1d
            self.qty_train = qty_train
            self.qty_val   = qty_val
            self.qty_test  = qty_test

        self.lp_kolumny_flagi = lp_kolumny_flagi

        self.params_kluski = {
            k: v for k, v in vars(self).items()
            if k not in ["params_kluski", "df_for_keras", "X_for_keras", "y_for_keras",
                         "X_windows_aligned", "y_windows_aligned"]
        }




    
    def _stworz_bazowe_slowniki(self):
        
        train_kdict= {"kds": None,
                        "tvt_name": "train_kds",
                        "data_name": "kdsMono",
                        "last_ywiersz_liczonyod0": self.qty_train -1
                         }
        val_kdict  = {"kds": None,
                        "tvt_name": "val_kds",
                        "data_name": "kdsMono",
                        "last_ywiersz_liczonyod0": self.qty_train + self.qty_val -1
                         }
        test_kdict = {"kds": None,
                        "tvt_name": "test_kds",
                        "data_name": "kdsMono",
                        "last_ywiersz_liczonyod0": self.qty_train + self.qty_val + self.qty_test -1 
                        # po co to -1? MUSI BYĆ !!! inaczej jest error w Kerasie
                        ###### JUŻ WIEM, PO PROSTU JAK DODAM ILOŚCI TO MAM TOTAL ILOŚĆ, 
                        # I TO JEST LICZBA WIĘKSZA NIŻ OSTATNI IDX, 
                        # BO ILOŚĆ LICZONA JEST OD 1, A IDX (TEN DEFAULTOWY) OD ZERA !!
                        # więc dla test_kds wychodzi indeks wiekszy niz ostatni indeks w df_for_keras, 
                        # technicznie to nie przeszkadza, patrz nauka__mozna_przestrzelic_idx_i_jest_ok()
                        # ale są asymetrie, a tak to krótszy set o jeden klusek a potem wszystko łatwo prosto
                        }

        return train_kdict, val_kdict, test_kdict
    
    

    def _wyczaruj_datasets_aligned(self, train_kdict, val_kdict, test_kdict):
        """Tworzy datasety dla aligned mode z pre-built windows (from_tensor_slices).
        Okna są nienakladające się (stride=seqlen), startują od północy.
        train_kdict["last_ywiersz_liczonyod0"] to INDEKS OKNA (nie wiersza).
        """
        X_w = self.X_windows_aligned  # (n_windows, seqlen, n_features)
        y_w = self.y_windows_aligned  # (n_windows, seqlen)

        train_end = train_kdict["last_ywiersz_liczonyod0"] + 1  # exclusive window index
        val_end   = val_kdict["last_ywiersz_liczonyod0"]   + 1
        test_end  = test_kdict["last_ywiersz_liczonyod0"]  + 1

        def make_ds(X, y, batchsize):
            ds = tf.data.Dataset.from_tensor_slices((X, y))
            return ds.batch(batchsize)

        train_kds = make_ds(X_w[:train_end],          y_w[:train_end],          self.batchsize)
        val_kds   = make_ds(X_w[train_end:val_end],   y_w[train_end:val_end],   self.batchsize // 2)
        test_kds  = None
        if self.testsetatlast:
            test_kds = make_ds(X_w[val_end:test_end], y_w[val_end:test_end],    self.batchsize // 2)

        return train_kds, val_kds, test_kds


    def _wyczaruj_keras_datasets(self, train_kdict, val_kdict, test_kdict):
        """Tworzy datasety Keras dla train/val/test.
        W trybie aligned deleguje do _wyczaruj_datasets_aligned.

        każdy kds to tuple (X,y), gdzie X to "kluski" (sekwencje), a y to floaty
        X_for_keras[0:5]   → klusek 0  → target y_for_keras[5]
        X_for_keras[1:6]   → klusek 1  → target y_for_keras[6]
        X_for_keras[2:7]   → klusek 2  → target y_for_keras[7]
        ...
        train_kds to tf.data.Dataset zawierający:
        - batche po (batchsize) klusków
        - każdy klusek ma shape: (seqlen, n_features)

        TU JEST TERAZ WSZYSTKO DOBRZE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        JAK CHCESZ ZOBACZYĆ JAK TO DZIAŁA TO URUCHOM kluski_zrozum.py

        !!!!!!! "end_index" TUTAJ DLA TEST_SET
        !!!!!!!  to wcale nie jest end_index po pythonowemu tylko ten ostatni co się jeszcze łapie !!!!!!!
        PATRZ nauka__idx_defaultowy_liczony_jest_od_zera
        
        jak piszę "df_for_keras[0:18]" to TO NIE SĄ INDEKSY tej konkretnej df tylko takie defaulotwe, 
        czyli że TO SĄ LICZBY PORZĄDKOWE ALE LICZONE OD ZERA !!!!!!!!!!!!!!
        TYMCZASEM QTY TO SĄ ILOŚCI czyli LICZBY PORZĄDKOWE LICZONE OD 1 !!!!!!!!!!!!!!
        !!! taki defaultowy idx to liczba porządkowa liczona od zera !!!
        """
        # self.nauka__idx_defaultowy_liczony_jest_od_zera()

        if self.aligned_krotnosc_24h:
            return self._wyczaruj_datasets_aligned(train_kdict, val_kdict, test_kdict)

        train_kds = tf.keras.utils.timeseries_dataset_from_array(
            self.X_for_keras, # [:-self.delay], # 2025.10.28 to było źle, kończył choć mogł zrobić jeszcze kilka klusków
            targets=self.y_for_keras[self.delay:],
            sequence_length=self.seqlen, 
            sequence_stride=self.sequence_stride, 
            sampling_rate=self.sampling_rate, 
            batch_size=self.batchsize,
            shuffle=False,
            start_index=  0,
            end_index=    train_kdict["last_ywiersz_liczonyod0"]
        )
        ################# jeśli wierzyć Claudowi, to:
            # 2. Keras używa start/end_index tylko na X !!!!!!!!!!!!!
            # 3. Keras bierze targety "na ślepo" z tego samego indeksu
            # 4. Może wziąć target spoza zakresu! 

        val_kds = tf.keras.utils.timeseries_dataset_from_array(
            self.X_for_keras, # [:-self.delay], # 2025.10.28 to było źle, kończył choć mogł zrobić jeszcze kilka klusków
            targets=self.y_for_keras[self.delay:],
            sequence_length=self.seqlen,
            sequence_stride=self.sequence_stride,
            sampling_rate=self.sampling_rate,
            batch_size=self.batchsize //2, # Dodaj różne batch size dla train i validation: validation_data=val_ds.batch(batch_size // 2),  # Mniejszy batch dla walidacji
            shuffle=False,
            start_index= train_kdict["last_ywiersz_liczonyod0"],
            end_index=   val_kdict["last_ywiersz_liczonyod0"]
        )

        test_kds = None
        if self.testsetatlast==True:
            test_kds = tf.keras.utils.timeseries_dataset_from_array(
                self.X_for_keras, # [:-self.delay], # 2025.10.28 to było źle, kończył choć mogł zrobić jeszcze kilka klusków
                targets=self.y_for_keras[self.delay:],
                sequence_length=self.seqlen,
                sequence_stride=self.sequence_stride,
                sampling_rate=self.sampling_rate,
                batch_size=self.batchsize //2, # Dodaj różne batch size dla train i validation: validation_data=val_ds.batch(batch_size // 2),  # Mniejszy batch dla walidacji
                shuffle=False,
                start_index=val_kdict["last_ywiersz_liczonyod0"],
                end_index=  test_kdict["last_ywiersz_liczonyod0"]
            )

        return train_kds, val_kds, test_kds


    def _add_info_to_slownik_bazowy(self, kds_dict):

        kds = kds_dict["kds"]
        tvt_name = kds_dict["tvt_name"]
        ilosc_batchy = len(list(kds))
        
        X_arraykluskow, y_arrayfloatow = self.batches_to_kluski(kds, tvt_name) ## tu mam assert lenX==leny
        
        ilosc_kluskow = len(y_arrayfloatow) ######### -1 if tvt_name=="test_kds" else len(y_arrayfloatow) #########
        
        last_ywiersz_liczonyod0 = kds_dict["last_ywiersz_liczonyod0"] 
        
# =============================================================================
#         last_ywiersz_liczonyod0 = ( 
#             kds_dict["last_ywiersz_liczonyod0"] -1 if tvt_name=="test_kds" 
#             else kds_dict["last_ywiersz_liczonyod0"]) ############# 2025.10.30 SZTUCZNA CHAMSKA ZMIANA
# =============================================================================

        first_ywiersz_liczonyod0 = last_ywiersz_liczonyod0 -ilosc_kluskow +1 # słupki w płocie

        if self.aligned_krotnosc_24h:
            # first/last są INDEKSAMI OKIEN; każde okno zajmuje seqlen wierszy.
            # Wiersz startowy okna w_idx = midnight_offset + w_idx * seqlen
            row_start = self.midnight_offset + first_ywiersz_liczonyod0 * self.seqlen
            row_end   = self.midnight_offset + (last_ywiersz_liczonyod0 + 1) * self.seqlen
            y_actuals = self.y_for_keras[row_start : row_end]          # 1D, n_windows * seqlen
            y_indices = self.df_for_keras.index[row_start : row_end]   # pełne timestampy
        else:
            y_actuals = self.y_for_keras[ first_ywiersz_liczonyod0 : last_ywiersz_liczonyod0 +1 ]
            y_indices = self.df_for_keras.index[ first_ywiersz_liczonyod0 : last_ywiersz_liczonyod0 +1 ]

        with open(self.data_prep_file, "a") as file:
            file.write(
                f"\n\n{'=' *20}\n"
                f"{tvt_name}\n"
                f"{'-' *20}\n"
                f"len df_for_keras: {len(self.df_for_keras)}\n"
                f"ilosc_batchy: {ilosc_batchy} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                f"len(y_arrayfloatow): {len(y_arrayfloatow)}\n"
                f"{'-' *20}"
                #
                f"ilosc_kluskow: {ilosc_kluskow}\n"
                f"first_ywiersz_liczonyod0: {first_ywiersz_liczonyod0}\n"
                f"last_ywiersz_liczonyod0: {last_ywiersz_liczonyod0}\n"
                f"y_actuals: {y_actuals[:10]}\n"
                f"y_indices: {y_indices[:10]}\n"
                f"len y_actuals: {len(y_actuals)}\n"
                f"len y_indices: {len(y_indices)}"
                )

      
        ## TO ZAJMUJE STRASZNIE DUŻO LINIJEK !!
# =============================================================================
#         for i, krotka in enumerate(list(kds)):
#             print(f"\n~~~krotki X,y in batch {i} tak jak je Keras stworzył:\n{krotka}\n")
# =============================================================================


        # zapis do slownikow
        kds_dict["ilosc_batchy"] = ilosc_batchy
        kds_dict["ilosc_kluskow"] = ilosc_kluskow
        
        kds_dict["X_arraykluskow"] = X_arraykluskow
        kds_dict["y_arrayfloatow"] = y_arrayfloatow
        
        kds_dict["y_indices"] = y_indices
        kds_dict["y_actuals"] = y_actuals


        return kds_dict



    def stworz_uzupelnione_slowniki(self):
        
        print('ZACZYNA ROBIC SLOWNIKI')
        
        (train_kdict, 
         val_kdict, 
         test_kdict) = self._stworz_bazowe_slowniki()
        print('ZROBIL BAZOWE SLOWNIKI')
        
        (train_kdict["kds"], 
         val_kdict["kds"], 
         test_kdict["kds"]) = self._wyczaruj_keras_datasets(train_kdict, 
                                                              val_kdict, 
                                                              test_kdict)
        print('WYCZAROWAL KDSY')
                

        with open(self.data_prep_file, "a") as file:
            file.write(
                f"\n\n\n{'=' *80}\n"
                f"Kds - Informacje podstawowe\n"
                f"  Epochs: {self.epochs}, Batch size: {self.batchsize}, Seq len: {self.seqlen}\n"
                f"  Qty train: {self.qty_train}, val: {self.qty_val}, test: {self.qty_test}\n"
                f"  Shape X_for_keras: {self.X_for_keras.shape}\n"
                )
                          
        for kds_dict in [train_kdict, val_kdict]:
            kds_dict = self._add_info_to_slownik_bazowy(kds_dict)  
        if self.testsetatlast:
            test_kdict = self._add_info_to_slownik_bazowy(test_kdict) 

        print('ZROBIL UZUPELNIONE SLOWNIKI')


        with open(self.data_prep_file, "a") as file:
            file.write(
                f"{'=' *80}\n"
                f"~~~~~~~~~~~~~~Zakonczyl stworz_uzupelnione_slowniki~~~~~~~~~~~~~~\n\n"
                )

        print("\nkds dict keys:")
        for key in kds_dict.keys():
            print(key)                               

        # kds
        # name
        # last_ywiersz_liczonyod0
        # ilosc_batchy
        # ilosc_kluskow
        # X_arraykluskow
        # y_arrayfloatow
        # y_actuals
        # y_indices
                               
        return (train_kdict, 
                val_kdict, 
                test_kdict)                                                                   




    

    def k():
        """TU DALEJ SĄ METODY DLA KTÓRYCH ARGUMENTEM JEST JEDEN kds I JEGO name"""
        pass
    
    def ekstrahuj_y_actual(self, kds, tvt_name):
        # y_targets # numpy.ndarray (1D)
        y_arrayfloatow = np.concatenate([y_batch.numpy() for _, y_batch in kds], axis=0).astype("float16")
                                                # ^^^^^^^^^^^^^^ skleja listę arrays → jeden wielki array
        return y_arrayfloatow

    def batches_to_kluski(self, kds, tvt_name):
        """Konwertuje tf.data.Dataset do numpy arrays."""
        
        # v długa:
        # X_batches = [] # <- lista numpy arrays
        # y_batches = []
        # for X_batch, y_batch in kds:
        #     X_batches.append(X_batch.numpy()) => # X_batches = [array1, array2, array3]  <- lista arrayów
        #     y_batches.append(y_batch.numpy())
        # X_arraykluskow = np.concatenate(X_batches, axis=0)
        # y_arrayfloatow = np.concatenate(y_batches, axis=0)
        
        
        # v kompaktowa:
        # X_sequences # numpy.ndarray (3D)
        X_arraykluskow = np.concatenate([X_batch.numpy() for X_batch, _ in kds], axis=0).astype("float16")
        # y_targets # numpy.ndarray (1D)
        y_arrayfloatow = self.ekstrahuj_y_actual(kds, tvt_name)
        # zamiast: 
# =============================================================================
#         y_arrayfloatow = np.concatenate([y_batch.numpy() for _, y_batch in kds], axis=0).astype("float16")
#                                                 # ^^^^^^^^^^^^^^ skleja listę arrays → jeden wielki array
# =============================================================================

        assert len(X_arraykluskow)==len(y_arrayfloatow)

# =============================================================================
#         self.printdebug_kluskiciurkiem(kds, tvt_name, 
#                                   X_arraykluskow, 
#                                   y_arrayfloatow)
# =============================================================================

        return X_arraykluskow, y_arrayfloatow



    def _stworz_maskA_dla_kluskow__OSTATNI(self, flagsA_dla_kluskow):
        # 2025.11.16 out bo pięknie prognozuje A i do bani B
        # KLUCZOWA LOGIKA: Bierz TYLKO OSTATNI timestep (t-1), nie .any()!
        maskA_OSTATNI = (flagsA_dla_kluskow[:, -1] == 1)  # shape: (n_sequences,)
        return maskA_OSTATNI

    def _stworz_maskA_dla_kluskow__50PROC(self, flagsA_dla_kluskow):
        # 2025.11.16 nowa maska 50% -- TUTAJ ZOSTAWIAM WERSJĘ Z WYJAŚNIENIAMI
        # KLUCZOWA LOGIKA: Maska=1 gdy przynajmniej połowa flag w sekwencji == 1
        # Zliczamy ile timestepów w każdej sekwencji ma flagę równą 1
        suma_flag = np.sum(flagsA_dla_kluskow == 1, axis=1)  # shape: (n_sequences,)
        # => Dla każdego kluska dostajemy jedną liczbę: ile było jedynek w całej sekwencji
        # Obliczamy próg: połowa długości sekwencji
        polowa_seqlen = flagsA_dla_kluskow.shape[1] / 2
        # => Przykład: jeśli seqlen=10, to polowa_seqlen=5.0
        # Tworzymy maskę boolowską: True gdy sekwencja ma >= 50% jedynek, False w przeciwnym razie
        maskA_50PROC = (suma_flag >= polowa_seqlen)  # shape: (n_sequences,)
        # => Ta maska decyduje, które kluski trafiają do modelu A (True) a które do B (False)
        return maskA_50PROC
        
    def _stworz_maskA_dla_kluskow__ANY(self, flagsA_dla_kluskow):
        # GDY .any(axis=1) TO JEST STABILNIEJ NIZ GDY .all(axis=1), 
        maskA_ANY = (flagsA_dla_kluskow == 1).any(axis=1)  # shape: (n_sequences,)
        return maskA_ANY
                
    def _stworz_maskA_dla_kluskow__ALL(self, flagsA_dla_kluskow):
        # GDY .any(axis=1) TO JEST STABILNIEJ NIZ GDY .all(axis=1), 
        maskA_ALL = (flagsA_dla_kluskow == 1).all(axis=1)  # shape: (n_sequences,)    
        return maskA_ALL







    
    
    def _get_maskA_dla_kluskow_for_kds(self, kds, tvt_name):
        """
        Zwraca maskę boolean określającą które sekwencje należą do regimeA.
        
        Określa regime na podstawie flagi w OSTATNIM timestepie sekwencji (t-1).
        LOGIKA:
        - Sekwencja X: [t-seqlen, ..., t-1]  (wejście do modelu)
        - Target y: wartość w momencie t (to co przewidujemy)
        - maskA_dla_kluskow[i] = True jeśli flaga[t-1] == 1 dla sekwencji i
        DLACZEGO t-1 a nie .any()?
        - .any() sprawdzał "czy KIEDYKOLWIEK w oknie była flaga"
        - To powodowało że model przy rozruchu ciągle przewidywał spadek temperatur
        - Sprawdzając tylko t-1 mówimy modelowi: "W momencie najbliższym targetowi jest/nie ma flagi"
        
        Args:
            kds: tf.data.Dataset z sekwencjami
            tvt_name
        Returns:
            numpy array: maska boolean (n_sequences,): True = regimeA, False = regimeB
        """
        # Wyciągnij kluski z kds
        X_arraykluskow, y_arrayfloatow = self.batches_to_kluski(kds, tvt_name)
        
        # Wyciągnij flagę reżimu A dla wszystkich timestepów dla każdej sekwencji
        flagsA_dla_kluskow = X_arraykluskow[:, :, self.lp_kolumny_flagi]  # shape: (n_sequences, seqlen)
        # Teraz mamy tablicę flag dla każdej sekwencji - każdy wiersz to jeden "klusek", każda kolumna to timestep
        









        maskA_dla_kluskow = self._stworz_maskA_dla_kluskow__OSTATNI(flagsA_dla_kluskow)
        # maskA_dla_kluskow = self._stworz_maskA_dla_kluskow__50PROC(flagsA_dla_kluskow)
        # maskA_dla_kluskow = self._stworz_maskA_dla_kluskow__ANY(flagsA_dla_kluskow)
        # maskA_dla_kluskow = self._stworz_maskA_dla_kluskow__ALL(flagsA_dla_kluskow)








        n_rezimA = np.sum(maskA_dla_kluskow)
        n_rezimB = np.sum(~maskA_dla_kluskow)
        n_total = len(maskA_dla_kluskow)



        with open(self.data_prep_file, "a") as file:
            file.write(
                "\n\n"
                f"{'---*' *15}"
                f"lp_kolumny_flagi: {self.lp_kolumny_flagi}\n"
                f'flagsA_dla_kluskow[:5] = \n{flagsA_dla_kluskow[:5]}\n\n'
                f'maskA_dla_kluskow[:5] = \n{maskA_dla_kluskow[:5]}\n\n'
                f"Kluski po rozdzieleniu na A i B dla {tvt_name}: A={n_rezimA}, B={n_rezimB}, total={n_total}.\n"
                f"Tak więc jest {np.sum(maskA_dla_kluskow)} klusków A z wszystkich {len(maskA_dla_kluskow)} klusków.\n"
                f"{'---*' *15}"
                "\n\n"
                )
        
        #### przykład:
            
        # flagsA_dla_kluskow = 
        # [[0. 0. 1.]
        #  [0. 1. 1.]
        #  [1. 1. 1.]
        #  [1. 1. 0.]
        #  [1. 0. 0.]
        #  [0. 0. 1.]
        #  [0. 1. 1.]]

        # maskA_dla_kluskow = 
        # [ True  True  True False False  True  True]

        # Kluski po rozdzieleniu na A i B dla train_kds: A=5, B=2, total=7.
        # Tak więc jest 5 klusków A z wszystkich 7 klusków.
        
        return X_arraykluskow, y_arrayfloatow, maskA_dla_kluskow




    def rozdziel_kluski_na_rezimy_A_i_B(self, kds, tvt_name, 
                                        y_indices, y_actuals,
                                        shuffle=False):
        """
        Dzieli dataset na regimeA i regimeB używając maski boolean.
        Returns:
            tuple: (kds_A, kds_B, maskA_dla_kluskow, y_indicesA, y_indices_B)
        """
        (X_arraykluskow, 
         y_arrayfloatow,
         maskA_dla_kluskow) = self._get_maskA_dla_kluskow_for_kds(kds, tvt_name)

# =============================================================================
#         if tvt_name=="test_kds":
#             print("\n\nuwaga, test set! ############# 2025.10.31 SZTUCZNA CHAMSKA ZMIANA\n")
#             maskA_dla_kluskow = maskA_dla_kluskow[:-1] ############# 2025.10.31 SZTUCZNA CHAMSKA ZMIANA
#             X_arraykluskow = X_arraykluskow[:-1]
#             y_arrayfloatow = y_arrayfloatow[:-1]
#         print(f"len maskA_dla_kluskow: {len(maskA_dla_kluskow)}")
# =============================================================================
        
        # Podział używając maski
        X_A = X_arraykluskow[maskA_dla_kluskow]
        y_A = y_arrayfloatow[maskA_dla_kluskow]
        y_indicesA = y_indices[maskA_dla_kluskow]
        y_actualsA = y_actuals[maskA_dla_kluskow]
        
        X_B = X_arraykluskow[~maskA_dla_kluskow]
        y_B = y_arrayfloatow[~maskA_dla_kluskow]
        y_indicesB = y_indices[~maskA_dla_kluskow]
        y_actualsB = y_actuals[~maskA_dla_kluskow]
        
        
        with open(self.data_prep_file, "a") as file:
            file.write(
                f"\n\n\ntvt_name: {tvt_name}\n"
                f"len y_indices: {len(y_indices)}\n"
                f"len y_actuals: {len(y_actuals)}\n"
                f"len maskA_dla_kluskow: {len(maskA_dla_kluskow)}\n"
                f"{tvt_name}: regimeA={len(X_A)}, regimeB={len(X_B)}\n\n"
                )
        
        
        # Tworzenie datasetów
        def kluski_back_to_batches(X_arraykluskow, y_arrayfloatow):
            if len(X_arraykluskow) == 0:
                return None
            ds = tf.data.Dataset.from_tensor_slices((X_arraykluskow, y_arrayfloatow))
            if shuffle:
                ds = ds.shuffle(buffer_size=len(X_arraykluskow))
            return ds.batch(self.batchsize)
        
        kdsA = kluski_back_to_batches(X_A, y_A)
        kdsB = kluski_back_to_batches(X_B, y_B)


        kdictA = {  "kds": kdsA,
                    "tvt_name" : tvt_name,
                    "data_name": "kdsA",
                    "y_indices": y_indicesA,
                    "y_actuals": y_actualsA,
                    }
        
        kdictB = {  "kds": kdsB,
                    "tvt_name" : tvt_name,
                    "data_name": "kdsB",
                    "y_indices": y_indicesB,
                    "y_actuals": y_actualsB,

                    }
               
        return (kdictA, kdictB,
                maskA_dla_kluskow)
        







    def d():
        pass
        
    
    def printdebug_kds(self, kds, tvt_name, indices=None, n_batches=2):
        """Debuguje pierwsze n_batches z datasetu.
        CLAUDE MI TO ZROBIŁ !!!"""
        
        #### zrób logger.debug zamiast print
        
        print(f"\n{'='*60}")
        print(f"DEBUG: {tvt_name}")
        print(f"{'='*60}")

        print("list(kds):")
        print(list(kds))
        print(f"Batch size: {self.batchsize}")

        try:
            print(f"Total number of batches in {tvt_name} (czyli len(list(kds)): {len(list(kds))}\n")
        except:
            print("Cannot determine total batches\n")
        
        if indices is not None:
            print(f"Total indices: {len(indices)}\n")
        
        # for i, (X_batch, y_batch) in enumerate(kds_list_czyli_lista_tupli[-2:], start=-2):
        # NAUKA !!! to w poprzedniej linii to by było prawie to samo co w linii poniżej, różnica jest taka że w przypadku listy biorę sobie ostatnie a nie pierwsze dwa (n) batches
        for i, (X_batch, y_batch) in enumerate(kds.take(n_batches)):
            print(f"--- Batch {i} ---")
            print(f"X: shape={X_batch.shape}, "
                        f"range=[{tf.reduce_min(X_batch):.3f}, {tf.reduce_max(X_batch):.3f}]\n")
            print(f"y: shape={y_batch.shape}, "
                        f"range=[{tf.reduce_min(y_batch):.3f}, {tf.reduce_max(y_batch):.3f}]")

            # X_batch shape: (batchsize, seqlen, n_features)
            #                (20,         5,       4) 
            # y_batch shape: (batchsize,)
            #                (20,)
            
            
            # to narazie zostawiam, bo to muszą być floaty bo inaczej nie robi
# =============================================================================
#             # Statystyki 
#             print(f"X: mean={tf.reduce_mean(X_batch):.3f}, "
#                         f"std={tf.math.reduce_std(X_batch):.3f}")
#             print(f"y: mean={tf.reduce_mean(y_batch):.3f}, "
#                         f"std={tf.math.reduce_std(y_batch):.3f}")
#             
#             # Sprawdź problemy
#             has_nan = tf.reduce_any(tf.math.is_nan(X_batch)) or tf.reduce_any(tf.math.is_nan(y_batch))
#             has_inf = tf.reduce_any(tf.math.is_inf(X_batch)) or tf.reduce_any(tf.math.is_inf(y_batch))
#             
#             if has_nan:
#                 logger.error("❌ NaN detected!")
#             if has_inf:
#                 logger.error("❌ Inf detected!")
# =============================================================================
            
            # Indices #### WRÓĆ
            if indices is not None:
                start_idx = i * self.batchsize
                end_idx = min((i + 1) * self.batchsize, len(indices))
                batch_indices = indices[start_idx:end_idx]
                print(f"Batch indices: {batch_indices} (n={len(batch_indices)})")
        
        print(f"\n{'='*60}\n")


    def printdebug_kluskiciurkiem(self, kds, tvt_name, 
                             X_arraykluskow, 
                             y_arrayfloatow, indices=None):
        """
        """
        #### nauka: 2 sposoby konwersji Keras Dataset do pamięci:
        kds_list_czyli_lista_tupli = list(kds)  # lista tupli (x_batch, y_batch) # prawie to samo co zwykły kds, ale ładuje wyszstko do pamięci, a kds to generator
        # - Tworzy listę batchy (x_batch, y_batch)
        # - Ładuje wszystko do pamięci
        # - Może mieć problemy z niektórymi typami datasetów
        kds_list_iterator = list(kds.as_numpy_iterator())
        # - Używa dedykowanego iteratora numpy
        # - Bardziej stabilne dla różnych typów tensorów
        # - Również ładuje wszystko do pamięci
        # - Zwraca tuple numpy arrays (X, y) dla każdego batcha

        
        #### zrób logger.debug zamiast print
        print("\n" *2 + "==^" *15)
        print(f"printdebug_kluskiciurkiem dla {tvt_name}:\n")
        #
        print("TYPY DANYCH:")
        print(f"kds list: {type(kds_list_czyli_lista_tupli)}") # lista tupli (x_batch, y_batch)
        print(f"kds list ITERATOR: {type(kds_list_iterator)}") # lista tupli (x_batch, y_batch)    
        #                                         
        print(f"X kluski ciurkiem: {type(X_arraykluskow)}")  # numpy.ndarray
        print(f"y kluski ciurkiem: {type(y_arrayfloatow)}\n")  # numpy.ndarray
        #
        print("TYPY DANYCH TAM W ŚRODKU -- Poziom 1 - pierwszy batch albo klusek albo float:")
        print(f"kds list [0]: {type(kds_list_czyli_lista_tupli[0])}") # 'tuple' (x_batch, y_batch) 
        print(f"X[0]: {type(X_arraykluskow[0])}")  # 'numpy.ndarray'
        print(f"X[0] shape: {X_arraykluskow[0].shape}") # X[0] shape: (2, 3)
        print(f"y[0]: {type(y_arrayfloatow[0])}")  # 'numpy.float64'
        print(f"y[0] shape: {y_arrayfloatow[0].shape}\n") # y[0] shape: ()
        #
        print("TYPY DANYCH TAM W ŚRODKU -- Poziom 2:")
        print(f"kds list [0][0]: {type(kds_list_czyli_lista_tupli[0][0])}") # 'tuple' (x_batch, y_batch) 
        print(f"X[0][0]: {type(X_arraykluskow[0][0])}")  # 'numpy.ndarray'
        print(f"X[0][0] shape: {X_arraykluskow[0][0].shape}\n") # X[0][0] shape: (2, 3)
        #
        print("LENGTHS:")
        print(f"len(kds_list_czyli_lista_tupli) czyli total ilość batchy = {len(kds_list_czyli_lista_tupli)}")   
        print(f"len(X_arraykluskow) czyli ilość klusków = {len(X_arraykluskow)}")
        print(f"len(y_arrayfloatow) czyli ilość klusków = {len(y_arrayfloatow)}\n")
        #
        print("PIERWSZE 30 ELEMENTY:")
        # print(f"~~~~~~~~~~~~~~kds_list_czyli_lista_tupli[:30] = \n{kds_list_czyli_lista_tupli[:30]}\n\n")
        for i, krotka in enumerate(kds_list_czyli_lista_tupli[:30]):
            print(f"~~~ {i}")
            print(krotka)
        print()
        print(f"~~~~~~~~~~~~~~X_arraykluskow[:30] = \n{X_arraykluskow[:30]}\n")
        print(f"~~~~~~~~~~~~~~y_arrayfloatow[:30] = \n{y_arrayfloatow[:30]}")
        #
        print("==^" *15 +"\n" *2)
        
        # claude KIEDYŚTAM
# =============================================================================
#         print("=" *80)
#         print("printdebug_kluskiciurkiem KIEDYŚTAM:")
#         print(f"  testsetatlast: {self.testsetatlast}")
#         print(f"  Pierwsze 15 wartości y: {y_arrayfloatow[:15]}")
#         print(f"  Pierwsze 3 indeksy: {indices[:3]}")
#         print(f"  Ostatnie 3 indeksy: {indices[-3:]}")
#         print(f"  Długość keras set: {len(y_arrayfloatow)}")
#         print(f"  Długość indices: {len(indices)}")
#         # ja:
#         print("***" *15)
#         print(f"{self.testsetatlast}")
#         print(f"{'First 15 values of y keras set:':<26} {y_arrayfloatow[:15]}")
#         # print(f"{'First 3 idx of keras set:':<26} {y_arrayfloatow[:3]}") # to 3 pierwsze igreki...
#         print(f"{'First 3 idx of indices:':<26} {indices[:3]}")
#         # print(f"{'Last 3 idx of keras set:':<26} {y_arrayfloatow[-3:]}") # to 3 ostatnie igreki...
#         print(f"{'Last 3 idx of indices:':<26} {indices[-3:]}")
#         print(f"{'Len of keras set:':<26} {len(y_arrayfloatow)}")
#         print(f"{'Len of indices:':<26} {len(indices)}")
# =============================================================================

        return y_arrayfloatow
    
    

    def nauka__idx_defaultowy_liczony_jest_od_zera(self):
        """tu widać, że gdbyby end_index był wzięty przez Kerasa tak jak go podaję, 
        to byłoby źle, bo zabrakłoby ostatniego wiersza z danymi"""

        X_for_keras = np.array([
            [1.0, 0.1, 0],   # wiersz 0 - shutdown=0
            [2.0, 0.2, 0],   # wiersz 1 - shutdown=0
            [3.0, 0.3, 1],   # wiersz 2 - shutdown=1 (start shutdownu)
            [4.0, 0.4, 1],   # wiersz 3 - shutdown=1
            [5.0, 0.5, 1],   # wiersz 4 - shutdown=1
            [6.0, 0.6, 0],   # wiersz 5 - shutdown=0 (koniec shutdownu)
            [7.0, 0.7, 0],   # wiersz 6 - shutdown=0
            [8.0, 0.8, 1],   # wiersz 7 - shutdown=1 (kolejny shutdown)
            [9.0, 0.9, 1],   # wiersz 8 - shutdown=1
            [1.9, 9.1, 1],   # wiersz 9  - shutdown=1
            [2.8, 8.2, 0],   # wiersz 10 - shutdown=0
            [3.7, 7.3, 0],   # wiersz 11 - shutdown=0
            [4.6, 6.4, 0],   # wiersz 12 - shutdown=0
            [5.5, 5.5, 1],   # wiersz 13 - shutdown=1
            [6.4, 4.6, 1],   # wiersz 14 - shutdown=1
            [7.3, 3.7, 0],   # wiersz 15 - shutdown=0
            [8.2, 2.8, 0],   # wiersz 16 - shutdown=0
            [9.1, 1.9, 0],   # wiersz 17 - shutdown=0
        ])
        y_for_keras = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90,  # train (0-8)
                                19, 28, 37, 46, 55, 64, 73, 82, 91]) # val (9-16)
        df_for_keras = pd.DataFrame({
            't_blade': X_for_keras[:, 0],
            'shut_minutes': X_for_keras[:, 1],
            'shutdown': X_for_keras[:, 2].astype(int),  # boolean kolumna
            'target': y_for_keras
        }, index=pd.date_range('2023-01-01', periods=18, freq='h'))
        
        print("\ndf_for_keras[0:18]")
        print(df_for_keras[0:18]) 
        ########## TO NIE SĄ INDEKSY, TO SĄ LICZBY PORZĄDKOWE !!!
        # ALE LICZONE OD ZERA !!!!!!!!!!!!!!
        
        print("\ndf_for_keras[15:17]") #
        print(df_for_keras[15:17])
        
        print("\ndf_for_keras[15:18]") #
        print(df_for_keras[15:18])
        
        print("\ndf_for_keras[15:19]") #
        print(df_for_keras[15:19])
        print("DZIWNE ŻE SIĘ NIE WYWALA... --> nie dziwne, patrz nauka__mozna_przestrzelic_idx_i_jest_ok")


    def nauka__mozna_przestrzelic_idx_i_jest_ok():
        
        df = pd.DataFrame({'A': [1, 2, 3]})

        # BEZPIECZNE - nie rzucą błędów:
        print(df[:100])        # ✓
        print(df[5:10])        # ✓ (pusty DataFrame)
        print(df[-10:2])       # ✓
        
        # NIEBEZPIECZNE - mogą rzucić błąd:
        print(df.iloc[5])      # IndexError
        print(df.loc[999])     # KeyError









    def cv():
        pass
    def wyczaruj_keras_folds(self, n_splits=4):
        """Tworzy datasety dla K-Fold cross-validation."""
        lenX_minusDELAY_minus1 = len(self.X_for_keras) - self.delay
        
        logger.debug("=" *80)
        logger.debug("wyczaruj_keras_folds")
        logger.debug(f"lenX_minusDELAY_minus1={lenX_minusDELAY_minus1}, delay={self.delay}")
        
        # Define train+val and test boundaries
        train_val_qty = min(self.qty_train + self.qty_val, lenX_minusDELAY_minus1)
        test_end = lenX_minusDELAY_minus1 if self.qty_test > 0 else None
        
        # Get indices for train+val and test
        train_val_indices = np.arange(self.seqlen, train_val_qty + 1, self.sequence_stride)
        test_indices = (
            np.arange(self.qty_train + self.qty_val + self.seqlen, test_end, self.sequence_stride)
            if self.qty_test > 0 else None
        )

        # Initialize K-Fold
        kf = KFold(n_splits=n_splits, shuffle=False)
        fold_datasets = []
        fold_indices = []
        
        for fold_idx, (fold_train_fold_indices, 
                       fold_val_fold_indices) in enumerate(kf.split(train_val_indices)):
            fold_train_data_indices = train_val_indices[fold_train_fold_indices]
            fold_val_data_indices = train_val_indices[fold_val_fold_indices]
            
            logger.debug(f"Fold {fold_idx}:")
            logger.debug(f"  Train indices: {fold_train_data_indices[:5]}...{fold_train_data_indices[-5:]}")
            logger.debug(f"  Val indices: {fold_val_data_indices[:5]}...{fold_val_data_indices[-5:]}")

            train_kds = tf.keras.utils.timeseries_dataset_from_array(
                data=self.X_for_keras[:-self.delay],
                targets=self.y_for_keras[self.delay:],
                sequence_length=self.seqlen,
                sequence_stride=self.sequence_stride,
                sampling_rate=self.sampling_rate,
                batch_size=self.batchsize,
                shuffle=False, #### CZEMU? RACZEJ  NIE BĘDĘ ROBIĆ I DUAL I KFOLD AND ONCE....
                start_index=fold_train_data_indices[0],
                end_index=fold_train_data_indices[-1]
            )
            
            val_kds = tf.keras.utils.timeseries_dataset_from_array(
                data=self.X_for_keras[:-self.delay],
                targets=self.y_for_keras[self.delay:],
                sequence_length=self.seqlen,
                sequence_stride=self.sequence_stride,
                sampling_rate=self.sampling_rate,
                batch_size=self.batchsize,
                shuffle=False,
                start_index=fold_val_data_indices[0],
                end_index=fold_val_data_indices[-1]
            )

            fold_datasets.append((train_kds, val_kds))
            fold_indices.append((fold_train_data_indices, fold_val_data_indices))

        test_kds = None
        if self.testsetatlast:
            test_kds = tf.keras.utils.timeseries_dataset_from_array(
                data=self.X_for_keras[:-self.delay],
                targets=self.y_for_keras[self.delay:],
                sequence_length=self.seqlen,
                sequence_stride=self.sequence_stride,
                sampling_rate=self.sampling_rate,
                batch_size=self.batchsize,
                shuffle=False,
                start_index=self.qty_train + self.qty_val,
                end_index=test_end - 1
            )
        
        return fold_datasets, test_kds, fold_indices, test_indices