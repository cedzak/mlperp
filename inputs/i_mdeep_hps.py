#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 14:54:14 2025; @author: sylwia
#
wyniki tune -- bierz z plikow txt a nie z tabelek !!!
"""
class DeepModelHps:
    def __init__(self, cfg_id = "no_id",
                 initial_learning_rate=0.01, momentum=0.8,
                 dropout=0.4, regl2=0, 
                 patience_lr=80, patience_es=80, min_delta=0.01,
                 warstwy = ["1xGRU", "simple_TA", 64]
                 ):
        
        self.cfg_id = cfg_id
        #
        self.patience_es = patience_es
        self.patience_lr = patience_lr # raczej nie lubiane przez optymalizatory
        #
        self.initial_learning_rate = initial_learning_rate
        self.momentum = momentum
        self.dropout = dropout
        self.regl2 = regl2
        self.min_delta = min_delta
        #
        self.warstwy = warstwy
        #
        self.loss = "mse"
        self.metrics = ["mae"]
        #
        self.params_deep_model = {
                k: v for k, v in vars(self).items() if k not in ["params_deep_model", "SPRKOD"]
                }






wynik_naj_gdaciep_bigruta = DeepModelHps(cfg_id = "naj_gdaciep_bigruta",
                                initial_learning_rate=0.000629, momentum=0.8, 
                                dropout=0.4, regl2=0.0001, 
                                patience_lr=20, patience_es=20, min_delta=0.1, 
                                warstwy = ["1xGRU", None, 128], # warstwy[1]: kiedyś był tu wybór typu TA (simple_TA, directional_TA...); teraz obsolete bo zawsze używamy Keras MHA
)

wynik_naj_gdaciep_enc_dec = DeepModelHps(cfg_id = "naj_gdaciep_enc_dec",
                                initial_learning_rate=0.00261516771201487, momentum=0.5, 
                                dropout=0.2, regl2=1e-05, 
                                patience_lr=50, patience_es=80, min_delta=0.5, 
                                warstwy = ["2xGRU", None, 32], # warstwy[1]: kiedyś był tu wybór typu TA (simple_TA, directional_TA...); teraz obsolete bo zawsze używamy Keras MHA
)





















wynik_5020_rezimA = DeepModelHps(cfg_id = "5020A",
                                initial_learning_rate=0.01, momentum=0.8, 
                                dropout=0, regl2=0, 
                                patience_lr=20, patience_es=80, min_delta=0.1, 
                                warstwy = ["1xGRU", "directional_TA", 64],
                                )


wynik_5021_rezimA = DeepModelHps(cfg_id = "5021A",
                                initial_learning_rate=0.001, momentum=0.8, 
                                dropout=0, regl2=0, 
                                patience_lr=10, patience_es=10, min_delta=0.01, 
                                warstwy = ["1xGRU", "simple_TA", 64],
                                )


wynik_5022_rezimA = DeepModelHps(cfg_id = "5022A",
                                initial_learning_rate=0.005, momentum=0.5, 
                                dropout=0, regl2=0, 
                                patience_lr=50, patience_es=10, min_delta=0.01, 
                                warstwy = ["1xGRU", "simple_TA", 32],
                                )



wynik_502_rezimB = DeepModelHps(cfg_id = "502B",
                                initial_learning_rate=0.001, momentum=0.8, 
                                dropout=0, regl2=0, 
                                patience_lr=10, patience_es=80, min_delta=0.01, 
                                warstwy = ["1xGRU", "simple_TA", 32],
                                )



#### wyniki TUNER-H-15 sprzed podzialu na rezimA i rezimB
wynik_naj411 = DeepModelHps(cfg_id = "naj411",
                                initial_learning_rate=0.01, momentum=0.8, 
                                dropout=0.4, regl2=0, 
                                patience_lr=80, patience_es=80, min_delta=0.01, 
                                warstwy = ["1xGRU", "simple_TA", 64],
                                )


#### wyniki TUNER-H-15 biGRU-TA--SINGLE 2025.04.30
wynik_ktH15_biGRUta_single = DeepModelHps( cfg_id = "ktH15_biGRUta_single",
                                    initial_learning_rate = 0.005,
                                    momentum        = 0.8,
                                    dropout         = 0.2,
                                    regl2           = 0.0,
                                    patience_lr     = 20,
                                    patience_es     = 20,
                                    min_delta       = 0.1,
                                    warstwy = ["1xGRU", "simple_TA", 32]
                                    )


#### wyniki TUNER-H-15 LSTM--SINGLE 2025.04.29
wynik_ktH15_lstm = DeepModelHps( cfg_id = "ktH15_lstm_single",
                                    initial_learning_rate = 0.001,
                                    momentum        = 0.5,
                                    dropout         = 0.2,
                                    regl2           = 0.0,
                                    patience_lr     = 10,
                                    patience_es     = 20,
                                    min_delta       = 0.01,
                                    warstwy = ["lstm", "lstm", 16]
                                    )

wynik_ktH30_rezimA = DeepModelHps(  cfg_id = "ktH30A",
                                        initial_learning_rate = 0.005,
                                        momentum        = 0.8,
                                        dropout         = 0.2,
                                        regl2           = 0.0,
                                        patience_lr     = 10,
                                        patience_es     = 10,
                                        min_delta       = 0.1,
                                        warstwy = ["2xGRU", "simple_TA", 64]
                                        )

wynik_ktH30_rezimB = DeepModelHps(   cfg_id = "ktH30B",
                                        initial_learning_rate = 0.01,
                                        momentum        = 0.8,
                                        dropout         = 0.2,
                                        regl2           = 0.0,
                                        patience_lr     = 10,
                                        patience_es     = 50,
                                        min_delta       = 0.1,
                                        warstwy = ["1xGRU", "simple_TA", 128]
                                        )

#### wyniki TUNER-H-15 2025.04.25 ## oba mają ten sam 1st best
wynik_ktH15_oba = DeepModelHps( cfg_id = "ktH15oba",
                                    initial_learning_rate=0.01, momentum=0.8, 
                                    dropout=0.2, regl2=0, 
                                    patience_lr=10, patience_es=50, min_delta=0.1, 
                                    warstwy = ["1xGRU", "simple_TA", 128]
                                    )

#### wyniki TUNER-H-15 2025.04.25 ## rezimA 2nd best
wynik_ktH15_rezimA = DeepModelHps(  cfg_id = "ktH15A",
                                        initial_learning_rate=0.01, momentum=0.8, 
                                        dropout=0.4, regl2=0, 
                                        patience_lr=10, patience_es=20, min_delta=0.1, 
                                        warstwy = ["1xGRU", "directional_TA", 32]
                                        )

#### wyniki TUNER-H-15 2025.04.25 ## rezimB
wynik_ktH15_rezimB  = DeepModelHps(  cfg_id = "ktH15B",
                                        initial_learning_rate=0.01, momentum=0.5, 
                                        dropout=0.2, regl2=0, 
                                        patience_lr=20, patience_es=10, min_delta=0.1, 
                                        warstwy = ["2xGRU", "simple_TA", 64]
                                        )


















