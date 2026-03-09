#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nauka:
base = BaseBehavior()  # ❌ BŁĄD: TypeError: Can't instantiate abstract class
OSIĄGNĘŁAM TO DZIĘKI @abstractmethod
"""
import pandas as pd; import numpy as np

from abc import ABC, abstractmethod
from klasy_jesien.archit0_base import BaseArchitecture
from klasy_jesien.archit1_shallow import ShallowArchitecture
from klasy_jesien.archit2_deep import DeepArchitecture


class BaseBehavior(ABC):
    """
    Abstrakcyjna klasa bazowa dla behavach (single/dual).
    Określa STRATEGIĘ eksperymentu.
    """
    
    def __init__(self, archit_instance: BaseArchitecture):
        """
        Args:
            archit_instance: Instancja ShallowArchitecture lub DeepArchitecture
        """
        self.archit = archit_instance
        
        # Rozpoznaj typ i wyciągnij dane z architektury
        if isinstance(self.archit, DeepArchitecture):
            self.is_deep = True
            self.d3_kds = self.archit.d3_kds
                        
            self.train_kdict = self.d3_kds.train_kdict
            self.val_kdict = self.d3_kds.val_kdict
            self.test_kdict = self.d3_kds.test_kdict
            
            self.testsetatlast = self.d3_kds.testsetatlast
            
        elif isinstance(self.archit, ShallowArchitecture):
            self.is_deep = False
            self.d1_pds = self.archit.d1_pds
            
            
            
            
            ########## PLAN: CHYBA USUNĄĆ TĄ PREAMBUŁĘ I W SŁOWNIKACH PODODAWĆ d1_pds
            self.X_train = self.d1_pds.X_train
            self.X_val = self.d1_pds.X_val
            self.X_test = self.d1_pds.X_test
            self.y_train = self.d1_pds.y_train
            self.y_val = self.d1_pds.y_val
            self.y_test = self.d1_pds.y_test
            (self.pd_train_idx, self.pd_val_idx, self.pd_test_idx) = ( 
                self.d1_pds.pd_train_idx, self.d1_pds.pd_val_idx, self.d1_pds.pd_test_idx)
            
            
            
            
            self.train_pdict = {'X_pds': self.X_train,  
                                'tvt_name': "train",
                                'y_indices': self.pd_train_idx, 
                                'y_actuals': self.y_train
                                }
            self.val_pdict = {'X_pds': self.X_val,  
                                'tvt_name': "val",
                                'y_indices': self.pd_val_idx, 
                                'y_actuals': self.y_val
                                }
            self.test_pdict = {'X_pds': self.X_test,  
                                'tvt_name': "test",
                                'y_indices': self.pd_test_idx, 
                                'y_actuals': self.y_test
                                }
            
            
            
            
            
            self.testsetatlast = self.d1_pds.testsetatlast        

        else:
            raise TypeError(f"Nieznany typ architektury: {type(self.archit)}")

        self._prepare_tvt_data_for_tvt()

 
    @abstractmethod
    def _prepare_tvt_data_for_tvt(self):
        """
        Przygotowuje dane do treningu/predykcji.
        
        Args:
            kds: Dataset (tf.data.Dataset lub tuple (X, y))
            indices: Indeksy próbek
            has_rezimA: Maska boolean określająca które próbki są "trudne" (tylko dla dual)
        Returns:
            dict: Słownik {model_name: (data, indices)}
                  Dla single: {'main': (kds, indices)}
                  Dla dual: {'rezimA': (kds_t, idx_t), 'rezimB': (kds_o, idx_o)}
        """
        pass
    


    @abstractmethod
    def run_runs_and_get_results_for_dfres(self, dmhpsA=None, dmhpsB=None):
        """Metoda musi być zaimplementowana w podklasach
        DLA SHALLOW NIE MA ŻADNYCH HPS !!!!"""
        pass
    

    

    
    def _log_po_tasowaniu(self, kds, label, y_indices, n_samples, y_actuals_before):
        """Loguje info o przetasowanym train kds do data_prep_file.
        y_actuals_before: y w oryginalnej kolejności (1D flat array).
        Porównuje jabłka z jabłkami: w seq2seq reshape do okien, w seq2one skalary.
        """
        data_prep_file = self.d3_kds.d2_kluski.data_prep_file
        y_batch = next(iter(kds))[1].numpy()  # (batch, seqlen) dla seq2seq, (batch,) dla seq2one

        if y_batch.ndim > 1:
            # seq2seq / aligned: każda próbka to okno — pokaż 3 całe okna
            seqlen = y_batch.shape[1]
            y_przed = y_actuals_before[:3 * seqlen].reshape(3, seqlen)
            y_po    = y_batch[:3]
        else:
            # seq2one: każda próbka to skalar
            y_przed = y_actuals_before[:3]
            y_po    = y_batch[:3]

        with open(data_prep_file, "a") as f:
            f.write(
                f"\n{'---' * 15}\n"
                f"[SHUFFLE] {label}\n"
                f"  n_samples: {n_samples}\n"
                f"  y_indices zakres: {y_indices[0]} .. {y_indices[-1]}\n"
                f"  pierwsze 3 y PRZED tasowaniem: {y_przed}\n"
                f"  pierwsze 3 y PO  tasowaniu:    {y_po}\n"
                f"{'---' * 15}\n"
            )

    # Bez @abstractmethod = MOŻESZ nadpisać → dla metod opcjonalnych
    def sklej_predykcje_indeksami(self, *args, **kwargs):
        """
        Domyślna implementacja - dla Single po prostu przepuszcza dane.
        DualBehavior nadpisuje z prawdziwą logiką.
        
        czemu 3 a nie 4, przecież dochodzi "self": return args[0], args[1], args[2] if len(args) >= 3 else (None, None, None)
        Dobra obserwacja, ale self NIE trafia do *args!
        """
        # Zakładam że args to: indices, actual, predictions
        return args[0], args[1], args[2] # NIE CHCĘ, WOLĘ ERROR NIŻ TAKIE ZABEZPIECZENIE: if len(args) == 3 else (None, None, None)
    


    @abstractmethod
    def get_count(self):
        """
        Zwraca liczbę modeli potrzebnych dla tej strategii.
        
        Returns:
            int: 1 dla single, 2 dla dual
        """
        pass
    
    @abstractmethod
    def get_names(self):
        """
        Zwraca nazwy modeli dla tej strategii.
        
        Returns:
            list: ['main'] dla single, ['rezimA', 'rezimB'] dla dual
        """
        pass
