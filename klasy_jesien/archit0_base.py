#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Architecture - abstrakcyjna klasa bazowa dla wszystkich architektur modeli

# nauka
idx_all = np.array([30, 10, 20])
sort_order = np.argsort(idx_all)  # wynik: array([1, 2, 0])
print(sort_order)
# bo jak zrobie [1 2 0] to [30, 10, 20] ustawię w kolejnosci rosnacej
"""
import os; import logging
from pathlib import Path
import numpy as np; import pandas as pd

logger = logging.getLogger(__name__)
#
from abc import ABC, abstractmethod


class BaseArchitecture(ABC):
    """
    Abstrakcyjna klasa bazowa definiująca interfejs dla architektur modeli.
    Określa JAK model jest zbudowany i trenowany.
    
    TU NIE MOGĘ DAĆ GOTOWYCH DATASETÓW Z KLASY k1 BO W PRZYPADKU DUAL APPROACH BĘDĄ TO INNE DATASETY !!
    A TO KLASĘ ARCHIT WKŁADAM DO KLASY APPROACH A NIE ODWROTNIE.
    """

    def __init__(self, ilerunow, timestampplus=None,
                       batchsize=None, seqlen=None):
        self.timestampplus = timestampplus
        self.ilerunow = ilerunow
        self.batchsize = batchsize
        self.seqlen = seqlen

    @abstractmethod
    def build(self, model_config):
        """
        Buduje architekturę modelu.
        Args:
            model_config: Obiekt konfiguracyjny z parametrami modelu
        Returns:
            Zbudowany model (gotowy do treningu)
        """
        pass
    
    @abstractmethod
    def fit(self, train_data, val_data=None, **kwargs):
        """
        Trenuje model.
        Args:
            train_data: Dane treningowe (format zależy od architektury)
            val_data: Dane walidacyjne (opcjonalne)
            **kwargs: Dodatkowe parametry (callbacks, EPOCHS, etc.)
        Returns:
            Historia treningu lub None
        """
        pass
    
    @abstractmethod
    def predict(self, data):
        """
        Wykonuje predykcję.
        Args:
            data: Dane do predykcji
        Returns:
            numpy array z predykcjami
        """
        pass
    
    def save(self, filepath):
        """
        Zapisuje model do pliku (może być nadpisane w podklasach).
        Args:
            filepath: Ścieżka do zapisu
        """
        raise NotImplementedError("Metoda save nie została zaimplementowana dla tej architektury")
    
    def load(self, filepath):
        """
        Wczytuje model z pliku (może być nadpisane w podklasach).
        Args:
            filepath: Ścieżka do wczytania
        """
        raise NotImplementedError("Metoda load nie została zaimplementowana dla tej architektury")

    @abstractmethod
    def get_name(self):
        """dla deep jest to po prostu string 'deep',
        dla shallow string 'shallow'
        """
        pass
        
    

    def _stworz_one_run_results_dict(self, run_idx, data_name,
                                        t_train_min, t_pred_sec,
                                        val_mse, val_mae, test_mse, test_mae,
                                        used_epochs=None):
        return locals()
        ## !! zamiast: 
        # return {
        #     'run_idx'  : run_idx,
        #     'data_name': data_name,
        #     't_train_min': t_train_min,
        #     't_pred_sec': t_pred_sec,
        #     'val_mse' : val_mse,
        #     'val_mae' : val_mae,
        #     'test_mse' : test_mse,
        #     'test_mae' : test_mae
        #     'used_epochs': used_epochs,
        #     }
    
        
    def save_best_run_results_dict(self, metrics_dict):
        # Dodaj timestamplus jako pierwszy klucz w słowniku
        metrics_dict_with_timestamp = {'timestamp': self.timestampplus, 
                                       'ilerunow': self.ilerunow,
                                       'batchsize': self.batchsize, 
                                       'seqlen': self.seqlen, 
                                       **metrics_dict}
        
        df = pd.DataFrame([metrics_dict_with_timestamp])
        
        full_output_path = self.results_path / f"best_run__all_models.csv"
        # Sprawdź, czy plik istnieje
        file_exists = os.path.isfile(full_output_path)
        # Zapisz, dopisując jeśli trzeba
        df.to_csv(full_output_path, mode='a', index=False, header=not file_exists)
        print(f"Dopisano best run do csv w dir parent!")



    def calc_and_save_all_runs_summary_stats_from_predictions(self, 
                                                  all_runs_metrics_dicts_list):
        
        summary_stats = {}
        
        # Pobierz i iteruj przez wszystkie klucze Z PIERWSZEGO Z BRZEGU słownika
        all_keys_of_metrics_dicts = all_runs_metrics_dicts_list[0].keys()
        
        for key in all_keys_of_metrics_dicts:
            values = [r[key] for r in all_runs_metrics_dicts_list]
            # Sprawdź czy wartości są numeryczne
            if isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
                # Oblicz mean i std dla wszystkich kluczy numerycznych
                summary_stats[f"{key}_mean"] = round(np.mean(values), 3)
                summary_stats[f"{key}_std"] = round(np.std(values), 3)


        # Dodaj timestamplus jako pierwszy klucz w słowniku
        metrics_dict_with_timestamp = {'timestampplus': self.timestampplus, 
                                       'ilerunow': self.ilerunow, 
                                       'batchsize': self.batchsize, 
                                       'seqlen': self.seqlen, 
                                       **summary_stats}
        
        df = pd.DataFrame([metrics_dict_with_timestamp])
        
        full_output_path = self.results_path / f"summary_stats__all_models.csv"
        # Sprawdź, czy plik istnieje
        file_exists = os.path.isfile(full_output_path)
        # Zapisz, dopisując jeśli trzeba
        df.to_csv(full_output_path, mode='a', index=False, header=not file_exists)
        print(f"Dopisano summary stats wyliczone z predictions do csv w dir parent!")


    