#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:09:12 2025; @author: sylwia
"""
import numpy as np; import pandas as pd
from pathlib import Path
#
from dataclasses import dataclass

@dataclass
class DataConfig:
    """
    Konfiguracja danych - parametry które rzadko się zmieniają.
    """
    # Dane wejściowe WYMAGANE
    projekt_akronim: str
    plik_dane: str
    kols_map: dict
    kol_target: str

    
    # Dane wejściowe OPCJONALNE (mają wartości domyślne)
    kol_flagA: str = None ###### to musi być kol bool, pewnie można to jakoś pro zapisać
    intkols_to_scale: list = None # NIE: intkols_to_scale: list = [] bo wtedy wszystkie obiekty DataConfig, 
    #                   które nie dostaną własnej listy, będą dzielić dokładnie tę samą listę w pamięci !!!
    sca_estymator: str = "MaxAbs"
    frac_val: float = 0.25
    frac_test: float = 0.25

    # Zakresy czasowe (WYMAGANE tylko dla mr)
    date_start: str = '2000-01-01 00:00:00'
    date_end:   str = '2000-12-01 00:00:00'
    #
    out1_start: str = '2000-01-01 00:00:00'
    out1_end:   str = '2000-01-02 00:00:00'
    out2_start: str = '2000-02-01 00:00:00'
    out2_end:   str = '2000-02-02 00:00:00'


    def __post_init__(self):
        """Konwertuj stringi dat na pandas datetime i utwórz katalogi projektu."""
        # Konwersja zakresów czasowych
        self.dates = {
            'start':      pd.to_datetime(self.date_start),
            'end':        pd.to_datetime(self.date_end),
            'out1_start': pd.to_datetime(self.out1_start),
            'out1_end':   pd.to_datetime(self.out1_end),
            'out2_start': pd.to_datetime(self.out2_start),
            'out2_end':   pd.to_datetime(self.out2_end)
        }

        if self.intkols_to_scale is None:
            self.intkols_to_scale = []

        # Tworzenie ścieżek projektu
        base_dir = Path.cwd().parent
        self.mojerys_path  = base_dir / "mojerys"  / self.projekt_akronim
        self.mojecsvy_path = base_dir / "mojecsvy" / self.projekt_akronim

        # Upewnij się, że katalogi istnieją
        self.mojerys_path.mkdir(parents=True, exist_ok=True)
        self.mojecsvy_path.mkdir(parents=True, exist_ok=True)
        
        # To chyba tu, prawda?
        self.params_data_to_train = {
            k: v for k, v in vars(self).items() 
            if k not in ["params_data_to_train"]
            }


