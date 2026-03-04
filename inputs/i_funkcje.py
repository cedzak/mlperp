#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 09:02:48 2025; @author: sylwia
"""
import pandas as pd
import numpy as np
 
    
def drukuj_strukture(dane, poziom=0, max_elementy=5):
    """Rekurencyjnie drukuje dowolną strukturę danych,
    przycinając wszystkie sekwencje (listy, tuple, sety, numpy arrays) do max_elementy elementów."""
    
    wciecie = "   " * poziom
    
    # Obsługa numpy array
    if isinstance(dane, np.ndarray):
        print(f"{wciecie}numpy.array (shape={dane.shape}, dtype={dane.dtype}):")
        if dane.size > max_elementy:
            print(f"{wciecie}  Pierwsze {max_elementy} elementów: {dane.flat[:max_elementy]}")
        else:
            print(f"{wciecie}  Wszystkie elementy: {dane.flat[:dane.size]}")
    
    # Obsługa pandas DatetimeIndex
    elif isinstance(dane, pd.DatetimeIndex):
        print(f"{wciecie}DatetimeIndex (length={len(dane)}):")
        if len(dane) > max_elementy:
            print(f"{wciecie}  Pierwsze {max_elementy}: {list(dane[:max_elementy])}")
        else:
            print(f"{wciecie}  Wszystkie: {list(dane)}")
    
    # Obsługa słownika
    elif isinstance(dane, dict):
        print(f"{wciecie}DICTIONARY ({len(dane)} keys):")
        for k, v in dane.items():
            print(f"\n{wciecie}  K: {repr(k)}, V:")
            drukuj_strukture(v, poziom + 2, max_elementy)
    
    # Obsługa list, tuple, set
    elif isinstance(dane, (list, tuple, set)):
        typ_nazwa = type(dane).__name__
        dane_lista = list(dane)
        if len(dane_lista) > max_elementy:
            print(f"{wciecie}{typ_nazwa.capitalize()} ({len(dane_lista)} elementów, pokazuję tylko {max_elementy}):")
            dane_lista = dane_lista[:max_elementy]
        else:
            print(f"{wciecie}{typ_nazwa.capitalize()} ({len(dane_lista)} elementów):")
        for idx, item in enumerate(dane_lista):
            print(f"{wciecie}  [{idx}]:")
            drukuj_strukture(item, poziom + 2, max_elementy)
    
    # Prosty typ
    else:
        print(f"{wciecie}{repr(dane)}  (typ: {type(dane).__name__})")


# Użycie
# =============================================================================
# for idx, element in enumerate(krotka_datadictow_A):
#     print(f"\n\n{idx + 1} ELEMENT KROTKI krotka_datadictow_A: (typ: {type(element).__name__})")
#     drukuj_strukture(element)
#     print("=" * 70)
#     
# for idx, element in enumerate(val_kdictA):
#     print(f"\n\n{idx + 1} ELEMENT SLOWNIKA val_kdictA: (typ: {type(element).__name__})")
#     drukuj_strukture(element)
#     print("=" * 70)
# 
# for idx, element in enumerate(val_kdictA.values()):
#     print(f"\n\n{idx + 1} ELEMENT SLOWNIKA val_kdictA: (typ: {type(element).__name__})")
#     drukuj_strukture(element)
#     print("=" * 70)
# =============================================================================
