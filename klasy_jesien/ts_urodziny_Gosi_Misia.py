#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 18:32:38 2025; @author: sylwia
"""
import pandas as pd
from pathlib import Path


#%% metody

def find_and_move_segment_od_konca(self, df, ttt, offset, segm_length, licz_porz=40000): # grok
    """Moves a segment of df based on target_at_ttt ± 1 to a new location.
    ttt - miejsce gdzie insert segment
    licz_porz - miejsce gdzie sprawdzam jak sie zmienia zbior; musi byc tak dobrana aby odczuwala zmiany
    offset - o ile sie odsunac od konca
    segm_length - dlugosc segmentu        
    """
    
    dt_idx_dla_licz_porz = df.index[licz_porz]
    
    print(f"\n~~~fams 1~~~ :   "
          f"target(licz_porz)={df['target'].iloc[licz_porz]}, indeks na licz_porz={dt_idx_dla_licz_porz}")
    df_index = df.index
    # Resetuj indeks na numeryczny (0, 1, 2, ...)
    df = df.reset_index(drop=True)
    print(f"\n~~~fams 2~~~ :   "
          f"target(licz_porz)={df['target'].iloc[licz_porz]}, indeks na licz_porz={df.index[licz_porz]}")
    
    target_at_ttt = df['target'].iloc[ttt]
    print(f"\n!!! ttt={ttt}, target_at_ttt={target_at_ttt}")
            
    # Step 1: Move back o offset (e.g. 25,000) from last index, find matching index
    last_idx = df.index[-1]
    rough_segment_end = last_idx - offset
    if rough_segment_end < 0:
        raise ValueError("Backtrack beyond df")
    search_df = df.loc[:rough_segment_end]
    matching_rows = search_df[search_df['target'].between(target_at_ttt - 1, target_at_ttt + 1)]
    if matching_rows.empty:
        raise ValueError("No matches before rough_segment_end")
    segment_end_idx = matching_rows.index[-1]
        
    # Step 2: Backtrack segm_length (e.g. 50,000), find anrezimB matching index
    rough_segment_start = segment_end_idx - segm_length
    if rough_segment_start < 0:
        raise ValueError("Backtrack beyond df")
    search_df = df.loc[:rough_segment_start]
    matching_rows = search_df[search_df['target'].between(target_at_ttt-1, target_at_ttt+1)]
    if matching_rows.empty:
        raise ValueError("No matches before rough_segment_start")
    segment_start_idx = matching_rows.index[-1]

    # Step 3: Extract segment
    segment = df.loc[segment_start_idx:segment_end_idx]
    
    if segment.empty:
        raise ValueError("Empty segment")

    print(f"!!! segment_start_idx={segment_start_idx}")
    print(f"!!! segment_end_idx={segment_end_idx}")
    print(f"len segmentu = {len(segment)}")
    print(f"len df przed usunieciem segmentu = {len(df)}")
           
    
    # Step 4: Move segment to new location
    print(f"\n~~~fams 3~~~ przed usunieciem segmentu (powinno być takie samo): "
          f"target(licz_porz)={df['target'].iloc[licz_porz]}, indeks na licz_porz={df.index[licz_porz]}")
    is_default_index = df.index.equals(pd.RangeIndex(start=0, stop=len(df)))
    print(f"Indeks domyślny: {is_default_index}")

    result_df = df.drop(index=range(segment_start_idx, segment_end_idx + 1))
    
    try: # moze byc za krotki ten pozostaly zbior
        print(f"\n~~~fams 4~~~ target po usunieciu segmentu: "
              f"target(licz_porz)={result_df['target'].iloc[licz_porz]}, indeks na licz_porz={result_df.index[licz_porz]}")
    except:
        pass
    print(f"len df po usunieciu segmentu = {len(result_df)}")
    is_default_index = result_df.index.equals(pd.RangeIndex(start=0, stop=len(result_df)))
    print(f"Indeks domyślny: {is_default_index}")
    
    # Insert at specified index
    before = result_df.loc[:ttt - 1]
    after = result_df.loc[ttt:]
    result_df = pd.concat([before, segment, after], ignore_index=True)
    
    print(f"\n~~~fams 5~~~ target po wstawieniu segmentu w innym miejscu: "
          f"target(licz_porz)={result_df['target'].iloc[licz_porz]}, indeks na licz_porz={result_df.index[licz_porz]}")
    print(f"len df po wstawieniu segmentu w innym miejscu = {len(result_df)}")
    is_default_index = result_df.index.equals(pd.RangeIndex(start=0, stop=len(result_df)))
    print(f"Indeks domyślny: {is_default_index}")
    
    # PRZYWROC DATATIME INDEX
    result_df = result_df.reset_index(drop=True)
    result_df.index = df_index
    try:
        print(f"\n~~~fams 6~~~ :   "
              f"target(licz_porz)={result_df['target'].iloc[licz_porz]}")
        print(f"~~~fams 6~~~ :   "
              f"target(dt_idx_dla_licz_porz)={result_df['target'].loc[dt_idx_dla_licz_porz]}")
    except:
        print("\n~~~fams 6~~~ :   nie robi")
    
    return result_df #, dt_idx_dla_licz_porz


def find_and_move_segment_od_poczatku(self, df, ttt, rough_segment_start, rough_segment_end): # grok
    """Moves a segment of df based on target_at_ttt ± 1 to a new location.
    ttt - miejsce gdzie insert segment
    licz_porz - miejsce gdzie sprawdzam jak sie zmienia zbior; musi byc tak dobrana aby odczuwala zmiany
    offset - o ile sie odsunac od konca
    segm_length - dlugosc segmentu        
    """
    # zatrzymaj sobie index dziewiczy
    df_index = df.index
    # Resetuj indeks na numeryczny (0, 1, 2, ...)
    df = df.reset_index(drop=True)

    target_at_ttt = df['target'].iloc[ttt]   
    
    
    # Step 1: segment_start_idx
    search_df = df.loc[rough_segment_start:]
    matching_rows = search_df[search_df['target'].between(target_at_ttt - 1, target_at_ttt + 1)]
    if matching_rows.empty:
        raise ValueError("No matches before rough_segment_end")
    segment_start_idx = matching_rows.index[0]
    
        
    # Step 2: segment_end_idx
    search_df = df.loc[:rough_segment_end]
    matching_rows = search_df[search_df['target'].between(target_at_ttt-1, target_at_ttt+1)]
    if matching_rows.empty:
        raise ValueError("No matches before end")
    segment_end_idx = matching_rows.index[-1]

    # Step 3: Extract segment
    segment = df.loc[segment_start_idx:segment_end_idx]
    
    if segment.empty:
        raise ValueError("Empty segment")

    print(f"!!! segment_start_idx={segment_start_idx}")
    print(f"!!! segment_end_idx={segment_end_idx}")
    print(f"len segmentu = {len(segment)}")
    print(f"len df przed usunieciem segmentu = {len(df)}")
           
    
    # Step 4: Move segment to new location
    is_default_index = df.index.equals(pd.RangeIndex(start=0, stop=len(df)))
    print(f"Indeks domyślny: {is_default_index}")
    result_df = df.drop(index=range(segment_start_idx, segment_end_idx + 1))

    print(f"len df po usunieciu segmentu = {len(result_df)}")
    is_default_index = result_df.index.equals(pd.RangeIndex(start=0, stop=len(result_df)))
    print(f"Indeks domyślny: {is_default_index}")
    
    # Insert at specified index
    before = result_df.loc[:ttt - 1]
    after = result_df.loc[ttt:]
    result_df = pd.concat([before, segment, after], ignore_index=True)
    
    print(f"len df po wstawieniu segmentu w innym miejscu = {len(result_df)}")
    is_default_index = result_df.index.equals(pd.RangeIndex(start=0, stop=len(result_df)))
    print(f"Indeks domyślny: {is_default_index}")
    
    # PRZYWROC DATATIME INDEX
    result_df = result_df.reset_index(drop=True)
    result_df.index = df_index

    return result_df



#%% jak ich używałam
class DataConfig:
    def __init__(self, timestamp, MOJERYS_Path, SPRKOD=False, 
                 kols_map=kols_map_mr, wybor_kols = "best_features_rfr_xgbr"):
# ...
    # def step1_prepare_df_customised_and_df_virgin(self): 
        # ...
        
        
        #### 5 balansowanie zbiorow: ## urodziny Gosi
        print("\n~~~~~~~~~~~~~~~~ttt  urodziny Gosi cz.1/2 ~~~~~~~~~~~~~~~~~")
        ttt = 242*1000
        rough_segment_start = 22*1000
        rough_segment_end = 52*1000
        #
        print("\n\n~~~~~~~~~~~~~~~~ttt virgin~~~~~~~~~~~~~~~~~")
        df_virgin = self.find_and_move_segment_od_poczatku(df_virgin, ttt, rough_segment_start, rough_segment_end)
        #
        print("\n\n~~~~~~~~~~~~~~~~ttt customized~~~~~~~~~~~~~~~~~")
        df_customised = self.find_and_move_segment_od_poczatku(df_customised, ttt, rough_segment_start, rough_segment_end)
        #
        #
        print("\n~~~~~~~~~~~~~~~~ttt  urodziny Gosi cz.2/2 ~~~~~~~~~~~~~~~~~")
        ttt = 0 
        offset = 0
        segm_length = 20*1000 # celuj w segment o dlugosci 15 tys. min
        #
        print("\n\n~~~~~~~~~~~~~~~~ttt virgin~~~~~~~~~~~~~~~~~")
        df_virgin = self.find_and_move_segment_od_konca(df_virgin, ttt, offset, segm_length)
        #
        print("\n\n~~~~~~~~~~~~~~~~ttt customized~~~~~~~~~~~~~~~~~")
        df_customised = self.find_and_move_segment_od_konca(df_customised, ttt, offset, segm_length)


        
        #### 5 balansowanie zbiorow: ## urodziny Misia
        print("\n~~~~~~~~~~~~~~~~ttt virgin i customized ~~~~~~~~~~~~~~~~~")
        ttt = int(4e3) # zaraz na poczatku wstaw segment z konca
        licz_porz=-int(40e3) # sprawdzaj zmiany na minucie 40 tys. od konca
        offset = int(25e3) # odsun sie od konca o 25 tys. minut
        segm_length = int(50e3) # celuj w segment o dlugosci 50 tys. min
        print("\n\n~~~ttt~~~", df_virgin['target'].iloc[licz_porz])
        
        df_virgin, dt_idx_dla_licz_porz = self.find_and_move_segment_ttt(df_virgin, ttt, licz_porz, offset, segm_length)
        df_customised, dt_idx_dla_licz_porz = self.find_and_move_segment_ttt(df_customised, ttt, licz_porz, offset, segm_length)
        
        try:
            print("~~~ttt~~~", df_virgin['target'].loc[dt_idx_dla_licz_porz])
        except:
            print("~~~ttt~~~: wyglada na to ze jest indeks numeryczny")
        #
        #
        print("\n~~~~~~~~~~~~~~~~ttt  virgin i customized ~~~~~~~~~~~~~~~~~")
        ttt = 186600 # zaraz po train set wstaw segment z konca
        licz_porz=-int(40e3) # sprawdzaj zmiany na minucie 40 tys. od konca
        print(licz_porz)
        offset = 10 # odsun sie od konca o 10 minut
        segm_length = int(15e3) # celuj w segment o dlugosci 15 tys. min
        print("\n\n~~~ttt~~~", df_virgin['target'].iloc[licz_porz])
        
        df_virgin, dt_idx_dla_licz_porz = self.find_and_move_segment_ttt(df_virgin, ttt, licz_porz, offset, segm_length)
        df_customised, dt_idx_dla_licz_porz = self.find_and_move_segment_ttt(df_customised, ttt, licz_porz, offset, segm_length)
        
        try:
            print("~~~ttt~~~", df_virgin['target'].loc[dt_idx_dla_licz_porz])
        except:
            print("~~~ttt~~~: wyglada na to ze jest indeks numeryczny")



#%% A GROK ZROBIŁ Z TEGO COŚ TAKIEGO: 
# Created on Fri Sep 19 18:16:15 2025

def find_segment(self, df, target_value, start_idx, end_idx, direction='forward'):
    """Znajduje indeks segmentu w DataFrame na podstawie wartości target ± 1.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi (z numerycznym indeksem).
        target_value (float): Wartość target do wyszukania.
        start_idx (int): Początkowy indeks wyszukiwania.
        end_idx (int): Końcowy indeks wyszukiwania.
        direction (str): Kierunek wyszukiwania ('forward' lub 'backward').
    
    Returns:
        int: Indeks pierwszego lub ostatniego pasującego wiersza.
    """
    search_df = df.loc[start_idx:end_idx] if direction == 'forward' else df.loc[:end_idx]
    matching_rows = search_df[search_df['target'].between(target_value - 1, target_value + 1)]
    if matching_rows.empty:
        logger.error(f"No matches in range {start_idx}:{end_idx}")
        raise ValueError(f"No matches in range {start_idx}:{end_idx}")
    return matching_rows.index[0 if direction == 'forward' else -1]


def move_segment(self, df, segment_start_idx, segment_end_idx, insert_idx):
    """Przenosi segment danych do nowej lokalizacji w DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi (z numerycznym indeksem).
        segment_start_idx (int): Początkowy indeks segmentu.
        segment_end_idx (int): Końcowy indeks segmentu.
        insert_idx (int): Indeks, gdzie wstawić segment.
    
    Returns:
        pd.DataFrame: DataFrame z przeniesionym segmentem (z numerycznym indeksem).
    """
    segment = df.loc[segment_start_idx:segment_end_idx]
    if segment.empty:
        logger.error("Empty segment")
        raise ValueError("Empty segment")
    
    logger.info(f"Moving segment: start={segment_start_idx}, end={segment_end_idx}, insert_idx={insert_idx}")
    result_df = df.drop(index=range(segment_start_idx, segment_end_idx + 1))
    before = result_df.loc[:insert_idx - 1]
    after = result_df.loc[insert_idx:]
    result_df = pd.concat([before, segment, after], ignore_index=True)
    return result_df


def process_segment(self, df, insert_idx, start_idx, end_idx, direction='forward'):
    """Łączy wyszukiwanie i przenoszenie segmentu, zachowując oryginalny indeks.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi (może mieć indeks datetime).
        insert_idx (int): Indeks, gdzie wstawić segment.
        start_idx (int): Początkowy indeks wyszukiwania.
        end_idx (int): Końcowy indeks wyszukiwania.
        direction (str): Kierunek wyszukiwania ('forward' lub 'backward').
    
    Returns:
        pd.DataFrame: Przetworzony DataFrame z przywróconym oryginalnym indeksem.
    """
    # Zachowaj oryginalny indeks
    original_index = df.index
    # Resetuj indeks na numeryczny
    df = df.reset_index(drop=True)
    
    target_value = df['target'].iloc[insert_idx]
    segment_start_idx = self.find_segment(df, target_value, start_idx, end_idx, direction)
    segment_end_idx = self.find_segment(df, target_value, segment_start_idx, end_idx, 
                                      'backward' if direction == 'forward' else 'forward')
    
    result_df = self.move_segment(df, segment_start_idx, segment_end_idx, insert_idx)
    # Przywróć oryginalny indeks
    result_df.index = original_index
    logger.info(f"Processed segment: len={len(result_df)}")
    return result_df


def find_and_move_segment_from_end(self, df, insert_idx, offset, segment_length, checkpoint_idx=40000):
    """Przenosi segment z końca DataFrame na podstawie offsetu i długości.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi (może mieć indeks datetime).
        insert_idx (int): Indeks, gdzie wstawić segment.
        offset (int): Odsunięcie od końca DataFrame (w liczbie wierszy).
        segment_length (int): Długość segmentu (w liczbie wierszy).
        checkpoint_idx (int): Indeks do sprawdzenia zmian.
    
    Returns:
        pd.DataFrame: Przetworzony DataFrame z przywróconym indeksem.
    """
    # Zachowaj oryginalny indeks
    original_index = df.index
    # Resetuj indeks na numeryczny
    df = df.reset_index(drop=True)
    
    last_idx = len(df) - 1  # Używamy długości DataFrame zamiast indeksu datetime
    rough_segment_end = last_idx - offset
    rough_segment_start = rough_segment_end - segment_length
    
    if rough_segment_start < 0 or rough_segment_end < 0:
        logger.error("Backtrack beyond DataFrame bounds")
        raise ValueError("Backtrack beyond DataFrame bounds")
    
    try:
        logger.info(f"Checkpoint target at idx {checkpoint_idx}: {df['target'].iloc[checkpoint_idx]}")
    except IndexError:
        logger.warning(f"Checkpoint index {checkpoint_idx} out of bounds")
    
    result_df = self.process_segment(df, insert_idx, rough_segment_start, rough_segment_end, direction='backward')
    # Przywróć oryginalny indeks
    result_df.index = original_index
    return result_df


def find_and_move_segment_from_start(self, df, insert_idx, rough_segment_start, rough_segment_end):
    """Przenosi segment z początku DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame z danymi (może mieć indeks datetime).
        insert_idx (int): Indeks, gdzie wstawić segment.
        rough_segment_start (int): Przybliżony początek segmentu.
        rough_segment_end (int): Przybliżony koniec segmentu.
    
    Returns:
        pd.DataFrame: Przetworzony DataFrame z przywróconym indeksem.
    """
    return self.process_segment(df, insert_idx, rough_segment_start, rough_segment_end, direction='forward')





### Kluczowe zmiany
# =============================================================================
# 1. **Resetowanie indeksu**:
#    - W metodzie `find_and_move_segment_from_end` i `process_segment` indeks jest resetowany na numeryczny (`df.reset_index(drop=True)`) przed wykonaniem operacji arytmetycznych, aby uniknąć błędu z `Timestamp`.
#    - Oryginalny indeks (np. datetime) jest zapisywany w zmiennej `original_index` i przywracany na końcu, aby zachować zgodność z Twoim kodem.
# 
# 2. **Operacje na długości DataFrame**:
#    - Zamiast odejmować `offset` od `last_idx` jako `Timestamp`, używam `len(df) - 1` do obliczenia ostatniego indeksu numerycznego. To pozwala na bezpieczne operacje arytmetyczne (`rough_segment_end = last_idx - offset`).
# 
# 3. **Obsługa błędów**:
#    - Dodano sprawdzanie, czy `rough_segment_start` lub `rough_segment_end` nie są ujemne, z odpowiednim komunikatem w loggerze.
#    - Sprawdzanie `checkpoint_idx` w `try/except`, aby uniknąć błędów, jeśli indeks jest poza zakresem.
# 
# 4. **Logowanie**:
#    - Logger jest używany do zapisywania informacji o postępie i błędach, zgodnie z Twoim pytaniem o logowanie z wielu plików. Konfiguracja w `logging.basicConfig` działa dla całego projektu, więc każdy moduł korzystający z `logging.getLogger(__name__)` zapisze logi do `data_processing.log`.
# 
# ### Jak to działa z Twoim kodem?
# - **Plik `config.py`** (dawniej `adamiewa.py`):
#   - Zawiera konfigurację, w tym `plik_dane`, `MOJERYS_Path`, `kols_map_mr` itd.
#   - Może zawierać konfigurację loggera, jeśli chcesz, aby była globalna:
#     ```python
#     # config.py
#     import logging
#     logging.basicConfig(
#         level=logging.INFO,
#         plik_dane='data_processing.log',
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )
#     ```
# 
# - **Plik `data_processor.py`**:
#   - Zawiera klasy `TimeSeriesProcessor` i `DataConfig` z poprawionym kodem.
#   - Używa loggera z `logging.getLogger(__name__)`.
# 
# - **Przykład użycia**:
#   ```python
#   # main.py
#   from config import plik_dane, MOJERYS_Path, kols_map_mr
#   from data_processor import DataConfig
# 
#   config = DataConfig(timestamp="20250919", MOJERYS_Path=MOJERYS_Path, kols_map=kols_map_mr)
#   df_customised, df_virgin = config.prepare_df_and_df_virgin(plik_dane)
# =============================================================================


### Odpowiedź na błąd
# =============================================================================
# Błąd `TypeError: Addition/subtraction of integers and integer-arrays with Timestamp is no longer supported` został rozwiązany przez:
# - Resetowanie indeksu na numeryczny przed operacjami arytmetycznymi.
# - Używanie `len(df) - 1` zamiast bezpośredniego odejmowania od `last_idx`.
# - Przywracanie oryginalnego indeksu po zakończeniu operacji, aby zachować spójność z Twoim kodem.
# 
# ### Dodatkowe uwagi
# - **Logowanie z wielu plików**: Konfiguracja loggera w `data_processor.py` działa dla tego pliku, ale jeśli chcesz, aby wszystkie moduły zapisywały do tego samego pliku `data_processing.log`, upewnij się, że `logging.basicConfig` jest wywołane tylko raz (np. w `config.py` lub `main.py`). W innych plikach po prostu używaj `logger = logging.getLogger(__name__)` bez ponownej konfiguracji.
# - **Dalsza refaktoryzacja**: Jeśli masz inne metody w `DataConfig`, które wymagają podobnego traktowania indeksu (np. `find_and_move_segment_from_start`), daj znać, a dostosuję je w ten sam sposób.
# - **Testowanie**: Rozważ dodanie testów jednostkowych dla `TimeSeriesProcessor`, aby sprawdzić, czy operacje na segmentach działają poprawnie z różnymi typami indeksów.
# 
# Jeśli masz dodatkowe pytania lub chcesz, abym dostosował inne części kodu, napisz!
# =============================================================================




