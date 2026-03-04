#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 18:49:06 2025; @author: sylwia
"""
import logging
logger = logging.getLogger(__name__)
#
import pandas as pd
from pathlib import Path


class TimeSeriesProcessor:
    """Klasa odpowiedzialna za przetwarzanie i manipulację szeregami czasowymi."""
    
    def add_licz_porz_kol(self, df):
        """zrób coś by mieć pewność że indeksy są ok 
           nawet po wycieczce do numerycznego indeksu"""
        df_temp = df.reset_index()
        df_temp['kol_licz_porz'] = df_temp.index+1000
        print("\n!!!!!!!!!!!! testuje zachowanie indeksow:")
        for i in range(0, 500, 100):
            print(i, df_temp['kol_licz_porz'].iloc[i])
            

    
    def get_quantities_from_proportions(self, df_index, frac_val, frac_test):
        """zwraca ilosc probek w setach: train_set, val_set i test_set"""
        
        # Check if indices are sorted (they should be)
        print("\n\njestem w ts_processor, w funkcji get_quantities_from_proportions. df_index.is_monotonic_increasing?")
        print(df_index.is_monotonic_increasing)  # Should be True  
        
        qty_val = int(frac_val * len(df_index)); print(qty_val)
        qty_test = int(frac_test * len(df_index)); print(qty_test)
        qty_train = len(df_index) - qty_val - qty_test; print(len(df_index)); print(qty_train)

        return qty_train, qty_val, qty_test

    
    def get_quantities_from_dates(self, df_index):

        print("\n\nget_quantities_from_dates")
        # Check if indices are sorted (they should be)
        print(df_index.is_monotonic_increasing)  # Should be True    

        granice_str = ['2023-10-04 00:00:00', 
                       '2023-10-29 00:00:00']
        granice_pos = []
        
        # szukam wiersza w ktorym chce przekroic
        for string_idx in granice_str:
            datetime_idx = pd.to_datetime(string_idx)
            # Find the position where the desired datetime
            pos = df_index.searchsorted(datetime_idx)
            granice_pos.append(pos)
            print(f"The first index after {datetime_idx} is: {df_index[pos]}")
            # any problem?
            if pos > len(df_index):
                print("CO TU SIE PISALO ZEBY BYL ERROR i zatrzymanie kodu ???")

        qty_train = granice_pos[0]
        qty_val   = granice_pos[1] - qty_train
        qty_test  = len(df_index) - qty_train - qty_val

        return qty_train, qty_val, qty_test








    def select_rows_and_get_quantities(self, df, dates):

        # PO CO JEST TO? #################################################### 2025.09.19 -- niepotrzebne, pewnie zapomniałam usunąć
# =============================================================================
#         aaa = 128*1000 +300
#         bbb = 609*1000 -400
#         df = df.iloc[aaa:bbb]
# =============================================================================
        # print(f'\n\n~~~df.isnull().sum()~~~\n', df.isnull().sum())
        print("len df = ", len(df))

        # Fragmenty, które zostawiamy
        part1 = df.loc[dates['start']:dates['out1_start']]
        part2 = df.loc[dates['out1_end']:dates['out2_start']]
        part3 = df.loc[dates['out2_end']:dates['end']]
        
        # Sklejenie bez resetowania indeksu
        df = pd.concat([part1, part2, part3])
        # print(f'\n\n~~~df.isnull().sum()~~~\n', df.isnull().sum())
        print("len df = ", len(df))
        
        # quantities
        qty_train = part1.shape[0]
        qty_val = part2.shape[0]
        qty_test = part3.shape[0]
        
        return df, qty_train, qty_val, qty_test
        


    
    def split_df_qtyami(self, pds, qty_train, qty_val, qty_test):
        """Manually split the pandas dataframe"""
        
        # normalnie
        train_pds = pds.iloc[: qty_train]  # nie trzeba +1 bo pierwsze jest ZERO !!
        val_pds   = pds.iloc[qty_train : qty_train+qty_val]
        test_pds   = pds.iloc[-(qty_test) :]

        # najpierw val i test
# =============================================================================
#         val_pds = pds.iloc[: qty_val]  # nie trzeba +1 bo pierwsze jest ZERO !!
#         test_pds   = pds.iloc[qty_val : qty_val+qty_test]
#         train_pds  = pds.iloc[-(qty_train) :]
# =============================================================================

        return train_pds, val_pds, test_pds

























