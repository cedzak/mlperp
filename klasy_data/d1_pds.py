#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PdsSetup - Zrefaktoryzowana wersja z klasą konfiguracyjną
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
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
#
from klasy_data.d0_data import DataConfig
from klasy_jesien.ts_processor import TimeSeriesProcessor

class PdsSetup:
    """
    Przygotowuje dane pandas (skalowanie, podział train/val/test).
    UPROSZCZONE: Zamiast 10+ argumentów, przyjmuje tylko datacfg i parametry eksperymentu.
    """

    def __init__(self,
                 datacfg: DataConfig,
                 wybor_kols: str,
                 timestampplus: str,
                 sprkod: bool = False,
                 testsetatlast: bool = False):
        """
        Args:
            datacfg: Obiekt DataConfig z konfiguracją danych
            wybor_kols: Nazwa zestawu kolumn z kols_map
            timestampplus: Timestamp dla nazw plików i info co robię
            sprkod: Czy tryb testowy (małe dane)
            testsetatlast: Czy test set na końcu
        """
        self.wybor_kols = wybor_kols
        self.timestampplus = timestampplus
        self.sprkod = sprkod
        self.testsetatlast = testsetatlast #### tutaj jest to tylko po to żeby dobrze nazwać podkatalog

        # NIE: Rozpakuj konfigurację wszystkich używanych tutaj danych -- lepiej wiedzieć skąd się wzięły

        self.datacfg = datacfg
        self.kol_flagA = self.datacfg.kol_flagA
        self.ts_processor = TimeSeriesProcessor()
        self.subdirectory = (f"{self.wybor_kols or 'all_kols'}"
                            f"{'__SPR' if self.sprkod else ''}"
                            f"{'__testset' if self.testsetatlast else '__valset'}"
                            )

        # Initialize attributes
        # to nie jest potrzebne !
# =============================================================================
#         self.df_for_keras = None
#         self.X_train = self.X_val = self.X_test = None
#         self.y_train = self.y_val = self.y_test = None
#         self.qty_train = self.qty_val = self.qty_test = None
# =============================================================================
        #
        self._setup()

        
        
    
    def _setup(self):
        """Prepare data and store for reuse."""

        logger.info("=" *80)
        logger.info("Inicjalizacja PdsSetup w PdsSetup")

        self.k1_path = self._get_k1_path()
        self.data_prep_file = self._get_k1_path() / f"data_prep_info__{self.timestampplus}.txt"
        
        (self.df_for_keras,
         self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test,
         self.pd_train_idx, self.pd_val_idx, self.pd_test_idx,
         self.qty_train, self.qty_val, self.qty_test,
         ) = self.main_main_prepare_scaled_pandas_sets()

        
        # Znajdź kolumnę z flagą reżimu
        self.get_lp_kolumny_flagi()        
        
        # Logowanie informacji
        self.write_setup_info_to_data_prep_file()

    def _get_k1_path(self):
        """Tworzy i zwraca ścieżkę do katalogu wyjściowego."""
        output_path = self.datacfg.mojerys_path / self.subdirectory / self.timestampplus
        output_path.mkdir(parents=True, exist_ok=True)
        print("UTWORZONO pds_setup_path")
        return output_path





    def get_lp_kolumny_flagi(self):
        """Zwraca indeks (liczba porządkowa) kolumny z flagą w X_for_keras."""
        self.lp_kolumny_flagi = (self.df_for_keras.drop("target", axis=1)
                                 .columns.get_loc(self.kol_flagA)) if self.kol_flagA else None
        return self.lp_kolumny_flagi










    def load_data(self):
        """Wczytuje dane z pliku CSV."""
        logger.info(f"Wczytuje dane z pliku CSV {self.datacfg.plik_dane}")
        return pd.read_csv(self.datacfg.plik_dane, index_col=0, date_format="%Y-%m-%d %H:%M:%S")
    
    def select_columns(self, df):
        """Wybiera kolumny na podstawie wybor_kols."""
        if self.wybor_kols:
            return df[self.datacfg.kols_map.get(self.wybor_kols, ValueError("Invalid wybor_kols value"))]
        return df

    def convert_types(self, df):
        """Konwertuje określone kolumny na float64. ################## nie musi być 32, right?
           2025.03.25 potrzebuje ufloacic kols int jesli chce zeby byly zeskalowane."""
        logger.debug(f"Sorted columns of df: {sorted(df.columns)}")
        for col in self.datacfg.intkols_to_scale:
            try:
                df[col] = df[col].astype("float64")
            except (KeyError, ValueError):
                logger.warning(f"Nie udało się ufloacić kolumny {col}, a byla taka kolumna?\n")
        return df
    
    
    
    
    

    def add_kol_target(self, df, df_virgin):
        """Przygotowuje kolumnę target i usuwa kol_target z df."""
        df['target'] = df[self.datacfg.kol_target]
        df_virgin['target'] = df_virgin[self.datacfg.kol_target]
        return df.drop(self.datacfg.kol_target, axis=1), df_virgin


















    def main1_prepare_df_and_df_virgin(self):
        """Przygotowuje dane: wczytuje, wybiera wiersze i kolumny, 
           ustala quantities, data types i target."""
        df = self.load_data()
        df_virgin = df.copy()
        print(f"\n\njestem w d1_pds, main1.0\nlen(df)={len(df)}, len(df.index) = {len(df.index)}")
        

        # wybor kols i int-to-float jesli dana kol ma byc skalowana
        df = self.select_columns(df)
        print(f"\njestem w d1_pds, main1.1, len(df)={len(df)}, len(df.index) = {len(df.index)}")
            
            
        ## nie rozumiem co robię w tej funkcji, ale na pewno majstruję przy liczbie wierszy, a więc 
        # to musi być pred ts_processor
        df, df_virgin = self.add_kol_target(df, df_virgin)
        print(f"\njestem w d1_pds, main1.2, len(df)={len(df)}, len(df.index) = {len(df.index)}")
        print(df.head(3).to_string())
        print(df.tail(3).to_string())
        for col in sorted(df.columns):
            print(col)
            

        # wybor rows
        if self.sprkod:
            df = df.iloc[::1000]
            # df = df[:1000]  ## dla mr już po tej poprzedniej operacji zostaje około 1000
            
            qty_train, qty_val, qty_test = self.ts_processor.get_quantities_from_proportions(
                            df.index, self.datacfg.frac_val, self.datacfg.frac_test)
        else:
            if self.datacfg.projekt_akronim=="mr":
                df, qty_train, qty_val, qty_test = self.ts_processor.select_rows_and_get_quantities(df, self.datacfg.dates)
            else:
                qty_train, qty_val, qty_test = self.ts_processor.get_quantities_from_proportions(
                                df.index, self.datacfg.frac_val, self.datacfg.frac_test)
            
       
        # int-to-float jesli dana kol ma byc skalowana
        df = self.convert_types(df)
        print(f"\njestem w d1_pds, main1.3, len(df)={len(df)}, len(df.index) = {len(df.index)}")

       
        return df, df_virgin, qty_train, qty_val, qty_test 



    def main2_scale_pandas_sets_in_place(self, train_set, val_set=None, test_set=None):
        """
        Funkcja skaluje numeryczne kolumny w podanych zbiorach danych (`train_set`, `val_set`, `test_set`) 
        przy uyciu `MinMaxScaler` (domylnie) lub `StandardScaler` (jeli podano `estymator="Standard"`).

        Argumenty:
        - train_set, wymagany
        - val_set
        - test_set
        - estymator: Typ skalowania; `"MinMax"` dla normalizacji (0-1) 
                     lub `"Standard"` dla standaryzacji (rednia 0, odchylenie 1).
        Uwagi:
        - Funkcja modyfikuje oryginalne DataFrame'y (przez `loc[:, cols]`), 
          co moe prowadziacdo nieoczekiwanych efektw, jeli nie sa one kopiami.
        - Funkcja skaluje tylko kolumny numeryczne, 
          ignorujac kolumny z innymi typami danych (np. tekstowe, kategoryczne).

        Zwracane wartoci:
        - `scaler`: Dopasowany obiekt skalera (do ewentualnego uycia w przyszoci).
        """
        
        train_set_float = train_set.select_dtypes(include=['float']) 
        # AHA!! TO DLATEGO MI KAZAŁO ROBIĆ float32 !! nie, nie rozumiem, float to alias dla float64...
        # train_set_float = train_set.select_dtypes(include=['float16', 'float32', 'float64'])

        cols=train_set_float.columns
        logger.info("^^^^^^UWAGA UWAGA, TYLKO KOLUMNY TYPU float ZOSTANA SCALED^^^^^")
        logger.info(cols)

        scaler =    (
                    StandardScaler()    if self.datacfg.sca_estymator == "Standard" 
                    else MaxAbsScaler() if self.datacfg.sca_estymator == "MaxAbs" 
                    else MinMaxScaler() if self.datacfg.sca_estymator == "MinMax" 
                    else None
                    )
        
        scaled = scaler.fit_transform(train_set[cols]).astype('float16')
        train_set.loc[:, cols] = scaled
        
        if val_set is not None:
            scaled = scaler.transform(val_set[cols]).astype('float16')
            val_set.loc[:, cols] = scaled
        
        if test_set is not None:
            scaled = scaler.transform(test_set[cols]).astype('float16')
            test_set.loc[:, cols] = scaled
        
        logger.info("^^^^^^SKALOWANIE POWIODLO SIE^^^^^")
        return scaler



    def main3_construct_df_for_keras(self,  X_train, X_val, X_test, 
                                            y_train, y_val, y_test,
                                            save=False):
        """
        Merges X and y datasets, adds a flag column indicating the dataset origin,
        and saves the combined DataFrame to a CSV file.

        Parameters:
        - X_train, X_val, X_test: Feature DataFrames
        - y_train, y_val, y_test: Target DataFrames
        - filename (str): The name of the CSV file to save the merged data
        """
        # v bez flag
        df_for_keras = pd.concat([pd.concat([X_train, y_train], axis=1), 
                                  pd.concat([X_val, y_val], axis=1), 
                                  pd.concat([X_test, y_test], axis=1)], axis=0)

        #### save csv
        if save==True:
            filename_full = self.get_k1_path() / 'df_for_keras_z_flagami.csv'
            #
            if not filename_full.exists():  # Sprawdzamy, czy plik NIE istnieje
            
                # v z flagami, żeby były w csv
                def add_flag(X, y, flag):
                    df = pd.concat([X, y], axis=1)
                    df["model_set"] = flag  # Add a flag column
                    return df
                # Merge datasets and add flags
                train_df = add_flag(X_train, y_train, 'train')
                val_df   = add_flag(X_val, y_val, 'val')
                test_df  = add_flag(X_test, y_test, 'test')
                # Concatenate all datasets
                df_for_keras_z_flagami = pd.concat([train_df, val_df, test_df], axis=0)
                df_for_keras_z_flagami.to_csv(filename_full)
            else:
                logger.info(f"Plik {filename_full} już istnieje, wiec pomijam zapis")


        return df_for_keras



    def main_main_prepare_scaled_pandas_sets(self):
        """
        Przygotowuje dane do treningu modelu, w tym podział i skalowanie.
        Returns:
            tuple: (df, X_train, X_val, X_test, y_train, y_val, y_test, scaler_X)
        """
        try:
            df, df_virgin, qty_train, qty_val, qty_test = self.main1_prepare_df_and_df_virgin()
            print(f"\n\njestem w d1_pds main_main, len(df)={len(df)}, len(df.index) = {len(df.index)}")


            # Podział danych na train-val-test
            train_pds, val_pds, test_pds = self.ts_processor.split_df_qtyami(df, qty_train, qty_val, qty_test)


            with open(self.data_prep_file, "a") as file:
                file.write(
                    f"Prepared data: {len(df)} rows, z czego train={qty_train}, val={qty_val}, test={qty_test}\n"
                    f"Sanity check: {len(train_pds)}, {len(val_pds)}, {len(test_pds)}\n"
                    f"Sanity check: "
                    f"len(train_pds) == qty_train: {len(train_pds) == qty_train}, "
                    f"len(val_pds) == qty_val: {len(val_pds) == qty_val}, "
                    f"len(test_pds) == qty_test: {len(test_pds) == qty_test}\n")
            

            # Podział pandas setów train-val-test na X i y
            X_train, y_train   = train_pds.drop('target', axis=1), train_pds['target'].to_frame()
            X_val,   y_val     = val_pds.drop('target', axis=1),   val_pds['target'].to_frame()
            X_test,  y_test    = test_pds.drop('target', axis=1),  test_pds['target'].to_frame()


            # Skalowanie danych ---- musi być zrobione na danych rozdzielonych na train i pozostałe
            if self.datacfg.scale:
                scaler_X = self.main2_scale_pandas_sets_in_place(X_train, X_val, X_test)
            else:
                scaler_X = None
            
           
            # Konstruowanie df_for_keras ---- to nie jest taka sama dataframe jak df, bo ta jest zeskalowana
            df_for_keras = self.main3_construct_df_for_keras( X_train, X_val, X_test, 
                                                              y_train, y_val, y_test )
            print(len(df), len(df_for_keras))
            logger.info(f"jest {len(df)} i {len(df_for_keras)}, "
                        f"len(df) i len(df_for_keras) powinny być sobie równe\n")
            if len(df) != len(df_for_keras):
                error_msg = f"MISSSSSSS-match in lengths: df and df_for_keras must have the same number of rows, a jest {len(df)}, {len(df_for_keras)}."
                logger.critical(error_msg)  # Logujemy krytyczny błąd
                raise ValueError(error_msg)  # Rzucamy wyjątek żeby ZATRZYMAĆ


            # round jesli to jest test kodu, albo nawet jak nie jest #### WROC -- Claude powiedział że tak będzie ok
            # if self.sprkod:
            # print("\n\nwszystkie dfy do returna round do trzech miejsc po przecinku i zrób z nich float16") # CHYBA NIE MA CO
            for _frame in (df_for_keras,   X_train,   X_val,   X_test,
                                           y_train,   y_val,   y_test):
                for col in _frame.columns:
                    if _frame[col].dtype.kind == 'f':  # 'f' oznacza floating point
                        _frame[col] = _frame[col].round(3)
                print("po round:\n", _frame.head(2))



            for col in df_for_keras.columns:
                print(f"\n\n{col}")
                
                n_zeros=(df_for_keras[col]==0).sum()
                n_nans=df_for_keras[col].isna().sum()
                
                print(f"Liczba zer: {n_zeros}")
                print(f"Liczba NaN: {n_nans}")



            return (df_for_keras,   X_train,   X_val,   X_test, 
                                    y_train,   y_val,   y_test,
                                    y_train.index, y_val.index, y_test.index,
                                    qty_train, qty_val, qty_test 
                    )



        except Exception as e:
            error_msg = f"Błąd w main_main_prepare_scaled_pandas_sets: {str(e)}"
            # Logowanie na różnych poziomach
            logger.error(error_msg)  # standardowy poziom błędu
            logger.exception(e)  # dodaje pełny traceback do logów
            # Dla bardzo poważnych błędów
            logger.critical(f"KRYTYCZNY BŁĄD: {error_msg}")
            
            raise # raise na końcu except oznacza: "Zalogowałem błąd, ale nadal uważam go za na tyle poważny, 
                                                    # że program nie powinien kontynuować normalnego wykonania."



    def write_setup_info_to_data_prep_file(self):
        """Zapisuje informacje diagnostyczne do pliku."""

        lines = [
                "=" * 80,
                "Pds - Informacje podstawowe \ndatacfg.params_data_to_train:"
                ]

        for k, v in self.datacfg.params_data_to_train.items():
            lines.append(f"  {k:<15} {v}")
        lines.append(f"len cols: {len(self.df_for_keras.columns)} \nsorted columns:")
        for col in sorted(self.df_for_keras.columns):
            lines.append(f"  {col}")
            
            
        with open(self.data_prep_file, "a") as file:
            file.write("\n".join(lines))
            file.write(
                f"\n\n{'=' *80}\n"
                # f"TARGET KOLUMN: {self.datacfg.kol_target}\n"
                f"D A T A  X : : : : : : : : : : : : : : : : : : : :\n"
                f"First index of train samples: {self.X_train.index[0]}\n"
                f"First index of val samples:   {self.X_val.index[0]}\n"
                f"First index of test samples:  {self.X_test.index[0]}\n\n"
                #
                f"shapes of self.X_train, self.X_val, self.X_test:\n"
                f"shape of X_train:     {self.X_train.shape}\n"
                f"shape of X_val:       {self.X_val.shape}\n"
                f"shape of X_test:      {self.X_test.shape}\n"
                f"X_train.describe()\n"
                f"{self.X_train.describe().to_string()}\n\n"
                #
                f"\nheads and tails of X_train, X_val, X_test:\n"
                f"~~~~~~head and tail of X_train:\n"
                f"{self.X_train.head(2).to_string()}\n{self.X_train.tail(2).to_string()}\n"
                f"~~~~~~head and tail of X_val:\n"
                f"{self.X_val.head(2).to_string()}\n{self.X_val.tail(2).to_string()}\n"
                f"~~~~~~head and tail of X_test:\n"
                f"{self.X_test.head(2).to_string()}\n{self.X_test.tail(2).to_string()}\n\n"
                #
                f"{'=' *80}\n"
                f"D A T A  y : : : : : : : : : : : : : : : : : : : :\n"
                f"shapes of y_train, y_val, y_test:\n"
                f"shape of y_train:     {self.y_train.shape}\n"
                f"shape of y_val:       {self.y_val.shape}\n"
                f"shape of y_test:      {self.y_test.shape}\n\n"
                #
                f"heads and tails of y_train, y_val, y_test:\n"
                f"~~~~~~y_train:\n"
                f"{self.y_train.head(2).to_string()}\n{self.y_train.tail(2).to_string()}\n"
                f"~~~~~~y_val:\n"
                f"{self.y_val.head(2).to_string()}\n{self.y_val.tail(2).to_string()}\n"
                f"~~~~~~y_test:\n"
                f"{self.y_test.head(2).to_string()}\n{self.y_test.tail(2).to_string()}\n"
                f"{'=' *80}\n"
                #
                f"Kolumna flagA o nazwie '{self.kol_flagA}' ma indeks {self.lp_kolumny_flagi}"
                f"~~~~~~~~~~~~~~Zakonczyl prepare_pandas_datasests~~~~~~~~~~~~~~\n\n")

            


    def plot_sensors(self, metoda):
        """
        """
        _, df_virgin = self.main1_prepare_df_and_df_virgin()
        
        logger.info("halo halo, plot sensors to robi sie na df_virgin !!! vir-gin !!!")

        # Podział danych na train-val-test
        train_pds, val_pds, test_pds = self.ts_processor.split_df_qtyami(df_virgin)

        dzien = 60*24; tydzien = dzien*7; miesiac = tydzien*4
        chunk_size = 10000
        
        from utils.plots import plot_sensors_only
        for df, set_type in zip( [train_pds, val_pds, test_pds], ["TRAIN SET", "VAL SET", "TEST SET"] ):
            # df = df.reset_index(drop=True)
            plot_sensors_only(df, chunk_size, self.get_k1_path(), add_info=set_type) # len(df)+1 zamiast chunk_size to jeden plot na set




    
    @staticmethod
    def _plot_sensors_only(df, chunk_size, mojerys_path, num_chunks=None, scaled=False, add_info=""):
        """zostawiam mojerys_path zeby byla mozliwosc zapisac obrazek w podkatalogu tego katalogu"""
        
        # Parameters
        chunk_size = chunk_size
        if num_chunks is None:
            num_chunks = (len(df) // chunk_size) + (1 if len(df) % chunk_size != 0 else 0)
        print(f"num_chunks = {num_chunks}")
        
        # Create plots
        for i in range(num_chunks):
            # Determine the slice of the DataFrame to plot
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            df_chunk = df.iloc[ start_idx : end_idx ]
        
            print(start_idx, end_idx, df_chunk.index[0])
        
        
            # Create plot
            plt.figure(figsize=(12, 8)) ########### DO WORDA: 13, 10 pod komputer: 35, 20
    # =============================================================================
    #         ax = plt.gca()  # Get current axes
    #         ax.set_facecolor('white')  # Set axes background to white
    # =============================================================================
            
            
            plt.plot(df_chunk.index, df_chunk['power'], label='power [MW]', color="black", alpha=0.7, linewidth=2)
            plt.plot(df_chunk.index, df_chunk['power_SP'], label='power_SP [MW]', color="maroon", alpha=0.5, linewidth=2)
            
            markersize_duzy_plot=.9
            ms_do_worda=2
     
            plt.plot(df_chunk.index, df_chunk["t_IP_steam"], label="t steam inlet [°C]",  color="blue", alpha=0.3,  linewidth=1, marker='o', markersize=ms_do_worda)
            plt.plot(df_chunk.index, df_chunk['t_blade'],    label='t blade [°C]',        color='darkorange',  linewidth=1, marker='o', markersize=ms_do_worda)
            plt.plot(df_chunk.index, df_chunk['t_LP_steam'], label='t steam outlet [°C]', color='c',    linewidth=1, marker='o', markersize=ms_do_worda)        
            plt.plot(df_chunk.index, df_chunk['t_casing'],   label='t casing [°C]',       color='navy', alpha=0.5,  linewidth=1, marker='o', markersize=ms_do_worda)   
            # wygladzanie gorsze
    # =============================================================================
    #         plt.plot(df_chunk.index, df_chunk['logdiff_t5smth_blade_2iron600']*10e5, label='logdiff_t5smth_blade_2iron600', color='navy', linewidth=1) 
    #         plt.plot(df_chunk.index, df_chunk['logdiff_t5smth_IP_steam_2iron600']*10e5, label='logdiff_t5smth_IP_steam_2iron600', color='c', linewidth=1) 
    #         plt.plot(df_chunk.index, df_chunk['logdiff_t5smth_LP_steam_2iron600']*10e5, label='logdiff_t5smth_LP_steam_2iron600', color='royalblue', linewidth=1) 
    # =============================================================================
            # wygladzanie lepsze
    # =============================================================================
    #         plt.plot(df_chunk.index, df_chunk['logdiff_t5smth_blade_2ewm600']*10e5, label='logdiff_t5smth_blade_2ewm600', color='navy', linewidth=2, linestyle='--') 
    #         plt.plot(df_chunk.index, df_chunk['logdiff_t5smth_IP_steam_2ewm600']*10e5, label='logdiff_t5smth_IP_steam_2ewm600', color='c', linewidth=2, linestyle='--') 
    #         plt.plot(df_chunk.index, df_chunk['logdiff_t5smth_LP_steam_2ewm600']*10e5, label='logdiff_t5smth_LP_steam_2ewm600', color='royalblue', linewidth=2, linestyle='--') 
    #         
    # =============================================================================
            
            
            
            plt.plot(df_chunk.index, df_chunk['RPM_smth']/10, label='RPM/10', color='magenta', linewidth=1, marker='o', markersize=ms_do_worda) #, alpha=0.6, )        
            # plt.plot(df_chunk.index, df_chunk['RPM_spp']/10, label='RPM_spp/10', color='purple', linewidth=5, alpha=0.4, )        
            # plt.plot(df_chunk.index, df_chunk['RPM_diff_binned']/10-10, label='RPM_diff_binned', color='purple', linewidth=1)
            
            
            # plt.plot(df_chunk.index, df_chunk["dupa"],                 label="dupa", color="yellow", linewidth=4, linestyle=':') #, alpha=0.6)
    # =============================================================================
    #         plt.plot(df_chunk.index, df_chunk["t_blade_go_down"]*100,  label="t_blade_go_down", color="green", linewidth=2)       
    #         plt.plot(df_chunk.index, df_chunk['shut_minutes']/10-195,  label='shut_minutes', color='green', linewidth=5, alpha=0.4, linestyle='--') 
    # =============================================================================
    
    
    
    # =============================================================================
    #         plt.plot(df_chunk.index, df_chunk['shutdown']*600-50,     label='shutdown',    color='green', linewidth=1.5) 
    #         plt.plot(df_chunk.index, df_chunk["startup_180"]*600-25,  label="startup_180", color='red', linewidth=1.0) 
    # =============================================================================
    
    
    
    
    
            # bar plots
    # =============================================================================
    #         GRUBOSC = 1
    #         
    #         plt.bar(df_chunk.index, df_chunk["startup_start"]*600, label="startup start", 
    #                 color='black', alpha=0.5, width=0.05*GRUBOSC, align='center') 
    # 
    #         plt.bar(df_chunk.index, df_chunk["RPM_diff_binned"], label="diff of RPM", 
    #                 color='purple', alpha=0.3, width=0.005*GRUBOSC, align='center') # 0.001
    # =============================================================================
            
    
    
    
            # Setup plot aesthetics
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.legend() # zamula
            
            plt.xticks(rotation=45)
    
            # plt.grid(which='both')
            plt.grid(which='both', linestyle='-', linewidth=.5) # color='r'
    
            # Set the y-axis limits
            plt.ylim(-50, 600)
    # =============================================================================
    #         plt.ylim(-400, 1000)
    #         if scaled==True:
    #             plt.ylim(0, 1)            
    # =============================================================================
                
            #*<
            # title = f'Sensor Data: {add_info} Chunk {i+1}' #'from {df_chunk.index[0]} to {df_chunk.index[-1]}'
            
            title = f'Sensor Data Chunk {i+1}' 
            title = f'Sensor Data' 
            
            plt.title(title); now = str(datetime.now())[:16].replace(":", "-")
            path = mojerys_path / f"{now}__{add_info}__{title}.png" 
            plt.tight_layout(); plt.savefig(path, dpi=100)
            plt.close() # Close the plot to free up memory
            #*>
        
        print("All plots have been generated and saved as separate files.")
        
        
        
        
    
    @staticmethod
    def _plot_cala_df(df, chunk_size, mojerys_path, info="", num_chunks=None):
        """zostawiam mojerys_path zeby byla mozliwosc zapisac obrazek w podkatalogu tego katalogu"""
        
        # Parameters
        chunk_size = chunk_size
        if num_chunks is None:
            num_chunks = (len(df) // chunk_size) + (1 if len(df) % chunk_size != 0 else 0)
        print(f"num_chunks = {num_chunks}")
        
        # Create plots
        for i in range(num_chunks):
            # Determine the slice of the DataFrame to plot
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            df_chunk = df.iloc[ start_idx : end_idx ]
        
            print(start_idx, end_idx, df_chunk.index[0])
        
        
                
            plt.figure(figsize=(20, 12))
            
            for col in df_chunk.columns:
                plt.plot(df_chunk.index, df_chunk[col], label=col, alpha=0.7)
            
            plt.xlabel('Timestep', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.title(f'Wszystkie kolumny {info} -- chunk {i} z {num_chunks}', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(mojerys_path/f'{info}_single_plot_chunk_{i}_z_{num_chunks}.png', dpi=150, bbox_inches='tight')
            # plt.show()
            plt.close() # Close the plot to free up memory
            
            print("Wykres zapisany jako 'all_columns_single_plot.png'")
        
        
        
        
        