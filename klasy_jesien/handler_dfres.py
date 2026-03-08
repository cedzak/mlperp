#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results Handler - zarządza wynikami i zapisem
"""
import os; import logging
from pathlib import Path
import numpy as np; import pandas as pd

logger = logging.getLogger(__name__)
#
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             r2_score, 
                             mean_absolute_percentage_error)



class DfResHandler:
    """
    Klasa do zarządzania wynikami eksperymentów:
    - Tworzenie DataFrame'ów z wynikami
    - Zapis do CSV
    - Generowanie podsumowań
    """
    
    def __init__(self, indices, actuals, predictions_dct,
                 kolor_actual, kolor_pred, kolor_res,
                 sciezka_play,
                 timestampplus):
        """
        Args:
            sciezka_play: Ścieżka do katalogu wyjściowego (str lub Path)
            timestampplus: str żeby zidentyfikować jaki model i jakie dane
            prefix: str - prefiks (np. 'val_', 'test_')
        """
        self.indices = indices
        self.actuals = actuals
        self.predictions_dct = predictions_dct
        
        self.kolor_actual, self.kolor_pred, self.kolor_res = kolor_actual, kolor_pred, kolor_res
        
        self.results_path = Path(sciezka_play).parent # oczko wyżej !!! # najprostsze !!! bo (Path() akceptuje string i Path) 
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n\njestem w dfres_handler, results_path to: {self.results_path}")
    
        self.timestampplus = timestampplus
       

    def calculate_metrics_from_one_predictions(self, predictions):
        """
        Oblicza wszystkie standardowe metryki.

        Returns:
            dict: Słownik z metrykami
        """
        mae  = mean_absolute_error(self.actuals, predictions)
        mse  = mean_squared_error(self.actuals, predictions)
        rmse = np.sqrt(mse)
        r2   = r2_score(self.actuals, predictions)
        mape = mean_absolute_percentage_error(self.actuals, predictions) * 100
        
        metrics_dict = {
            'mae':  round(mae, 3),
            'mse':  round(mse, 3),
            'rmse': round(rmse, 3),
            'r2':   round(r2, 3),
            'mape': round(mape, 3),
        }

        return metrics_dict
    
    
    
    def calculate_metrics_dicts_list_from_all_runs_predictions(self):
        
        all_runs_metrics_dicts_list = []
        
        for run_idx, predictions in self.predictions_dct.items():
            metrics_dict = self.calculate_metrics_from_one_predictions(
                                                                predictions)
            # Dodanie do listy wyników
            all_runs_metrics_dicts_list.append(metrics_dict)
            
        return all_runs_metrics_dicts_list  
    
    
        

    @staticmethod
    def format_metrics(metrics_dict):
        """
        Formatuje metryki do czytelnego stringa.
        
        Args:
            metrics: dict z metrykami
        Returns:
            str: Sformatowane metryki
        """
        lineslist = []
        lineslist.append(f"\n\n\nMAE:  {metrics_dict['mae']:.2f}")
        lineslist.append(f"MSE:  {metrics_dict['mse']:.2f}")
        lineslist.append(f"RMSE: {metrics_dict['rmse']:.2f}")
        lineslist.append(f"R2:   {metrics_dict['r2']:.3f}")
        lineslist.append(f"MAPE: {metrics_dict['mape']:.2f}%")
        
        return "\n".join(lineslist)
    
   
    
   
    
    
    def _stworz_best_dfres(self, predictions, save=True):
        """
        tworzy dfres   z użyciem predictions z najlepszego runu Mono / najlepszych runów AB
        to jest chyba tylko potrzebne do plot_all_in_one... no chyba że chcę po coś zapisać
        """
        # DEBUGGING
        print(f"predictions.shape: {predictions.shape}")
        print(f"self.actuals.shape: {self.actuals.shape}")
        print(f"self.indices length: {len(self.indices)}")
        

        # Uniwersalne rozwiązanie - działa dla DataFrame, Series i numpy array
        actuals_1d = np.asarray(self.actuals).flatten()
        predictions_1d = np.asarray(predictions).flatten()
        
        self.best_dfres = pd.DataFrame(
            {'Actual': actuals_1d,
             'Predicted': predictions_1d}, 
            index=self.indices
        )


        self.best_dfres['Residuals'] = self.best_dfres['Actual'] - self.best_dfres['Predicted']

        if save:
            self.best_dfres.to_csv(self.results_path / f"dfres  __{self.timestampplus}.csv") 
            #### nauka: UKOŚNIK !!! nie można dodawać Path + string
            print("--Co żeś zrobił? \n--Zapisałem dfres  ")
    
        return self.best_dfres
                

    
    
    def plot_all_in_one(self, predictions, lililimit=40, bins=11):   
        """
        Combines four plots into a single 2x2 grid figure with custom widths:
        - Top-left (70-75%): Actual vs Predicted vs Residuals
        - Top-right (25-30%): Histograms of Actual and Predicted
        - Bottom-left (70-75%): Residuals vs Target (Scatter)
        - Bottom-right (25-30%): Histogram of Residuals
        
        Args:
            dfres  : pandas.DataFrame z wynikami
            model_name: str
            dataset_name: str
            
        Returns:
            bool: True jeśli sukces, False jeśli błąd
        """
        
        # print("\n\n\n??????????????????????? czy ja w ogole doszłam do plot_all_in_one ??")
        # print('predictions:')
        # print(predictions)
        
        dfres   = self._stworz_best_dfres(predictions)
        
        figure_name = self.timestampplus
        kolor_actual, kolor_pred, kolor_res = self.kolor_actual, self.kolor_pred, self.kolor_res
            
        try:
            import matplotlib.pyplot as plt
            plt.style.use("seaborn-v0_8-poster")
            import seaborn as sns
            #
            import matplotlib.dates as mdates
            import matplotlib.gridspec as gridspec

            INDEXX=dfres.index
            print(f"INDEXX type: {type(INDEXX)}")
            print(f"INDEXX dtype: {INDEXX.dtype if hasattr(INDEXX, 'dtype') else 'No dtype'}")
            print(f"Is datetime64: {pd.api.types.is_datetime64_any_dtype(INDEXX)}")

            ####### 2025.05.11 mam problem z indeksem !!! robię numeryczny
            # INDEXX = range(1,len(dfres  )+1)
        # =============================================================================
        #     if not pd.api.types.is_datetime64_any_dtype(INDEXX):
        #         try:
        #             INDEXX = pd.to_datetime(INDEXX)
        #             print("Converted INDEXX to DatetimeIndex")
        #         except Exception as e:
        #             print(f"Failed to convert INDEXX to datetime: {e}")
        # =============================================================================

            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(INDEXX, dfres  ['Actual'], color=kolor_actual, linewidth=0.2, marker='o', markersize=3, alpha=0.1, label='Actual')
            ax1.plot(INDEXX, dfres  ['Predicted'], color=kolor_pred, linewidth=0.2, marker='o', markersize=1, alpha=0.3, label='Predicted')
            ax1.bar(INDEXX, dfres  ['Residuals'], color=kolor_res, alpha=0.8, label='Residuals', width=0.01, edgecolor=kolor_res)
            residuals_mean = dfres  ['Residuals'].mean()
            residuals_std = dfres  ['Residuals'].std()
            ax1.fill_between(INDEXX, residuals_mean - residuals_std, residuals_mean + residuals_std, color=kolor_res, alpha=0.2, label='±1 std')
            ax1.set_xlim(INDEXX[0], INDEXX[-1])
            ax1.legend(loc='center left') #        'upper left')

            if pd.api.types.is_datetime64_any_dtype(INDEXX):
                # Pokazuj tylko pierwszy dzień każdego miesiąca
                ax1.xaxis.set_major_locator(mdates.MonthLocator())  # ← ZMIANA!
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax1.tick_params(axis='x', rotation=45)
                print("Applied datetime formatting to Plot 1")
            else:
                print("INDEXX is not datetime; skipping datetime formatting")

            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_xlabel('Time' if pd.api.types.is_datetime64_any_dtype(INDEXX) else 'Index')
            ax1.set_ylabel('Actual, Predicted, Residuals')
            ax1.set_title('Actual vs Predicted vs Residuals')

            ax4 = fig.add_subplot(gs[0, 1])
            ax4.hist(dfres  ['Actual'], bins=bins, color=kolor_actual, alpha=0.5, edgecolor='grey', label="Actual")
            ax4.hist(dfres  ['Predicted'], bins=bins, color=kolor_pred, alpha=0.4, edgecolor='grey', label="Predicted")
            ax4.set_xlabel("Values")
            ax4.set_ylabel("Frequency")
            ax4.legend()
            ax4.grid(axis='y', linestyle='--', linewidth=0.5)
            ax4.set_title("Histograms of Actual and Predicted")

            ax2 = fig.add_subplot(gs[1, 0])
            sns.scatterplot(x=dfres  ["Actual"], y=dfres  ["Residuals"], alpha=0.2, color=kolor_res, ax=ax2)
            ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
            ax2.set_xlabel("Actual Values (Target)")
            ax2.set_ylabel("Residuals (Actual - Predicted)")
            ax2.set_title("Residuals vs Target")  ##############
            ax2.set_ylim(-lililimit,lililimit)
            ax2.grid(True, linestyle='--', alpha=0.7)

            ax3 = fig.add_subplot(gs[1, 1])
            counts, bin_edges, _ = ax3.hist(dfres  ['Residuals'], bins=bins, color=kolor_res, edgecolor='grey', alpha=0.6)
            ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Frequency')
            ax3.set_title(f"Histogram of Residuals") ##############
            ax3.set_xlim(-lililimit,lililimit)
            ax3.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            full_output_path = self.results_path / f"plot_all_in_one__{figure_name}.png"
            plt.savefig(full_output_path, dpi=200)
            plt.close()
            print("paths:")
            print(self.results_path)
            print(full_output_path)
            
            logger.info(f"✓ Wygenerowano wykresy dla {figure_name}")
            return True
            
        except Exception as e:
            logger.warning(f"Nie udało się wygenerować wykresów: {e}")
            return False
