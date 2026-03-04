#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tuner Results Analyzer - narzędzie do analizy wyników tuningu
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class TunerResultsAnalyzer:
    """
    Klasa do analizy wyników hyperparameter tuningu.
    
    Funkcje:
    - Porównanie różnych strategii tuningu
    - Analiza konwergencji
    - Wizualizacja przestrzeni hiperparametrów
    - Ranking najlepszych konfiguracji
    """
    
    def __init__(self, tuner_dir):
        """
        Args:
            tuner_dir: Ścieżka do katalogu z wynikami tuningu
        """
        self.tuner_dir = Path(tuner_dir)
        
        if not self.tuner_dir.exists():
            raise ValueError(f"Katalog nie istnieje: {tuner_dir}")
        
        # Znajdź pliki z wynikami
        self.trials_file = self._find_file("trials_results_*.csv")
        self.best_hps_file = self._find_file("best_hyperparameters_*.json")
        
        # Wczytaj dane
        self.df_trials = pd.read_csv(self.trials_file) if self.trials_file else None
        
        with open(self.best_hps_file, 'r') as f:
            self.best_hps = json.load(f)
        
        print(f"Wczytano {len(self.df_trials)} trials z {self.tuner_dir}")
    
    
    def _find_file(self, pattern):
        """Znajduje plik pasujący do wzorca"""
        files = list(self.tuner_dir.glob(pattern))
        return files[0] if files else None
    
    
    def print_summary(self):
        """Wyświetla podsumowanie wyników"""
        print("\n" + "="*70)
        print("PODSUMOWANIE WYNIKÓW TUNINGU")
        print("="*70)
        
        print(f"\nLiczba trials: {len(self.df_trials)}")
        print(f"\nStatystyki score (RMSE):")
        print(f"  Najlepszy:  {self.df_trials['score'].min():.4f}")
        print(f"  Najgorszy:  {self.df_trials['score'].max():.4f}")
        print(f"  Średni:     {self.df_trials['score'].mean():.4f}")
        print(f"  Mediana:    {self.df_trials['score'].median():.4f}")
        print(f"  Std:        {self.df_trials['score'].std():.4f}")
        
        print(f"\nNajlepsze hiperparametry:")
        for key, value in self.best_hps.items():
            print(f"  {key:<25} {value}")
        
        print("\n" + "="*70)
    
    
    def plot_convergence(self, save=True):
        """Wykres konwergencji tuningu"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Subplot 1: Wszystkie trials
        axes[0].scatter(self.df_trials['trial_id'], self.df_trials['score'], 
                       alpha=0.6, s=50, color='skyblue', edgecolor='navy')
        axes[0].plot(self.df_trials['trial_id'], self.df_trials['score'], 
                    alpha=0.3, color='gray', linestyle='--')
        axes[0].set_xlabel('Trial ID')
        axes[0].set_ylabel('Validation RMSE')
        axes[0].set_title('All Trials')
        axes[0].grid(True, alpha=0.3)
        
        # Subplot 2: Best score progression
        best_scores = self.df_trials['score'].cummin()
        axes[1].plot(self.df_trials['trial_id'], best_scores, 
                    marker='o', linewidth=2, markersize=4, 
                    color='green', alpha=0.8)
        axes[1].fill_between(self.df_trials['trial_id'], 
                            best_scores, 
                            self.df_trials['score'].max(),
                            alpha=0.2, color='green')
        axes[1].set_xlabel('Trial ID')
        axes[1].set_ylabel('Best RMSE So Far')
        axes[1].set_title('Convergence')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plot_file = self.tuner_dir / "analysis_convergence.png"
            plt.savefig(plot_file, dpi=150)
            print(f"Zapisano wykres konwergencji: {plot_file}")
        else:
            plt.show()
        
        plt.close()
    
    
    def plot_hyperparameter_distributions(self, save=True):
        """Wykresy rozkładów hiperparametrów"""
        # Wybierz tylko kolumny z hiperparametrami (nie trial_id, score)
        hp_cols = [col for col in self.df_trials.columns 
                   if col not in ['trial_id', 'score']]
        
        n_cols = len(hp_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(hp_cols):
            ax = axes[idx]
            
            # Sprawdź typ danych
            if self.df_trials[col].dtype in ['object', 'string']:
                # Kategoryczne
                value_counts = self.df_trials[col].value_counts()
                value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='navy')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
            else:
                # Numeryczne
                ax.hist(self.df_trials[col], bins=20, 
                       color='skyblue', edgecolor='navy', alpha=0.7)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.axvline(self.best_hps.get(col, 0), 
                          color='red', linestyle='--', linewidth=2, 
                          label='Best')
                ax.legend()
            
            ax.set_title(f'Distribution: {col}')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Ukryj puste subploty
        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save:
            plot_file = self.tuner_dir / "analysis_distributions.png"
            plt.savefig(plot_file, dpi=150)
            print(f"Zapisano wykres rozkładów: {plot_file}")
        else:
            plt.show()
        
        plt.close()
    
    
    def plot_hyperparameter_vs_score(self, save=True):
        """Wykresy: każdy hiperparametr vs score"""
        hp_cols = [col for col in self.df_trials.columns 
                   if col not in ['trial_id', 'score']]
        
        n_cols = len(hp_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(hp_cols):
            ax = axes[idx]
            
            if self.df_trials[col].dtype in ['object', 'string']:
                # Kategoryczne - boxplot
                self.df_trials.boxplot(column='score', by=col, ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel('Score (RMSE)')
                ax.set_title(f'{col} vs Score')
                plt.sca(ax)
                plt.xticks(rotation=45)
            else:
                # Numeryczne - scatter
                ax.scatter(self.df_trials[col], self.df_trials['score'], 
                          alpha=0.6, s=50, color='skyblue', edgecolor='navy')
                
                # Dodaj best point
                best_value = self.best_hps.get(col, None)
                if best_value is not None:
                    best_score = self.df_trials['score'].min()
                    ax.scatter([best_value], [best_score], 
                              s=200, color='red', marker='*', 
                              edgecolor='darkred', linewidth=2,
                              label='Best', zorder=5)
                    ax.legend()
                
                ax.set_xlabel(col)
                ax.set_ylabel('Score (RMSE)')
                ax.set_title(f'{col} vs Score')
                ax.grid(True, alpha=0.3)
        
        # Ukryj puste subploty
        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save:
            plot_file = self.tuner_dir / "analysis_hp_vs_score.png"
            plt.savefig(plot_file, dpi=150)
            print(f"Zapisano wykres HP vs Score: {plot_file}")
        else:
            plt.show()
        
        plt.close()
    
    
    def get_top_trials(self, n=10):
        """Zwraca top N najlepszych trials"""
        top_trials = self.df_trials.nsmallest(n, 'score')
        
        print(f"\n{'='*70}")
        print(f"TOP {n} TRIALS:")
        print('='*70)
        
        for idx, row in top_trials.iterrows():
            print(f"\nTrial {row['trial_id']}: RMSE = {row['score']:.4f}")
            for col in self.df_trials.columns:
                if col not in ['trial_id', 'score']:
                    print(f"  {col:<25} {row[col]}")
        
        print('='*70)
        
        return top_trials
    
    
    def compare_with_baseline(self, baseline_rmse):
        """Porównaj wyniki tuningu z baseline"""
        best_rmse = self.df_trials['score'].min()
        improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
        
        print(f"\n{'='*70}")
        print("PORÓWNANIE Z BASELINE:")
        print('='*70)
        print(f"Baseline RMSE:     {baseline_rmse:.4f}")
        print(f"Best Tuned RMSE:   {best_rmse:.4f}")
        print(f"Improvement:       {improvement:.2f}%")
        
        if improvement > 0:
            print(f"\n✓ Tuning poprawił wyniki o {improvement:.2f}%")
        else:
            print(f"\n✗ Tuning nie poprawił wyników (pogorszenie: {abs(improvement):.2f}%)")
        
        print('='*70)
    
    
    def generate_full_report(self):
        """Generuje kompletny raport analizy"""
        print("\n\n" + "="*70)
        print("GENEROWANIE PEŁNEGO RAPORTU")
        print("="*70 + "\n")
        
        # Podsumowanie
        self.print_summary()
        
        # Top trials
        self.get_top_trials(n=5)
        
        # Wykresy
        self.plot_convergence(save=True)
        self.plot_hyperparameter_distributions(save=True)
        self.plot_hyperparameter_vs_score(save=True)
        
        print("\n" + "="*70)
        print("RAPORT WYGENEROWANY!")
        print(f"Pliki zapisane w: {self.tuner_dir}")
        print("="*70 + "\n")


# =============================================================================
# PRZYKŁAD UŻYCIA
# =============================================================================
if __name__ == "__main__":
    """
    Przykład analizy wyników tuningu
    """
    import sys
    
    if len(sys.argv) > 1:
        tuner_dir = sys.argv[1]
    else:
        # Domyślnie: znajdź najnowszy katalog tunera
        base_dir = Path("../mojerys/gda")
        tuner_dirs = sorted(base_dir.glob("tuner_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if tuner_dirs:
            tuner_dir = tuner_dirs[0]
            print(f"Znaleziono najnowszy katalog tunera: {tuner_dir}")
        else:
            print("Nie znaleziono katalogów z wynikami tuningu!")
            print("Użycie: python analyze_tuner_results.py <ścieżka_do_katalogu_tunera>")
            sys.exit(1)
    
    # Stwórz analyzer
    analyzer = TunerResultsAnalyzer(tuner_dir)
    
    # Wygeneruj pełny raport
    analyzer.generate_full_report()
    
    # Opcjonalnie: porównaj z baseline
    # analyzer.compare_with_baseline(baseline_rmse=15.5)
