#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tak, dokładnie! To jest klasyczny przypadek polimorfizmu - możesz wykorzystać, 
że SingleBehavior i DualBehavior dziedziczą po BaseBehavior i mają te same metody. 
"""
import numpy as np
import tensorflow as tf
#
from klasy_jesien.behav0_base import BaseBehavior
from klasy_jesien.archit0_base import BaseArchitecture


class DualBehavior(BaseBehavior):
    """
    Dual behavach - osobne modele dla reżim A i reżim B.
    """
    
    def __init__(self, archit_instance: BaseArchitecture):
        super().__init__(archit_instance)
        # Wszystko już jest w BaseBehavior!
    
   
    
    def _prepare_2AB_krotki_of_3tvt_pdicts(self):
        """
        Przygotowuje pary krotek (X, y) dla dwóch reżimów A i B na podstawie kolumny flagowej.
        
        Metoda dzieli dane treningowe, walidacyjne i testowe na dwa osobne zestawy
        odpowiadające różnym reżimom pracy. Podział odbywa się na podstawie kolumny boolowskiej 
        kol_flagA zdefiniowanej w konfiguracji danych.
        
        Returns
        -------
        tuple of tuple
            (krotka_pdictow_A, krotka_pdictow_B) gdzie każda zawiera (train_pdict, val_pdict, test_pdict)
            z podzielonymi danymi dla odpowiedniego reżimu.
        """
        # Pobierz nazwę kolumny flagowej z konfiguracji danych pierwszego modelu
        kol_flagA = self.d1_pds.datacfg.kol_flagA
        
        print("\n\n\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(self.train_pdict['X_pds'].head())
        print("\n\n\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # Sprawdź czy kolumna flagowa jest zdefiniowana (wymagana dla dual-regime)
        if kol_flagA is None:
            raise ValueError("Dual Behavior wymaga parametru kol_flagA, a nie ma.")
        
        # Inicjalizuj puste słowniki dla trzech zestawów danych (train/val/test) dla obu reżimów
        train_pdictA, val_pdictA, test_pdictA = {}, {}, {}
        train_pdictB, val_pdictB, test_pdictB = {}, {}, {}
        
        # Zgrupuj słowniki w krotki dla wygodniejszego zwracania
        krotka_pdictow_A = (train_pdictA, val_pdictA, test_pdictA)
        krotka_pdictow_B = (train_pdictB, val_pdictB, test_pdictB)
        
        # Iteruj przez wszystkie trzy zestawy danych: treningowy, walidacyjny, testowy
        for dct, dctA, dctB in zip([self.train_pdict, self.val_pdict, self.test_pdict],
                                   krotka_pdictow_A,
                                   krotka_pdictow_B):
            
            # Struktura dct (przykład na train):
            # self.train_pdict = {'X_pds': self.X_train,  
            #                     'tvt_name': "train",
            #                     'y_indices': self.pd_train_idx, 
            #                     'y_actuals': self.y_train}
            
            # Skopiuj nazwę zestawu (train/val/test) do obu reżimów
            dctA['tvt_name'] = dctB['tvt_name'] = dct['tvt_name']
            
            # było źle, wróć:
            # # Wyciągnij maskę boolean z DataFrame X_pds (kolumna flagA jest tylko tutaj!)
            # maska_A = dct['X_pds'][kol_flagA].values  # .values zwraca numpy array z wartościami True/False
            # maska_B = ~maska_A  # Odwrócona maska dla reżimu B
            
            
            # Wyciągnij maskę boolean z DataFrame X_pds
            kolumna_flag = dct['X_pds'][kol_flagA]  # Series z wartościami 0/1
            maska_A = (kolumna_flag == 1).values  # Konwersja: 1 -> True, 0 -> False
            maska_B = ~maska_A  # Odwrócenie: True -> False, False -> True
            
            # Zastosuj maski do indeksowania WIERSZY
            dctA['X_pds'] = dct['X_pds'][maska_A].reset_index(drop=True)
            dctA['y_actuals'] = dct['y_actuals'][maska_A]
            dctA['y_indices'] = dct['y_indices'][maska_A]
            
            dctB['X_pds'] = dct['X_pds'][maska_B].reset_index(drop=True)
            dctB['y_actuals'] = dct['y_actuals'][maska_B]
            dctB['y_indices'] = dct['y_indices'][maska_B]
            

            
        # Zwróć dwie krotki słowników - jedną dla reżimu A, drugą dla reżimu B
        return krotka_pdictow_A, krotka_pdictow_B
    







    
    def _prepare_2AB_krotki_of_3tvt_kdicts(self): ## to raczej powinno być w klasy_jesien.behav2_dual -- już jest
        """
        Metoda dzieli dane treningowe, walidacyjne i testowe na dwa osobne zestawy
        odpowiadające różnym reżimom pracy. Podział odbywa się na podstawie kolumny boolowskiej kol_flagA zdefiniowanej
        w konfiguracji danych.
        """
        train_kds = self.train_kdict['kds']
        val_kds   = self.val_kdict['kds']
        test_kds  = self.test_kdict['kds'] if self.testsetatlast==True else None
        
        # Train
        (train_kdictA, train_kdictB, 
         train_maskA_dla_kluskow
         ) = self.d3_kds.d2_kluski.rozdziel_kluski_na_rezimy_A_i_B(
                                            self.train_kdict['kds'], 
                                            self.train_kdict['tvt_name'], 
                                            self.train_kdict['y_indices'], 
                                            self.train_kdict['y_actuals'], 
                                            shuffle=True)
        if train_kdictA["kds"] is not None:
            self._log_po_tasowaniu(train_kdictA["kds"], "Dual trainA",
                                   train_kdictA["y_indices"],
                                   int(train_maskA_dla_kluskow.sum()),
                                   train_kdictA["y_actuals"])
        if train_kdictB["kds"] is not None:
            self._log_po_tasowaniu(train_kdictB["kds"], "Dual trainB",
                                   train_kdictB["y_indices"],
                                   int((~train_maskA_dla_kluskow).sum()),
                                   train_kdictB["y_actuals"])

        # Val
        (val_kdictA, val_kdictB,
         val_maskA_dla_kluskow
         ) = self.d3_kds.d2_kluski.rozdziel_kluski_na_rezimy_A_i_B(
                                            self.val_kdict['kds'], 
                                            self.val_kdict['tvt_name'], 
                                            self.val_kdict['y_indices'],
                                            self.val_kdict['y_actuals'])
        # Test
        if self.testsetatlast==True:
            (test_kdictA, test_kdictB, 
             test_maskA_dla_kluskow
             ) = self.d3_kds.d2_kluski.rozdziel_kluski_na_rezimy_A_i_B(
                                                self.test_kdict['kds'], 
                                                self.test_kdict['tvt_name'], 
                                                self.test_kdict['y_indices'],
                                                self.test_kdict['y_actuals'])
        else:
            test_kdictA = test_kdictB = None
            test_maskA_dla_kluskow = None


        krotka_kdictow_A = (train_kdictA, val_kdictA, test_kdictA)
        krotka_kdictow_B = (train_kdictB, val_kdictB, test_kdictB)
        
        ## potrzebna?
        # krotka_masek_dla_kluskow = (train_maskA_dla_kluskow, 
        #                             val_maskA_dla_kluskow,
        #                             test_maskA_dla_kluskow)
        
        # Zwróć dwie krotki słowników - jedną dla reżimu A, drugą dla reżimu B
        return krotka_kdictow_A, krotka_kdictow_B









    def _prepare_tvt_data_for_tvt(self):
        """
        Dzieli dane na dwa zestawy: rezimA i rezimB.
        
        if self.is_deep:
            Returns: (krotkaA_of_3tvtdatadicts,
                      krotkaB_of_3tvtdatadicts)
                gdzie:
                krotkaA_of_3tvtdatadicts = (train_kdictA, val_kdictA, test_kdictA)
                krotkaB_of_3tvtdatadicts = (train_kdictB, val_kdictB, test_kdictB)
        """
        if self.is_deep:
            self.data_for_tvt = self._prepare_2AB_krotki_of_3tvt_kdicts()

        else:
            self.data_for_tvt = self._prepare_2AB_krotki_of_3tvt_pdicts()
        
        return self.data_for_tvt
    
    
    
    

    
    def run_runs_and_get_results_for_dfres(self, hpsA=None, hpsB=None):
        """
        Przeprowadza multiple runy dla dual-regime (A i B) i zbiera wyniki do późniejszej analizy.
        
        Metoda:
        1. Wykonuje 'ilerunow' runów dla każdego rezimu (A i B) osobno
        2. Dla każdego runu łączy (combo) predykcje z obu rezimów
        3. Zwraca indeksy, wartości rzeczywiste i słownik ze wszystkimi predykcjami combo
        
        Args:
            hpsA: hiperparametry dla rezimu A
            hpsB: hiperparametry dla rezimu B  
            
        Returns:
            KROTKA DWÓCH KROTEK UZUPEŁNIONYCH O WYNIKI !!!
            tuple: (indices_combo, actual_combo, all_runs_pred_dct_combo)
            - indices_combo: indeksy połączonych danych z obu rezimów
            - actual_combo: rzeczywiste wartości połączone z obu rezimów
            - all_runs_pred_dct_combo: słownik z predykcjami combo dla każdego runu
        """
        if self.is_deep:
            
            (krotkaA_of_3tvtdatadicts,  # train/val/test data dicts dla rezimu A
             predictions_dctA_do_combo,  # słownik predykcji z wszystkich runów A
             best_run_idxA,              # indeks najlepszego runu A
             all_runs_params_dictA) = self.archit.run_runs_elaborate("rezimA", 
                                            self.data_for_tvt[0],  # dane dla rezimu A
                                            hpsA
                                            )
            (krotkaB_of_3tvtdatadicts,  # train/val/test data dicts dla rezimu B
             predictions_dctB_do_combo,  # słownik predykcji z wszystkich runów B
             best_run_idxB,              # indeks najlepszego runu B
             all_runs_params_dictB) = self.archit.run_runs_elaborate("rezimB", 
                                            self.data_for_tvt[1],  # dane dla rezimu B
                                            hpsB
                                            )
            
            print("W TYM 'BEST' NIE MUSZĄ BYĆ TO TE SAME RUN INDICES!\nbest_run_idxA, best_run_idxB")
            print(best_run_idxA, best_run_idxB)
            
            
            
            
        else:
            (krotkaA_of_3tvtdatadicts,  # train/val/test data dicts dla rezimu A
             predictions_dctA_do_combo,  # słownik predykcji z wszystkich runów A
             best_run_idxA,              # indeks najlepszego runu A
             all_runs_params_dictA)  = self.archit.run_runs_elaborate("rezimA", 
                                            self.data_for_tvt[0],  # dane dla rezimu A
                                            )
            (krotkaB_of_3tvtdatadicts,  # train/val/test data dicts dla rezimu B
             predictions_dctB_do_combo,  # słownik predykcji z wszystkich runów B
             best_run_idxB,              # indeks najlepszego runu B
             all_runs_params_dictB) = self.archit.run_runs_elaborate("rezimB", 
                                            self.data_for_tvt[1],  # dane dla rezimu B
                                            )



            
        # Rozpakuj KROTKI UZUPEŁNIONE O WYNIKI (bez train) dla obu rezimów
        (_, val_kdictA, test_kdictA) = krotkaA_of_3tvtdatadicts
        (_, val_kdictB, test_kdictB) = krotkaB_of_3tvtdatadicts
        
        # Wybierz test set lub validation set do dalszych obliczeń
        if self.testsetatlast:
            dictA, dictB = test_kdictA, test_kdictB
        else: 
            dictA, dictB = val_kdictA, val_kdictB
        
        # Uzyskaj combo predykcję dla najlepszych runów (best_run_idxA i best_run_idxB)
        ##### WIEM ŻE TO SĄ NAJEPSZE RUNY BO JAKO kdict["y_pred"] ZAPISUJĘ BEST RUN 
        (indices_combo,      # połączone indeksy z obu rezimów
         actual_combo,       # połączone actual values z obu rezimów
         self.pred_best     # połączone predictions z obu rezimów z ich najlepszych runów
         ) = self._get_iap_combo_one_run(dictA, dictB)
        
# =============================================================================
#         print('\n\nself.best_pred:')
#         print(self.d3_kds.d1_pds.drukuj_strukture(self.pred_best))
# =============================================================================
        
        # if ilerunow > 1:
        # Jeśli było więcej niż 1 run, uzyskaj combo dla WSZYSTKICH runów
        (indices_combo, 
         actual_combo, 
         all_runs_pred_dct_combo) = self._get_iap_combo_all_runs(
                                              dictA, dictB, 
                                              predictions_dctA_do_combo, 
                                              predictions_dctB_do_combo
                                              )
# =============================================================================
#         print('all_runs_pred_dct_combo:')
#         print(self.d3_kds.d1_pds.drukuj_strukture(all_runs_pred_dct_combo))
# =============================================================================
        # Dodaj best combo jako osobny klucz (do porównania z innymi runami)
        # all_runs_pred_dct_combo["best_runMono_or_runsAB"] = pred_combo_best ## NIE BO TO POPSUJE STATYSTYKI
    
        # else:
            # Jeśli tylko 1 run, słownik zawiera tylko best combo
            # all_runs_pred_dct_combo = {0: self.pred_best} ########### ALE PO CO TA LINIJKA KODU???
        
        
        
        # Zapisz wyniki jako atrybut obiektu
        self.results_for_dfres = (indices_combo, 
                                  actual_combo, 
                                  all_runs_pred_dct_combo)
                 

        
        return self.results_for_dfres
    

        
        
        
        
    
    def _get_iap_combo_one_run(self, dictA, dictB):
        """returns:
            krotka i-a-p: indices_combo, actual_combo, pred_combo 
        """
        (indices_combo, 
         actual_combo, 
         pred_combo) = self._sklej_predykcje_indeksami(
                            dictA['y_indices'], dictA['y_actuals'], dictA['y_pred'], 
                            dictB['y_indices'], dictB['y_actuals'], dictB['y_pred']       
                            )
                            
        return indices_combo, actual_combo, pred_combo # krotka i-a-p



    def _get_iap_combo_all_runs(self, dictA, dictB, 
                                  predictions_dctA, predictions_dctB):
        """returns:
            krotka i-a-p_dct: indices_combo, actual_combo, all_runs_pred_dct_combo
        """
        all_runs_pred_dct_combo = {}
       
        for run_idx in predictions_dctA.keys():
            pred_rezimA = predictions_dctA[run_idx]
            pred_rezimB = predictions_dctB[run_idx]
            
            (indices_combo,
             actual_combo, 
             pred_combo) = self._sklej_predykcje_indeksami(
                                dictA['y_indices'], dictA['y_actuals'], pred_rezimA, 
                                dictB['y_indices'], dictB['y_actuals'], pred_rezimB       
                                )
                            
            all_runs_pred_dct_combo[run_idx] = pred_combo
            
        return indices_combo, actual_combo, all_runs_pred_dct_combo # krotka i-a-p_dict
    
    
    
    
    
    
    
    
    
    
    
    



    def _sklej_predykcje_indeksami(self, 
                                  indices_rezimA, actual_rezimA, predictions_rezimA,
                                  indices_rezimB, actual_rezimB, predictions_rezimB):
        """wyjete z funkcji oblicz_combined_metryki, bo niepotrzebnie bylo w niej 
        i przez to nie moglam jej uzyc jak nie bylo sklejania"""

        indices_combo = np.concatenate([indices_rezimA, indices_rezimB])
        actual_combo = np.concatenate([actual_rezimA, actual_rezimB])
        predictions_combo = np.concatenate([predictions_rezimA, predictions_rezimB])

        return indices_combo, actual_combo, predictions_combo


    










    
    def get_count(self):
        """Zwraca 2 - dwa modele"""
        return 2
    
    def get_names(self):
        """Zwraca ['rezimA', 'rezimB']"""
        return ['rezimA', 'rezimB']
    
    
    
