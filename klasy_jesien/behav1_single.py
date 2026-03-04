#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Strategy - jeden model dla wszystkich danych
"""
import numpy as np
import tensorflow as tf
#
from klasy_jesien.behav0_base import BaseBehavior
from klasy_jesien.archit0_base import BaseArchitecture



        
class SingleBehavior(BaseBehavior):
    """
    Single behavach - jeden model dla wszystkich danych.
    """
    
    def __init__(self, archit_instance: BaseArchitecture):
        super().__init__(archit_instance)
        # Wszystko już jest w BaseBehavior!
    
        

    def _prepare_1Mono_krotke_of_3tvt_pdicts(self): ## to raczej powinno być w klasy_jesien.behav1_single -- juz jest 
        """
        Returns:
            dict: Dane gotowe do run_runs
        """
        krotka_kdictow_Mono = (self.train_pdict,
                               self.val_pdict, 
                               self.test_pdict)
        return krotka_kdictow_Mono    
    

        
    def _prepare_1Mono_krotke_of_3tvt_kdicts(self): ## to raczej powinno być w klasy_jesien.behav1_single -- juz jest 
        """
        Returns:
            dict: Dane gotowe do run_runs
        """
        keys_to_keep = ["kds", "tvt_name", "data_name", "y_indices", "y_actuals"]

        train_kdict = {k: v for k, v in self.train_kdict.items() 
               if k in keys_to_keep}
        train_kdict["kds"]= train_kdict["kds"].shuffle( 
                                buffer_size=self.train_kdict["ilosc_kluskow"])
        val_kdict = {k: v for k, v in self.val_kdict.items() 
               if k in keys_to_keep}
        test_kdict = {k: v for k, v in self.test_kdict.items() 
               if k in keys_to_keep}

        krotka_kdictow_Mono = (train_kdict, 
                                val_kdict, 
                                test_kdict)
        return krotka_kdictow_Mono

    


    def _prepare_tvt_data_for_tvt(self):
        """
        Returns: 
            1 Mono-krotka of 3 tvt-kdicts/tvt-pdicts przed treningiem, niepełna:
            (train_kdictMono, val_kdictMono, test_kdictMono)
        """
        if self.is_deep:
            # (train_kdictMono, val_kdictMono, test_kdictMono)
            self.data_for_tvt = self._prepare_1Mono_krotke_of_3tvt_kdicts()
  
        else:
            # (train_pdictMono, val_pdictMono, test_pdictMono)
            self.data_for_tvt = self._prepare_1Mono_krotke_of_3tvt_pdicts()
        
        return self.data_for_tvt


    
    def run_runs_and_get_results_for_dfres(self, hpsMono=None, hpsB=None):
        """
        Przeprowadza multiple runy dla single-regime (Mono) i zbiera wyniki do późniejszej analizy.
        
        Metoda:
        1. Wykonuje 'ilerunow' runów 
        2. Zwraca indeksy, wartości rzeczywiste i słownik ze wszystkimi predykcjami
        
        Args:
            hpsMono: hiperparametry dla rezimu Mono
                        
        Returns:
            JEDNA KROTKA UZUPEŁNIONA O WYNIKI !!!
            tuple: (indices, actual, combo_AB_all_runs_pred_dct)
            - indices: indeksy połączonych danych z obu rezimów
            - actual: rzeczywiste wartości połączone z obu rezimów
            - combo_AB_all_runs_pred_dct: słownik z predykcjami combo dla każdego runu
        """
        if self.is_deep:
            # 1 Mono-krotka of 3 tvt-kdicts po treningu, pełna
            (krotkaMono_of_3tvt_results_dicts,
             predictions_dctMono, 
             best_run_idxMono, 
             all_runs_params_dictMono) = self.archit.run_runs_elaborate(
                                                        "Mono", 
                                                        self.data_for_tvt,
                                                        hpsMono
                                                        )
                 
            self.pred_best = predictions_dctMono[best_run_idxMono] #### czemu tu to jest a w dual nie ma?

                
        else:
            # 1 Mono-krotka of 3 tvt-kdicts po treningu, pełna
            (krotkaMono_of_3tvt_results_dicts,
             predictions_dctMono, 
             best_run_idxMono, 
             all_runs_params_dictMono) = self.archit.run_runs_elaborate(
                                                        "Mono", 
                                                        self.data_for_tvt,
                                                        hpsMono
                                                        )
                 
            self.pred_best = predictions_dctMono[best_run_idxMono]#### czemu tu to jest a w dual nie ma?
            
            
            
            

       # Wybierz test set lub validation set do dalszych obliczeń
        dict_val_or_test = (krotkaMono_of_3tvt_results_dicts[2] # test_kdictMono
                          if self.testsetatlast 
                          else krotkaMono_of_3tvt_results_dicts[1] # val_kdictMono
                          )
        
        self.results_for_dfres = (dict_val_or_test['y_indices'], 
                                  dict_val_or_test['y_actuals'], 
                                  predictions_dctMono
                                  )

       
        return self.results_for_dfres






    def sklej_predykcje_indeksami(self, *args, **kwargs):
        print("Sylwia, jesteś w SingleBehavior, z jakiej racji tych chcesz coś sklejać?")


    def get_count(self):
        """Zwraca 1 - tylko jeden model"""
        return 1
    
    def get_names(self):
        """Zwraca ['Mono']"""
        return ['Mono']
    



















































