#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KdsSetup - Zrefaktoryzowana wersja
"""
import logging
logger = logging.getLogger(__name__)
#
import numpy as np
import pandas as pd
#
from klasy_data.d0_data import DataConfig
from klasy_data.d1_pds import PdsSetup
from klasy_data.d2_kluski import KluskiConfig

class KdsSetup:
    
    def __init__(self, 
                 datacfg_instance: DataConfig,
                 wybor_kols: str,
                 epochs: int,
                 batchsize: int,
                 seqlen: int,
                 #
                 timestampplus: str,
                 sprkod: bool = False,
                 testsetatlast: bool = False):

        self.datacfg = datacfg_instance # ze środka tej klasy nic nie rozpakowywuję, bo używam sporadycznie
        #
        self.wybor_kols = wybor_kols
        self.epochs = epochs
        self.batchsize = batchsize
        self.seqlen = seqlen
        #
        self.timestampplus = timestampplus
        self.sprkod = sprkod
        self.testsetatlast = testsetatlast
                
        # Initialize attributes --- TO JEST TYLKO DLA PRZEJRZYSTOŚCI, nie wymagane
        self.d1_pds = None
        self.d2_kluski = None
        self.train_kdict = self.val_kdict = self.test_kdict = None
        #
        self._setup()


    def _setup(self):
        """Prepare data and store for reuse."""
        
        # PdsSetup ============================================================
        self.d1_pds = PdsSetup(
                            self.datacfg,
                            self.wybor_kols, 
                            self.timestampplus, 
                            self.sprkod, 
                            self.testsetatlast
                            )


        # KluskiConfig ========================================================
        self.d2_kluski = KluskiConfig(
                            self.d1_pds.df_for_keras, 
                            self.d1_pds.qty_train, self.d1_pds.qty_val, self.d1_pds.qty_test,
                            self.d1_pds.lp_kolumny_flagi,
                            self.epochs, self.batchsize, self.seqlen,
                            self.d1_pds.data_prep_file,
                            self.testsetatlast,
                            )
        


        # Prepare base kds dicts; ZAWSZE OD RAZU ZROBI test_kds ALE JAK NIEPOTRZEBNY TO GO ŻADNA METODA NIE UŻYJE
        (self.train_kdict, 
         self.val_kdict, 
         self.test_kdict) = self.d2_kluski.stworz_uzupelnione_slowniki()

           
       

    
