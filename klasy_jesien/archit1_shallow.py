#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shallow Architecture - modele sklearn
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
import joblib

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from klasy_jesien.archit0_base import BaseArchitecture
from klasy_data.d1_pds import PdsSetup


class ShallowArchitecture(BaseArchitecture):
    """
    Architektura modeli płytkich (sklearn regressors).
    
    TU NIE MOGĘ DAĆ GOTOWYCH DATASETÓW Z KLASY k1 BO W PRZYPADKU DUAL APPROACH BĘDĄ TO INNE DATASETY !!
    A TO KLASĘ ARCHIT WKŁADAM DO KLASY APPROACH A NIE ODWROTNIE.
    """

    def __init__(self, pds_setup_instance: PdsSetup, 
                       ilerunow: int, 
                       sm_type: str):
        """
        Args:

        """
        super().__init__(
            ilerunow=ilerunow,  # Przekazujesz to, co dostałeś jako argument
            timestampplus=pds_setup_instance.timestampplus,
            batchsize=None,
            seqlen=None
            )

        self.d1_pds = pds_setup_instance
        self.sm_type = sm_type
        
        self.timestampplus = self.d1_pds.timestampplus
        self.testsetatlast = self.d1_pds.testsetatlast
      
        # Bezpośrednie przypisania zamiast _setup()
        self.sciezka_play = self.d1_pds.k1_path
        self.results_path = self.d1_pds.k1_path.parent
        self.model_params_file = self.d1_pds.k1_path / f"model_params_info__{self.timestampplus}.txt"
        

        print(f"\n\njestem w klasy_jesien.archit1_shallow\nresults_path: {self.results_path}\n")
        
        
    def build(self, dshps=None, random_state=None):
        """
        Buduje model sklearn.
        
        Args:
            dshps: Obiekt konfiguracyjny (opcjonalny dla sklearn)
            random_state: Seed dla reprodukowalności
        Returns:
            Sklearn regressor
        """
        self.model = self._choose_regressor(self.sm_type, random_state)
        return self.model

    
    
    
    def _choose_regressor(self, sm_type, random_state=None):
        """Wybiera odpowiedni regressor - z oryginalnego choose_regressor
        !!! domyslnie Losowość włączona
        """
        
        regressor = None
        
        # PLS
        if sm_type == "pls":
            from sklearn.cross_decomposition import PLSRegression
            regressor = PLSRegression(n_components=5)
    
        # Linear Regression
        elif sm_type == "lr":
            from sklearn.linear_model import LinearRegression
            regressor = LinearRegression()
        
        
        # Lasso
        elif sm_type == "lasso":
            """lasso:
            wroc MH: 
            model = linear_model.LassoLarsCV( cv=tscv, max_n_alphas=10.fit(X_train, y_train) )
            wroc, grok:
            Czy random_state jest poprawny?: Tak. Lasso w sklearn obsługuje parametr random_state, który kontroluje losowość w przypadku, gdy solver iteracyjny (np. solver='saga') jest używany lub gdy model wymaga losowego inicjalizowania parametrów w pewnych przypadkach. Domyślnie Lasso używa deterministycznego solvera (coordinate descent), ale random_state jest akceptowalny i może być użyty w specyficznych przypadkach.
            Poprawka: Kod jest poprawny, ale możesz upewnić się, że random_state ma sens, jeśli używasz np. selection='random' w Lasso. Jeśli nie używasz losowości w obliczeniach, random_state jest ignorowane, ale nie powoduje błędu:
            python
            regressor = Lasso(alpha=0.10, random_state=random_state)
            Uwagi: Jeśli chcesz użyć LassoLarsCV (z komentarza w kodzie), pamiętaj, że LassoLarsCV również obsługuje random_state dla losowości w walidacji krzyżowej:
            python
            from sklearn.linear_model import LassoLarsCV
            regressor = LassoLarsCV(cv=5, max_n_alphas=10, random_state=random_state)
            """
            from sklearn.linear_model import Lasso
            regressor = Lasso(alpha=0.10, random_state=random_state)
    
    
        # Ridge
        elif sm_type == "ridge":
            """ridge:
            wroc, grok:
            Czy random_state jest poprawny?: Tak, ale z zastrzeżeniem. Ridge w sklearn obsługuje random_state, ale jest on używany tylko dla niektórych solverów, które wprowadzają losowość, np. solver='sag' (Stochastic Average Gradient) lub solver='saga'. Solver cholesky, który jest użyty w Twoim kodzie, jest deterministyczny i nie wykorzystuje random_state. W takim przypadku random_state jest ignorowane, ale nie powoduje błędu.
            Poprawka: Jeśli używasz solver="cholesky", random_state nie jest potrzebne i można je usunąć dla przejrzystości:
            regressor = Ridge(alpha=0.1, solver="cholesky")
            Jeśli jednak planujesz użyć solvera z losowością (np. sag lub saga), random_state jest poprawny:
            regressor = Ridge(alpha=0.1, solver="saga", random_state=random_state)
            Uwagi: Upewnij się, że solver pasuje do Twoich potrzeb. cholesky jest szybki i deterministyczny dla małych i średnich zbiorów danych, ale sag lub saga mogą być lepsze dla dużych zbiorów, gdzie losowość (i random_state) ma znaczenie.
            """
            from sklearn.linear_model import Ridge
            regressor = Ridge(alpha=0.1, solver="cholesky")
    
        
        # Random Forest
        elif sm_type == "rfr":
            """RFR:
            Number of trees (n_estimators): 100 gdy mało kols, 1000 gdy dużo kols,
            Max depth: not set, 
            Min samples per leaf (min_samples_leaf): 1, 
            
            max features: 0.9 gdy mało kols, 0.2 gdy dużo kols --- ja tak sobie kiedyś zapisałam, 
            ale 0.2 to chyba za mało...
            max_features — ile cech (kolumn X) losuje każde drzewo przy szukaniu najlepszego podziału. 
            Im mniej, tym bardziej różnorodne drzewa → mniejszy overfitting.
                "sqrt" — pierwiastek z liczby cech (domyślne dla klasyfikacji)
                1.0 — wszystkie cechy (domyślne dla regresji!) = każde drzewo widzi to samo = większy overfitting
                0.5 — połowa cech losowo
                int — konkretna liczba cech
                
            oob_score — Out-Of-Bag score. Każde drzewo trenuje się na ~63% próbek (bootstrap), pozostałe 37% to "oob". Sklearn automatycznie testuje model na tych pominiętych próbkach — dostajesz darmową ocenę jakości bez osobnego test setu.
            True — liczy, dostępne jako model.oob_score_
            False — nie liczy → domyślne: False
            Dokładnie. `oob_score=True` to tylko **przełącznik który mówi "policz mi tę dodatkową metrykę"** — sposób trenowania modelu się nie zmienia ani trochę. Random Forest i tak robi bootstrap (losuje próbki z powtórzeniami) przy budowie każdego drzewa, `oob_score` tylko mówi "przy okazji przetestuj każde drzewo na tych próbkach które pominął". Zero kosztu dla jakości modelu, minimalny koszt obliczeniowy.
            Przy Twoim przypadku — tak, warto włączyć — ale tylko jako dodatkowa informacja diagnostyczna, nie zamiast test setu.
            Konkretnie: masz shuffle=False czyli podział chronologiczny — train = starsza historia, test = najnowsze dane. To bardzo sensowne dla danych z turbiny (model uczy się na przeszłości, sprawdzasz na przyszłości). oob_score natomiast losuje próbki z całego zbioru treningowego — czyli jego R² nie uwzględnia tej chronologii.
            Więc:
            główna ocena modelu → Twój test R² (chronologiczny) — to jest prawdziwy egzamin
            oob_score → przydatny żeby szybko sprawdzić czy model w ogóle działa, albo porównać warianty bez patrzenia na test set
            Praktycznie: jeśli oob_score_ ≈ test R² to dobrze — model jest stabilny. Jeśli oob_score_ >> test R² to sygnał że dane z czasem się zmieniają (drift) i model gorzej radzi sobie z "przyszłością".


            n_jobs — ile rdzeni CPU używa do treningu.
            -1 — wszystkie dostępne
            1 — jeden rdzeń (→ domyślne: 1 )
            4 — cztery rdzenie
             """
            from sklearn.ensemble import RandomForestRegressor
            regressor = RandomForestRegressor(
                
                                     ## projekt soft sensor
                                     n_estimators=1000,
                                     #
                                     max_depth=15,
                                     min_samples_leaf=1,
                                     #
                                     max_features=0.5,
                                     oob_score=False, ########## 2026.02.22 bez sensu to dawać skoro potem nigdzie tego nie drukuję !!!
                                     n_jobs=-1, ## KONIECZNIE! JA MAM 4 CORY
                                     random_state=random_state)
                   
            
        # XGBoost
        elif sm_type == "xgbr":
            """XGBoost:	
            Learning rate: 0.1,
            Number of trees (n_estimators): 100 gdy mało kols, 1000 gdy dużo kols,
            Max depth: not set, 
            Column sample by tree (podobne do Max features): 0.9 gdy mało kols, 0.2 gdy dużo kols,
            Subsample: 0.9, Column sample by tree: 0.9, 
            L1 regularization (reg_alpha): 10, L2 regularization (reg_lambda): 10
            """
            import xgboost as xgb
            regressor = xgb.XGBRegressor(
                learning_rate=0.01,             # 0.001 Very small step size
                n_estimators=1000,              # 10 tys. very ! Large number of trees
                max_depth=15,                
                subsample=0.2,                  # 0.9 Use nearly all rows per tree
                colsample_bytree=0.5,           # 0.9 Use nearly all features per tree
                reg_alpha=10,                   # 10 High regularization
                reg_lambda=10,                  # 10 High regularization
                random_state=random_state
                )
    
    
        # Gradient Boosting
        elif sm_type == "gbr":
            from sklearn.ensemble import GradientBoostingRegressor
            regressor = GradientBoostingRegressor(max_depth=15*10,
                                             n_estimators=100,
                                             learning_rate=1.0,
                                             n_iter_no_change=100,
                                             random_state=random_state
                                             )
    
        else:
            raise ValueError(f"Nieznany typ modelu: {sm_type}")
    
        return regressor
        
        
        
    
    
    def fit(self, train_pds, val_pds=None, **kwargs):
        """
        Trenuje model sklearn.
        
        Args:
            train_pds: tuple (X_train, y_train) - pandas DataFrame/Series lub numpy
            val_pds:   tuple (X_val, y_val) - opcjonalne, nieużywane w sklearn
            **kwargs:  Dodatkowe parametry (nieużywane)
        Returns:
            None (sklearn nie zwraca historii)
        """
        if self.model is None:
            raise ValueError("Model nie został zbudowany. Wywołaj build() najpierw.")
        
        X_train, y_train = train_pds
        
        # Konwersja do odpowiednich typów
        X_train = X_train.astype("float64")
        
        if hasattr(y_train, 'values'):  # pandas Series
            y_train = y_train.astype("float64").values.ravel()
        else:  # numpy
            y_train = y_train.astype("float64").ravel()
        
        self.model.fit(X_train, y_train)
        return None
    
    
    
    

    def wiesz_co_masz_robic(self, run_str_id, train_pds, val_pds=None, dshps=None):
        
        t_train_start = time.time() # ⏱️ 
        
        self.model= self.build(dshps)
        self.fit(train_pds, val_pds)
        
        t_train_minutes = round((time.time()-t_train_start)/60) # czas treningu (minutes) ⏱️ 
        
        return self.model, t_train_minutes
    
    
    
    
    
    
    def evaluate(self, pds):
        """
        Obliczenie podstawowych metryk dla dataset.
        
        Args:
            pds: tuple (X_val, y_val) lub tuple (X_test, y_test)
        Returns:
            tuple (mse, mae)
        """
        if self.model is None:
            raise ValueError("Model nie został zbudowany.")
            
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        X_pds, y_pds = pds
        
        y_pred = self.model.predict(X_pds)
        mse = mean_squared_error(y_pds, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_pds, y_pred)
        
        mse, rmse, mae = round(mse, 3), round(rmse, 3), round(mae, 3) 
        print(f"mse={mse}, rmse={rmse},   mae={mae}")
        print("to jest liczone w mojej metodzie evaluate\n")
        
        return mse, mae

   
    
    
    
    def predict(self, pds):
        """
        Predykcja dla pandas dataset val lub test.
        Args:
            pds: DataFrame lub numpy array ########## A JA NIE ROZUMIEM, A DLACZEGO "LUB"?
        Returns:
            numpy array z predykcjami
        """
        if self.model is None:
            raise ValueError("Model nie został zbudowany.")
            
        X_pds, y_pds = pds
        
        t_pred_start = time.time() # ⏱️                 
        predictions = (self.model.predict(X_pds)
                       .round(2).astype("float64")
                       )
        t_pred_seconds = round(time.time()-t_pred_start)  # czas predykcji (s) ⏱️ 
        return predictions, t_pred_seconds
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def run_runs_elaborate( self, data_name, krotka_of_3tvtdatadicts, dshps=None):
        """RUN fit_and_evaluate_mdeep RUNS
        tutaj jest robiony i val i test sets bo chcę zobaczyć czy mam dobrze zbalansowane te zbiory
        """
        lines = [
                f"\n\n==================Data {data_name}=================="
                ]
        # PLAN: ZMIEŃ POD SHALLOW
        # for k, v in dshps.params_deep_model.items():
        #     lines.append(f"  {k:<15} {v}")
        # with open(self.model_params_file, "a") as file:
        #     file.write("\n".join(lines))
        logger.info(f"\n\n==================Data {data_name}==================")
                    
        (train_pdict, 
         val_pdict, 
         test_pdict) = krotka_of_3tvtdatadicts
    
        train_pds   = (train_pdict["X_pds"], train_pdict["y_actuals"])
        
        # print("\n\n\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!----->")
        # print(train_pds)
        # print("<------!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n\n")
        
        val_pds     = (val_pdict["X_pds"], val_pdict["y_actuals"])
        val_indices = val_pdict["y_indices"]


        if self.testsetatlast:
            test_pds     = (test_pdict["X_pds"], test_pdict["y_actuals"])
            test_indices = test_pdict["y_indices"]
        
            
        ################## WSZYSTKO GOTOWE! MOŻNA ROBIĆ COMBO DLA KAŻDEGO RUN !        
        val_predictions_dct = {}
        test_predictions_dct = {}
        
        
        best_val_mse = float('inf')
        best_run_idx = None
        all_runs_results_dicts_list = []
        # all_runs_str_id = f"{self.timestampplus}_{dshps.cfg_id or ''}_{data_name}" --- nauka: AttributeError: 'NoneType' object has no attribute 'cfg_id'
        all_runs_str_id = f"{self.timestampplus}_{dshps.cfg_id if dshps else ''}_{data_name}"
    
    
        for run_idx in range(self.ilerunow):
            logger.info(f"\n\n=========Run {run_idx}=========")
            run_str_id = all_runs_str_id + f"__run{run_idx}"
            
            
            # Fit i  Obliczenie podstawowych metryk dla val
            model, t_train_minutes = self.wiesz_co_masz_robic(
                                                        run_str_id,
                                                        train_pds, val_pds 
                                                        )
            val_mse, val_mae = self.evaluate(val_pds)
            val_predictions_dct[run_idx], t_pred_seconds = self.predict(val_pds)
            
            logger.info(f"Czas treningu dla Run {run_idx}: {t_train_minutes} minut \n"
                        f"nval_mse, val_mae = {val_mse}, {val_mae}")
           
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # obliczenia dla test set
            if self.testsetatlast:
                test_mse, test_mae = self.evaluate(test_pds)
                test_predictions_dct[run_idx], t_pred_seconds = self.predict(test_pds)

            else:
                test_predictions_dct[run_idx], test_mse, test_mae = None, None, None
            
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            one_run_results_dict = self._stworz_one_run_results_dict(
                                                    run_idx, data_name, 
                                                    t_train_minutes, t_pred_seconds,
                                                    val_mse, val_mae, 
                                                    test_mse, test_mae)
            # Dodanie do listy wyników
            all_runs_results_dicts_list.append(one_run_results_dict)
            
            # Polowanie na najlepszy model
            if (len(all_runs_results_dicts_list) == 0) or (val_mse < best_val_mse):
                best_val_mse = val_mse
                # best_model = model  # obiekt Keras ## mi to nie jest potrzebne; zapisują się wszystkie - po best_run_idx wiem który model jest best
                best_run_idx = run_idx
    
        print()
        print()
        # Zapis wyników do CSV
        results_df = pd.DataFrame(all_runs_results_dicts_list)
        results_df.to_csv(self.sciezka_play / f"all_runs_results__{all_runs_str_id}.csv", index=False)
        print(f"Zapisano csv-a all_runs_results dla {data_name}")
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # tego nie robię dla shallow: 
# =============================================================================
#         all_runs_params_dict = self._stworz_all_runs_params_dict(dshps, model.count_params())
#         self.calc_and_save_all_runs_summary_stats( all_runs_str_id, 
#                                           all_runs_results_dicts_list,
#                                           all_runs_params_dict )
#         print(f"Dopisano summary stats dla {data_name} do summary_stats__{self.timestampplus}.csv")
# =============================================================================
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ##### y_pred dla wielu runów TO JEST y_pred_best_run !!!
        val_pdict["y_pred"] = val_predictions_dct[best_run_idx]
        if self.testsetatlast:
            test_pdict["y_pred"] = test_predictions_dct[best_run_idx]
        
        krotka_of_3tvtdatadicts = (train_pdict, 
                                     val_pdict, 
                                     test_pdict) 
    
        predictions_dct_val_or_test = test_predictions_dct if self.testsetatlast else val_predictions_dct
    
        return (krotka_of_3tvtdatadicts, predictions_dct_val_or_test, best_run_idx, None)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def get_feature_importance(self):
        """
        Zwraca ważność cech (jeśli dostępna).
        Returns:
            numpy array lub None
        """
        print("\n\nget_feature_importance() - to mogą być po prostu coefficients:")
        if hasattr(self.model, 'coef_'):
            return self.model.coef_.reshape(-1)
        elif hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None
        
        
        
        
    def get_permut_importance(self, krotka_of_3tvtdatadicts):
        """
        Permutation Importance Needs a Scoring Metric
        By default, permutation_importance uses the model's default score function (typically r2_score for regressors).
        If you suspect outliers, try mean absolute error (MAE) instead.
        
        MUSI BYĆ JUŻ WYTRENOWANY MODEL. If the model's predictions are far from y_val, the metric might fail.

        Returns
        -------
        None.

        """
        print("\n\nget_permut_importance():")
        if self.model is None:
            raise ValueError("Model nie został zbudowany. Wywołaj build() najpierw.")
            
        from sklearn.inspection import permutation_importance
        
        (_, val_pdict, _) = krotka_of_3tvtdatadicts
        X_val, y_val     = (val_pdict["X_pds"], val_pdict["y_actuals"])
            
        permut_importance = permutation_importance(
                                self.model, X_val, y_val,
                                n_repeats=10, random_state=42, 
                                scoring="neg_mean_absolute_error"
                            )

        # Convert to a DataFrame for easy analysis
        importance_df = pd.DataFrame({
            'feature': X_val.columns,
            'importance_mean': permut_importance.importances_mean,
            'importance_std': permut_importance.importances_std
        }).sort_values(by="importance_mean", ascending=False)
    
        # Display results
        logger.info(f"\n\npermutation importance, użyty model shallow: {self.sm_type}\n")
        logger.info(importance_df.to_string())
        
        return importance_df
        
        
        
        
        
        
        
    
    def save(self, filepath):
        """Zapisuje model sklearn używając joblib"""
        if self.model is not None:
            joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        """Wczytuje model sklearn używając joblib"""
        self.model = joblib.load(filepath)
        
        
    def get_name(self):
        """dla deep jest to po prostu string 'deep',
        dla shallow string 'shallow'
        """
        return 'shallow'
    
    
    

