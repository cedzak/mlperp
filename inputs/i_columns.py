#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 19:06:57 2025; @author: sylwia
"""
#### cols :: gdatge
kols_map_gdatge = None

#### cols :: gdaciep

top8_gdaciep_kols_feat_importance = [
    "lag_1_MW_th",
    "roll_mean_6_MW_th",
    "lag_2_MW_th",
    "t_air",
    "roll_mean_6_t_air",
    "t_zasilania",
    "lag_24_MW_th",
    "roll_mean_48_MW_th",
] + ['MW_th', 'is_weekend']

top10_gdaciep_kols_permut_importance = [
    "lag_1_MW_th",
    "roll_mean_6_MW_th",
    "t_zasilania",
    "lag_2_MW_th",
    "t_air",
    "roll_mean_48_MW_th",
    "lag_24_MW_th",
    "roll_mean_48_t_air",
    "ghi",
    "lag_48_MW_th",
] + ['MW_th', 'is_weekend']


# `dict.fromkeys()` usuwa duplikaty zachowując kolejność (od Pythona 3.7+)
# `dict.fromkeys()` tworzy słownik gdzie klucze to elementy listy, a wartości to `None` !!!
# Ale zaraz owijamy to w `list()`, więc wyciągamy tylko klucze — i dostajemy czystą listę bez duplikatów. Trochę hacky trick, ale powszechnie używany.
nowosc = dict.fromkeys(top8_gdaciep_kols_feat_importance + top10_gdaciep_kols_permut_importance)
top_gdaciep_kols = list(nowosc)





gdaciep_base_kols = [
    'MW_th',
    'cloud_cover',
    'day_of_week',
    'day_of_year',
    'diffuse_radiation',
    'direct_radiation',
    'ghi',
    'hour',
    'humidity',
    'is_weekend',
    'month',
    't_air',
    # 't_powrotu',
    # 't_zasilania',
    'wind_speed_100m',
    'wind_speed_10m',]


gdaciep_sincos_kols = [
    'sin_day_of_week',
    'sin_hour',
    'sin_month',
    'cos_day_of_week',
    'cos_hour',
    'cos_month',]


gdaciep_lagtar_kols = [
    'lag_168_MW_th',
    'lag_1_MW_th',
    'lag_24_MW_th',
    'lag_2_MW_th',
    'lag_48_MW_th',]


gdaciep_roll_kols = [
    'roll_mean_48_MW_th',
    'roll_mean_48_cloud_cover',
    'roll_mean_48_diffuse_radiation',
    'roll_mean_48_direct_radiation',
    'roll_mean_48_humidity',
    'roll_mean_48_t_air',
    'roll_mean_48_t_powrotu',
    'roll_mean_48_t_zasilania',
    'roll_mean_48_wind_speed_100m',
    'roll_mean_48_wind_speed_10m',
    'roll_mean_6_MW_th',
    'roll_mean_6_cloud_cover',
    'roll_mean_6_diffuse_radiation',
    'roll_mean_6_direct_radiation',
    'roll_mean_6_humidity',
    'roll_mean_6_t_air',
    'roll_mean_6_t_powrotu',
    'roll_mean_6_t_zasilania',
    'roll_mean_6_wind_speed_100m',
    'roll_mean_6_wind_speed_10m',]




kols_map_gdaciep = {
    'top_gdaciep_kols':     top_gdaciep_kols,
    'top8_gdaciep_kols_feat_importance':    top8_gdaciep_kols_feat_importance,
    'top10_gdaciep_kols_permut_importance': top10_gdaciep_kols_permut_importance,
    'gdaciep_base_kols':                   gdaciep_base_kols,
    'gdaciep_base_sincos_kols':            gdaciep_base_kols + gdaciep_sincos_kols,
    'gdaciep_base_sincos_lagtar_kols':     gdaciep_base_kols + gdaciep_sincos_kols + gdaciep_lagtar_kols,
    'gdaciep_base_sincos_roll_kols':       gdaciep_base_kols + gdaciep_sincos_kols + gdaciep_roll_kols,
    'gdaciep_base_lagtar_kols':            gdaciep_base_kols + gdaciep_lagtar_kols,
    'gdaciep_base_lagtar_roll_kols':       gdaciep_base_kols + gdaciep_lagtar_kols + gdaciep_roll_kols,
    'gdaciep_base_roll_kols':              gdaciep_base_kols + gdaciep_roll_kols,
    }



#### cols :: tg5

## STERE DANE (inne kolumny były)
# tg_rokujace=[  'MW_moc', 'shutdown',
#                 'mm_condenser',
#                 "t_LPT-inlet_IPT-outlet",
#                 't_LPT outlet_A',
#                 't_CT inlet_10A',       
#                 't_ctp outlet_1',
#                 't_LP-FWH inlet_3',
#                 't__paraswieza'  ]

# # permutation importance, użyty model shallow: rfr, target: 't_LPT outlet_A'
# # 2025-11-29 20:50:07 - klasy_jesien.archit1_shallow - INFO - feature  importance_mean  importance_std

# # 3          t_CT inlet_10A         0.910088        0.001487
# # 4          t_ctp outlet_1         0.833387        0.001234
# # 5        t_LP-FWH inlet_3         0.722663        0.001033
# # 0                  MW_moc         0.578413        0.001149
# # 
# # 6           t__paraswieza         0.225606        0.000636
# # 2  t_LPT-inlet_IPT-outlet         0.070014        0.000661
# # 1            mm_condenser        -0.006241        0.000314


tg_all_kols_bez_A=[
  "MW_moc",
  "shutdown",
  "mm_rozpr1",
  "mm_rozpr2",
  "mm_rozpr3",
  "p_LP-FWH inlet_3",
  "p_LP-FWH inlet_4",
  "p_LPT inlet_A",
  "p_LPT inlet_B",
  "p_LPT inlet_C",
  "p_LPT outlet_A",
  "p_LPT outlet_B",
  "p_LPT outlet_C",
  "p_M_para_11",
  "p_M_para_12",
  "p_condenser",
  "t_LP-FWH inlet_3",
  "t_LP-FWH inlet_4",
  "t_LPT outlet_A",
  "t_LPT outlet_B",
  "t_LPT outlet_C",
  "t_LPT-inlet_IPT-outlet",
  "t_ST_INL_HRS_11",
  "t_ST_INL_HRS_12",
  ]

# get_permut_importance():
#                    feature  importance_mean  importance_std
# 18           p_LPT inlet_A     7.495465e+01    2.249496e-01
# 20           p_LPT inlet_C     6.873170e+01    1.993705e-01
# 19           p_LPT inlet_B     4.980150e+01    1.320584e-01
# 31                shutdown     9.603163e+00    4.860423e-02
# 12             p_M_para_12     5.533490e+00    2.066059e-02
# 3         p_LP-FWH inlet_4     4.529728e+00    2.178711e-02
# 11             p_M_para_11     3.544676e+00    1.573659e-02
# 1         p_LP-FWH inlet_3     2.149979e+00    2.043367e-02
# 4         t_LP-FWH inlet_4     1.897772e+00    1.209779e-02
# 16          t_LPT outlet_B     1.109079e+00    1.406940e-02
# 15          t_LPT outlet_A     7.495243e-01    8.719557e-03
# 17          t_LPT outlet_C     6.996674e-01    7.687521e-03
# 9           p_LPT outlet_B     5.660278e-01    1.041004e-02
# 8           p_LPT outlet_A     5.217390e-01    7.795266e-03
# 2         t_LP-FWH inlet_3     4.335065e-01    6.229923e-03
# 14         t_ST_INL_HRS_11     3.986533e-01    8.979792e-03
# 13         t_ST_INL_HRS_12     3.479588e-01    4.920327e-03
# 5                mm_rozpr1     2.760511e-01    8.190380e-03
# 7                mm_rozpr3     2.748884e-01    9.820308e-03
# 0              p_condenser     2.599078e-01    5.615635e-03
# 10          p_LPT outlet_C     2.402982e-01    5.476275e-03
# 21  t_LPT-inlet_IPT-outlet     4.955229e-02    2.105034e-03
# 6                mm_rozpr2     4.139550e-02    1.931278e-03



tg_naj_wg_pairplots = [
    'MW_moc', 'shutdown' ,
    # "p_LP-FWH inlet_3",
    "p_LP-FWH inlet_4",
    # "p_LPT inlet_A",
    # "p_LPT inlet_B",
    "p_LPT inlet_C",
    # "p_M_para_11",
    "p_M_para_12",
    ]



tg_naj_wg_pairplots_plethora = [
    'MW_moc', 'shutdown' ,
    "p_LP-FWH inlet_3",
    "p_LP-FWH inlet_4",
    "p_LPT inlet_A",
    "p_LPT inlet_B",
    "p_LPT inlet_C",
    "p_M_para_11",
    "p_M_para_12",
    ]



tg_rfr_important=[
    "MW_moc",
    "shutdown",
    # "mm_rozpr1",
    # "mm_rozpr2",
    # "mm_rozpr3",
    # "p_LP-FWH inlet_3",
    "p_LP-FWH inlet_4",
    "p_LPT inlet_A",
    # "p_LPT inlet_B",
    # "p_LPT inlet_C",
    "p_LPT outlet_A",
    # "p_LPT outlet_B",
    # "p_LPT outlet_C",
    "p_M_para_11",
    "p_M_para_12",
    "p_condenser",
    # "t_LP-FWH inlet_3",
    "t_LP-FWH inlet_4",
    # "t_LPT outlet_A",
    # "t_LPT outlet_B",
    # "t_LPT outlet_C",
    # "t_LPT-inlet_IPT-outlet",
    # "t_ST_INL_HRS_11",
    # "t_ST_INL_HRS_12",
    ]


tg_rokujace=[  'MW_moc', 'shutdown',
                # 'mm_condenser',
                "t_LPT-inlet_IPT-outlet",
                't_LPT outlet_A',
                # 't_CT inlet_10A',       
                # 't_ctp outlet_1',
                't_LP-FWH inlet_3',
                # 't__paraswieza'  
                ]


# DLA STARYCH DANYCH część tych kolumn usunęłam przy sprzątaniu, DLA NOWYCH DANYCH mam inne kols, więc tutaj to zmieniłm (2025.12.01)
tg_coolprop = ['MW_moc', 'shutdown' 
                'p_condenser', 
                'p_LPT outlet_A', 'p_LPT outlet_B', 'p_LPT outlet_C',
                't_LPT outlet_A', 't_LPT outlet_B', 't_LPT outlet_C',
                ]


tg_do_pred_mocy = ['MW_moc', 'shutdown',
                    't_LP-FWH inlet_4', 'p_LPT inlet_A', 'mm_rozpr1']
    
    




kols_map_tg5 = {'tg_rokujace': tg_rokujace,  
                'tg_rfr_important': tg_rfr_important,
                'tg_coolprop': tg_coolprop,
                'tg_do_pred_mocy': tg_do_pred_mocy,
                'tg_all_kols_bez_A': tg_all_kols_bez_A,
                'tg_naj_wg_pairplots': tg_naj_wg_pairplots,
                'tg_naj_wg_pairplots_plethora': tg_naj_wg_pairplots_plethora,
                }










#### cols :: mr
mr_testowe = ['t_blade', 'shut_minutes', 'shutdown', 't_casing']



mr_sensors_raw = [   'power',
                       'RPM', 
                       't_casing',
                       't_blade',
                       #
                       'p_IP_sec-steam', # 't_IP_steam', ## nie ma dla 20250521
                       'p_LP_steam_A', 't_LP_steam', 
                       #
                       # 'flow_IP_sec-steam', ##### calculated, wcale ze nie raw
                       #
                       # 'power_SP',
                       # 't_IP_steam_SP'
                       ]


mr_almost_all = ['RPM', 'RPM_diff_binned', 
                   'power', 
                   'shut_minutes', 'shutdown', 'startup_180', 'startup_start', 
                   't5smth_LP_steam', 't5smth_blade', 
                   't5smth_LP_steam__lag30', 't5smth_LP_steam__lag60', 't5smth_blade__lag30', 't5smth_blade__lag60', 
                   'logdiff_t5smth_LP_steam_2ewm600', 'logdiff_t5smth_blade_2ewm600', 
                   't_LP_steam', 't_LP_steam_diff5', 't_LP_steam_lag1', 't_LP_steam_lag2', 't_LP_steam_lag3', 't_LP_steam_lag4', 
                   't_blade', 't_blade_diff5', 't_blade_lag1', 't_blade_lag2', 't_blade_lag3', 't_blade_lag4', 
                   # 'power_SP', ## nie ma dla 20250521 
                   # 't_IP_steam', 't_IP_steam_SP', 't5smth_IP_steam', 't5smth_IP_steam__lag30', 't5smth_IP_steam__lag60', 'logdiff_t5smth_IP_steam_2ewm600', ## nie ma dla 20250521
                   't_casing']

mr_active =  ['t_LP_steam', 't_blade', 
                # 't_IP_steam', # bo nie ma w danych z 20250521
                'RPM',
                'power',
                # 'power_SP', # bo nie ma w danych z 20250521
                # 't_IP_steam_SP', # bo nie ma w danych z 20250521
                "startup_180",
                "shutdown",
                't_casing']


mr_shutdown =['t_blade', 't_LP_steam', 
                # "t_IP_steam", # bo nie ma w danych z 20250521
                't5smth_blade', 't5smth_LP_steam', 
                # "t5smth_IP_steam", # bo nie ma w danych z 20250521
                'shut_minutes', 
                "shutdown",
                't_casing']


mr_dualowe = list( set(mr_active) | set(mr_shutdown) )


mr_dualowe_i_ewm = mr_dualowe + ['logdiff_t5smth_LP_steam_2ewm600', 'logdiff_t5smth_blade_2ewm600']


mr_20250521_niedokonczony =  ['t_LP_steam', 't_blade', 
                                't5smth_blade', 't5smth_LP_steam',
                                'RPM',
                                'power',
                                't_casing']


mr_niedokonczony_i_shutdown =  ['t_LP_steam', 't_blade', 
                                't5smth_blade', 't5smth_LP_steam',
                                'RPM',
                                'power',
                                "shutdown",
                                't_casing']

mr_best_features_rfr_xgbr = ['t_blade_lag4', 't5smth_LP_steam__lag60', 'shut_minutes', 'shutdown', 't_casing']


kols_map_mr = {
                "mr_dualowe": mr_dualowe,
                "mr_best_features_rfr_xgbr": mr_best_features_rfr_xgbr,
                "mr_almost_all": mr_almost_all,
                "mr_testowe": mr_testowe,
                "mr_sensors_raw": mr_sensors_raw,
                "mr_niedokonczony_i_shutdown": mr_niedokonczony_i_shutdown
                }





























