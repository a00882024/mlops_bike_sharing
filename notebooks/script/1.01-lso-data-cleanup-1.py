#!/usr/bin/env python
# coding: utf-8

# # Librerías:

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import holidays
import calendar

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# # Lectura de datos:
# Se carga el dataset original y modificado para comparar los datos.

original_bike_sharing_df = pd.read_csv('../data/raw/bike_sharing_original.csv')
bike_sharing_df = pd.read_csv('../data/raw/bike_sharing_modified.csv')


target_rows_to_drop = bike_sharing_df.shape[0] - original_bike_sharing_df.shape[0]
target_cols_to_drop = bike_sharing_df.shape[1] - original_bike_sharing_df.shape[1]

print(f'Registros objetivo a eliminar {target_rows_to_drop}')
print(f'Columnas objetivo a eliminar {target_cols_to_drop}')


def cast_values(df):
        column_types_map = {
            'instant': 'int',
            'season': 'int',
            'yr': 'int',
            'mnth': 'int',
            'hr': 'int',
            'holiday': 'int',
            'weekday': 'int',
            'workingday': 'int',
            'weathersit': 'int',
            'temp': 'float',
            'atemp': 'float',
            'hum': 'float',
            'windspeed': 'float',
            'casual': 'int',
            'registered': 'int',
            'cnt': 'int',
            'mixed_type_col': 'int'
        }

        cast_map = {
            'int': lambda df, col: pd.to_numeric(df[col], errors='coerce').astype('Int64'),
            'float': lambda df, col: pd.to_numeric(df[col], errors='coerce').astype('float64')
        }

        for col_name, target_dtype in column_types_map.items():
            if col_name in df.columns:
                cast_map.get(target_dtype.lower())

                cast_action = cast_map.get(target_dtype.lower())

                if cast_action:
                    df[col_name] = cast_action(df, col_name)

cast_values(bike_sharing_df)


def get_cat_cols(df):
    cat_cols = df.select_dtypes(include = 'object').columns.tolist()

    return cat_cols

def get_num_cols(df):
    num_cols = df.select_dtypes(exclude = 'object').columns.tolist()

    return num_cols


cat_cols = get_cat_cols(bike_sharing_df)
num_cols = get_num_cols(bike_sharing_df)


def drop_cols(df, cols):
    df.drop(columns = cols, inplace=True)
    print(f'Columna (s) {cols} eliminada (s) del dataframe.')


def get_cols_to_drop(baseline_df, modified_df):
    original_cols = set(baseline_df.columns)
    modified_cols = set(modified_df.columns)

    cols_to_drop = []
    diff_cols = modified_cols - original_cols

    if len(diff_cols) == target_cols_to_drop:
        cols_to_drop = diff_cols.pop()
        print(f'Columna (s) a eliminar: {cols_to_drop}')

    return cols_to_drop


cols_to_drop = get_cols_to_drop(original_bike_sharing_df, bike_sharing_df)
drop_cols(bike_sharing_df, cols_to_drop)


cat_cols = get_cat_cols(bike_sharing_df)
num_cols = get_num_cols(bike_sharing_df)


def cast_values(df):
        column_types_map = {
            'instant': 'int',
            'season': 'int',
            'yr': 'int',
            'mnth': 'int',
            'hr': 'int',
            'holiday': 'int',
            'weekday': 'int',
            'workingday': 'int',
            'weathersit': 'int',
            'temp': 'float',
            'atemp': 'float',
            'hum': 'float',
            'windspeed': 'float',
            'casual': 'int',
            'registered': 'int',
            'cnt': 'int',
            'mixed_type_col': 'int'
        }

        cast_map = {
            'int': lambda df, col: pd.to_numeric(df[col], errors='coerce').astype('Int64'),
            'float': lambda df, col: pd.to_numeric(df[col], errors='coerce').astype('float64')
        }

        for col_name, target_dtype in column_types_map.items():
            if col_name in df.columns:
                cast_map.get(target_dtype.lower())

                cast_action = cast_map.get(target_dtype.lower())

                if cast_action:
                    df[col_name] = cast_action(df, col_name)

cast_values(bike_sharing_df)


# # Limpieza de datos:
# - Manejo de valores faltantes en todas las variables
# - Manejo de valores atípicos en todas las variables
# - Estandarización de formato de fechas
# - Estandarización de parámetros (ej: formato 24 hrs para variable 'hr')
# - Manejo de operaciones aritméticas correctas ('casual' + 'registered' = 'cnt')

## DATEDAY -> formato de fecha
##
def convert_date_format(df, date_col):
    count = 0

    df[date_col] = df[date_col].astype(str).str.strip()
    initial_series = df[date_col].copy()

    # Formato 1: YYYY-MM-DD
    df[date_col] = pd.to_datetime(initial_series, format='%Y-%m-%d', errors='coerce')

    # Formato 2: MM/DD/YYYY (US)
    mask_nan_us = df[date_col].isna()

    df.loc[mask_nan_us, date_col] = pd.to_datetime(
        initial_series[mask_nan_us], 
        format='%m/%d/%Y', 
        errors='coerce'
    )

    count += (~df[date_col].isna() | mask_nan_us).sum()

    # Formato 3: DD/MM/YYYY (EU)
    mask_nan_eu = df[date_col].isna()

    df.loc[mask_nan_eu, date_col] = pd.to_datetime(
        initial_series[mask_nan_eu], 
        format='%d/%m/%Y', 
        errors='coerce'
    )

    count += (~df[date_col].isna() | mask_nan_eu).sum()

    print(f'Datos manipulados: {count}')

    return df


bike_sharing_df = convert_date_format(bike_sharing_df, 'dteday')


## HOUR
##
def clean_hour(df, hour_col, date_col):
    CUTOFF_DATE = pd.to_datetime('2012-12-31')
    CUTOFF_HR = 23.0 

    count = 0

    df[hour_col] = pd.to_numeric(df[hour_col], errors='coerce') # cast string a numérica para convertir valores corruptos a NaN
    df[hour_col].fillna(-1.0, inplace=True)
    df[hour_col] = df[hour_col].astype(float)

    for i in range(1, len(df)):

        current_date = df.loc[i, date_col]
        current_hr = df.loc[i, hour_col]

        prev_date = df.loc[i - 1, date_col]
        prev_hr = df.loc[i - 1, hour_col]

        if prev_date == CUTOFF_DATE and prev_hr == CUTOFF_HR: # poka-yoke en la última hora del último día de 2012
            break

        expected_hr = (prev_hr + 1) % 24 # formato 24 horas

        out_of_range_flag = (current_hr < 0) or (current_hr > 23)

        if out_of_range_flag:
            # Caso 1: si el valor es 1) corrupto o 2) fuera del rango de 24 hr, imputamos el valor secuencial esperado
            df.loc[i, hour_col] = expected_hr

            count += 1

            if expected_hr == 0:
                if pd.notna(prev_date):
                    df.loc[i, date_col] = prev_date + pd.Timedelta(days=1)

        elif current_hr != expected_hr:
            # Caso 2: si es un salto en la secuencia numérica (ej. 1 -> 4) y NO es un cambio de día, no se realiza imputación
            pass

        df[hour_col] = df[hour_col].astype(int)

        mask_final_outlier = (df[hour_col] < 0) | (df[hour_col] > 23)

        if mask_final_outlier.sum() > 0:
            indices_to_fix = df[mask_final_outlier].index

            for idx in indices_to_fix:
                if idx > 0:
                    prev_hr_fixed = df.loc[idx - 1, hour_col]
                    prev_date_fixed = df.loc[idx - 1, date_col]

                    expected_hr_current = (prev_hr_fixed + 1) % 24

                    df.loc[idx, hour_col] = expected_hr_current

                    count += 1

                    if expected_hr_current == 0: # corregir dteday si hay salto de día
                        if pd.notna(prev_date_fixed):
                            df.loc[idx, date_col] = prev_date_fixed + pd.Timedelta(days=1)
                else:
                   df.loc[idx, hour_col] = 0.0

        print(f'Datos manipulados: {count}')

        df[hour_col] = df[hour_col].astype(int)

        return df


bike_sharing_df = clean_hour(bike_sharing_df, 'hr', 'dteday')


## DATE DAY
#
def impute_date(df, date_col, hour_col):
    count = 0

    mask_to_impute = df[date_col].isna() & \
                     df[hour_col].notna()

    indixes_to_impute = df[mask_to_impute].index

    for idx in indixes_to_impute:
        if idx > 0:

            current_hr = df.loc[idx, hour_col]
            prev_idx = idx - 1
            prev_dteday = df.loc[prev_idx, date_col]
            prev_hr = df.loc[prev_idx, hour_col]

            if pd.isna(prev_dteday):
                continue # si no hay fecha anterior, se omite

            sequential_jump_flag = (current_hr == (prev_hr + 1)) or (prev_hr == 23 and current_hr == 0)

            if sequential_jump_flag:

                if current_hr == 0:
                    new_date = prev_dteday + pd.Timedelta(days=1)

                else:
                    new_date = prev_dteday

                df.loc[idx, date_col] = new_date
                count += 1

    print(f'Datos manipulados: {count}')

    return df


bike_sharing_df = impute_date(bike_sharing_df, 'dteday', 'hr')


## YEAR
##
def clean_year(df, year_col, date_col):
    YEAR_MAPPING = {2011: 0.0, 2012: 1.0}

    count = 0

    count_before_fillna = df[year_col].isna().sum()
    df[year_col].fillna(df[date_col].dt.year.map(YEAR_MAPPING), inplace=True)
    count += count_before_fillna - df[year_col].isna().sum()

    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')

    mask_corrupt_year = (df[year_col].isna()) | \
                    (~df[year_col].isin([0.0, 1.0])) # rango [0,1] para años (2011, 2012)
    count += mask_corrupt_year.sum() 

    imputed_yr_values = df.loc[mask_corrupt_year, date_col].dt.year.map(YEAR_MAPPING)

    df.loc[mask_corrupt_year, year_col] = imputed_yr_values

    print(f'Datos manipulados: {count}')

    return df


bike_sharing_df = clean_year(bike_sharing_df, 'yr', 'dteday')


## MONTH
##
def clean_month(df, month_col, date_col):
    count = 0

    count_before_fillna = df[month_col].isna().sum() 
    df[month_col].fillna(df[date_col].dt.month, inplace=True)
    count += count_before_fillna - df[month_col].isna().sum()

    df[month_col] = pd.to_numeric(df[month_col], errors='coerce')

    mask_corrupt_month = (df[month_col].isna()) | \
                        (df[month_col] < 1) | \
                        (df[month_col] > 12) # rango [1-12] para meses
    count += mask_corrupt_month.sum()

    df.loc[mask_corrupt_month, month_col] = df.loc[mask_corrupt_month, date_col].dt.month.astype(float)

    print(f'Datos manipulados: {count}')

    return df


bike_sharing_df = clean_month(bike_sharing_df, 'mnth', 'dteday')


## WEEKDAY
##
def clean_weekday(df, weekday_col, date_col):
    count = 0

    count_before_fillna = df[weekday_col].isna().sum()
    df[weekday_col].fillna(df[date_col].dt.weekday, inplace=True)
    count += count_before_fillna - df[weekday_col].isna().sum()

    df[weekday_col] = pd.to_numeric(df[weekday_col], errors='coerce')

    mask_corrupt_weekday = (bike_sharing_df[weekday_col].isna()) | \
                        (bike_sharing_df[weekday_col] < 0) | \
                        (bike_sharing_df[weekday_col] > 6) # rango [0-6] para días de la semana
    count += mask_corrupt_weekday.sum()

    imputed_weekday_values = bike_sharing_df.loc[mask_corrupt_weekday, date_col].dt.weekday.astype(float)

    bike_sharing_df.loc[mask_corrupt_weekday, weekday_col] = imputed_weekday_values

    print(f'Datos manipulados: {count}')

    return df


bike_sharing_df = clean_weekday(bike_sharing_df, 'weekday', 'dteday')


## HOLIDAY
#
def clean_holiday(df, holiday_col, date_col):
    count = 0

    us_holidays = holidays.US(years=[2011, 2012])

    def get_us_holiday(date):
        if pd.isna(date):
            return np.nan

        return 1.0 if date in us_holidays else 0.0


    df[holiday_col] = pd.to_numeric(df[holiday_col], errors='coerce')

    mask_corrupt_holiday = (df[holiday_col].isna()) | \
                        (~df[holiday_col].isin([0.0, 1.0]))
    count += mask_corrupt_holiday.sum()

    imputed_holiday_values = df.loc[mask_corrupt_holiday, date_col].apply(get_us_holiday)

    df.loc[mask_corrupt_holiday, holiday_col] = imputed_holiday_values
    df[holiday_col].astype(int)

    print(f'Datos manipulados: {count}')

    return df


bike_sharing_df = clean_holiday(bike_sharing_df, 'holiday', 'dteday')


## WORKING DAY
#
def clean_workday(df, workday_col, weekday_col, holiday_col):
    count = 0

    df[workday_col] = pd.to_numeric(df[workday_col], errors='coerce')

    mask_corrupt_workingday = (df[workday_col].isna()) | \
                            (~df[workday_col].isin([0.0, 1.0]))
    count += mask_corrupt_workingday.sum()

    imputed_workingday_values = df.loc[mask_corrupt_workingday, weekday_col].apply(
        lambda x: 0.0 if x in [0.0, 6.0] else 1.0
    ) # si 'weekday' está en [0, 6], 'workingday' debe ser 0, sino debe ser 1

    df.loc[mask_corrupt_workingday, workday_col] = imputed_workingday_values
    count += (df.loc[df[holiday_col] == 1.0, workday_col] != 0.0).sum()

    df.loc[df[holiday_col] == 1.0, workday_col] = 0.0 # si es día festivo (holiday=1), NO puede ser día laboral (workingday=0)

    print(f'Datos manipulados: {count}')

    return df


bike_sharing_df = clean_workday(bike_sharing_df, 'workingday', 'weekday', 'holiday')


## SEASON
#
def map_season(current_date):

    if pd.isna(current_date):
        return np.nan

    month_day = (current_date.month, current_date.day)

    # 1.0 == invierno: 21-Dic al 20-Mar
    if (month_day >= (12, 21)) or (month_day <= (3, 20)):
        return 1.0 

    # 2.0 == primavera: 21-Mar al 20-Jun
    elif (month_day >= (3, 21)) and (month_day <= (6, 20)):
        return 2.0

    # 3.0 == verano: 21-Jun al 22-Sep
    elif (month_day >= (6, 21)) and (month_day <= (9, 22)):
        return 3.0

    # 4.0 == otoño): 23-Sep al 20-Dic
    elif (month_day >= (9, 23)) and (month_day <= (12, 20)):
        return 4.0

    return np.nan

def clean_season(df, season_col, date_col):
    count = 0

    df[season_col] = pd.to_numeric(df[season_col], errors='coerce')

    mask_corrupt_season = (df[season_col].isna()) | \
                        (df[season_col] < 1.0) | \
                        (df[season_col] > 4.0) | \
                        (df[season_col] % 1 != 0)
    count += mask_corrupt_season.sum()

    imputed_season_values = df.loc[mask_corrupt_season, date_col].apply(map_season)

    df.loc[mask_corrupt_season, season_col] = imputed_season_values

    print(f'Datos manipulados: {count}')

    return df


bike_sharing_df = clean_season(bike_sharing_df, 'season', 'dteday')


## INSTANT
#
def impute_instant(df, instant_col, date_col, hour_col):
    count = 0

    mask_instant_nan = df[instant_col].isna()
    indexes_to_impute = df[mask_instant_nan].index

    for idx in indexes_to_impute:

        if idx > 0 and idx < len(df) - 1:

            prev_idx = idx - 1
            prev_instant = df.loc[prev_idx, instant_col]
            prev_date = df.loc[prev_idx, date_col]
            prev_hr = df.loc[prev_idx, hour_col]

            next_idx = idx + 1
            next_instant = df.loc[next_idx, instant_col]
            next_date = df.loc[next_idx, date_col]
            next_hr = df.loc[next_idx, hour_col]

            current_date = df.loc[idx, date_col]
            current_hr = df.loc[idx, hour_col]

            if pd.isna(prev_instant) or pd.isna(next_instant):
                continue

            if next_instant - prev_instant != 2:
                continue

            expected_hr_current = (prev_hr + 1) % 24
            sequential_hr_flag = current_hr == expected_hr_current

            if expected_hr_current == 0:
                expected_date = prev_date + pd.Timedelta(days=1)
                sequential_date_flag = current_date == expected_date
            else:
                sequential_date_flag = current_date == prev_date

            if sequential_hr_flag and sequential_date_flag:                
                imputed_value = prev_instant + 1.0
                df.loc[idx, instant_col] = imputed_value
                count += 1

    print(f'Datos manipulados: {count}')

    return df


def get_max_records():
    HOURS_PER_DAY = 24

    days_in_2011 = 366 if calendar.isleap(2011) else 365
    days_in_2012 = 366 if calendar.isleap(2012) else 365    

    total = (days_in_2011 + days_in_2012) * HOURS_PER_DAY

    return total


bike_sharing_df = impute_instant(bike_sharing_df, 'instant', 'dteday', 'hr')


def crosscheck_duplicates(df): # eliminar registros duplicados con menor cantidad de valores faltantes
    key_columns = ['instant', 'dteday', 'hr']
    rows_before_cleanup = df.shape[0]

    # corección: hay registros con un valor en 'instant' duplicados (1 registro con valores en dteday y otro con valores faltantes)
    df_valid_dates = df.dropna(subset=['dteday']).copy()

    df_valid_dates.drop_duplicates(subset=['instant'], keep='first', inplace=True)

    instant_to_dteday_map = df_valid_dates.set_index('instant')['dteday']

    mask_dteday_nan = df['dteday'].isna()
    df.loc[mask_dteday_nan, 'dteday'] = df.loc[mask_dteday_nan, 'instant'].map(instant_to_dteday_map)

    df['tmp_nan_count'] = df.isnull().sum(axis=1)

    df.sort_values(
        by=key_columns + ['tmp_nan_count'],
        ascending=[True, True, True, True],
        inplace=True
    )

    df.drop_duplicates(subset=key_columns, keep='first', inplace=True)

    df.drop(columns=['tmp_nan_count'], inplace=True, errors='ignore')

    rows_after_cleanup = df.shape[0]
    total_dropped_rows = rows_before_cleanup - rows_after_cleanup

    print(f'Registros eliminados: {total_dropped_rows}')

    return df


bike_sharing_df = crosscheck_duplicates(bike_sharing_df)


def impute_weather_details(df):
    count = 0

    weather_cols = ['temp', 'atemp', 'hum', 'windspeed']

    mask_hum_outlier = (df['hum'] < 0) | (df['hum'] > 100)
    df.loc[mask_hum_outlier, 'hum'] = np.nan

    mask_windspeed_outlier = (df['windspeed'] == 0) | (df['windspeed'] > 60)
    df.loc[mask_windspeed_outlier, 'windspeed'] = np.nan

    nan_count_knn = df[weather_cols].isna().sum().sum()

    imputer = KNNImputer(n_neighbors=5)

    df[weather_cols] = imputer.fit_transform(df[weather_cols])

    nan_count_weathersit = df['weathersit'].isna().sum() 

    df['weathersit'].fillna(method='ffill', inplace=True)
    df['weathersit'].fillna(method='bfill', inplace=True)

    count = nan_count_knn + nan_count_weathersit

    df['weathersit'] = df['weathersit'].astype(int)

    print(f'Datos manipulados: {count}')

    return df


bike_sharing_df = impute_weather_details(bike_sharing_df)


def impute_bikes_total_count(df):
    count = 0

    count_cols = ['cnt', 'registered', 'casual']
    rows_initial = df.shape[0]

    mask_not_nan = df[count_cols].notna().all(axis=1)

    # registered > cnt
    mask_reg_too_high = mask_not_nan & (df['registered'] > df['cnt'])
    count += mask_reg_too_high.sum()
    df.loc[mask_reg_too_high, 'registered'] = df['cnt'] - df['casual']

    # casual > cnt
    mask_casual_too_high = mask_not_nan & (df['casual'] > df['cnt'])
    count += mask_casual_too_high.sum()
    df.loc[mask_casual_too_high, 'casual'] = df['cnt'] - df['registered']

    # correción de 'cnt' donde la suma no cuadra (ej: cnt=13, reg=10, cas=0),
    mask_incorrect_sum = df[count_cols].notna().all(axis=1) & (df['cnt'] != (df['registered'] + df['casual']))
    count += mask_incorrect_sum.sum()
    df.loc[mask_incorrect_sum, 'cnt'] = df['registered'] + df['casual']

    # cnt = registered + casual
    mask_impute_cnt = df['cnt'].isna() & df['registered'].notna() & df['casual'].notna()
    count += mask_impute_cnt.sum()
    df.loc[mask_impute_cnt, 'cnt'] = df['registered'] + df['casual']

    # registered = cnt - casual
    mask_impute_registered = df['registered'].isna() & df['cnt'].notna() & df['casual'].notna()
    count += mask_impute_registered.sum()
    df.loc[mask_impute_registered, 'registered'] = df['cnt'] - df['casual']

    # casual = cnt - registered
    mask_impute_casual = df['casual'].isna() & df['cnt'].notna() & df['registered'].notna()
    count += mask_impute_casual.sum()
    df.loc[mask_impute_casual, 'casual'] = df['cnt'] - df['registered']

    # ajustar valores negativos a NaN (resultado de restas o corrupción)
    mask_negatives = (df['registered'] < 0) | (df['casual'] < 0) | (df['cnt'] < 0)
    count += mask_negatives.sum()
    df.loc[mask_negatives, count_cols] = np.nan

    # eliminar filas que aún tienen NaN (2+ valores NaN, o se volvieron negativos)
    df.dropna(subset=count_cols, inplace=True)
    rows_dropped_total = rows_initial - df.shape[0]

    print(f'Datos manipulados (celdas): {count}')
    print(f'Registros eliminados: {rows_dropped_total}')

    return df


bike_sharing_df = impute_bikes_total_count(bike_sharing_df)


def rebuild_instant(df, instant_col):
    count = 0

    sort_key = ['dteday', 'hr']
    df.sort_values(by=sort_key, inplace=True, ignore_index=True)

    count = df.shape[0] 

    new_instant = df.index + 1

    df[instant_col] = new_instant.astype(int)

    max_instant = df[instant_col].max()

    print(f'Datos manipulados (celdas): {count}')

    return df


bike_sharing_df = rebuild_instant(bike_sharing_df, 'instant')


# # Verificación de datos post-limpieza:

def get_NaN_values(df):
    print(f'Valores NaN:\n')
    print(df.isna().sum())


get_NaN_values(bike_sharing_df)


bike_sharing_df.shape


def get_distribution_summary(df, num_cols):
    summary_data = []

    for col in num_cols:
        skew = df[col].skew()

        if skew > 0.5:
            skew_interpret = 'Sesgo Derecho (Positivo)'
        elif skew < -0.5:
            skew_interpret = 'Sesgo Izquierdo (Negativo)'
        else:
            skew_interpret = 'Simétrico/Ligero'


        kurtosis = df[col].kurtosis()

        if kurtosis > 0.5:
            kurtosis_interpret = 'Leptocúrtica (Colas Pesadas)'
        elif kurtosis < -0.5:
            kurtosis_interpret = 'Platicúrtica (Colas Ligeras)'
        else:
            kurtosis_interpret = 'Mesocúrtica (Normal)'

        summary_data.append({
            'Variable': col,
            'Skewness': skew,
            'Kurtosis': kurtosis,
            'Tipo de Sesgo': skew_interpret,
            'Forma de la Distribucion': kurtosis_interpret
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df[['Skewness', 'Kurtosis']] = summary_df[['Skewness', 'Kurtosis']].round(2)

    return summary_df


get_distribution_summary(bike_sharing_df, num_cols)


# # Versionamiento de datos:

bike_sharing_df.to_csv('../data/processed/bike_sharing_cleaned.csv', index=False)

