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


# # Exploratory Data Analysis (EDA):
# - Identificar tipos de datos
# - Funciones base de manipulación de datos (eliminar variables, castear valores, etc.)
# - Estadísticas descriptivas de los datos
# - Identificar valores atípicos & faltantes
# - Gráficas para visualizar la distribución por variable

def get_df_info(df):
    df.info()


def get_EDA(df):
    print(df.describe().T)


def get_outliers(df, col):
    percentile_25 = df[col].quantile(0.25)
    percentile_75 = df[col].quantile(0.75)
    iqr = percentile_75 - percentile_25
    upper_limit = percentile_75 + 1.5 * iqr
    lower_limit = percentile_25 - 1.5 * iqr
    IQR_outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]

    return IQR_outliers


def get_NaN_values(df):
    print(f'Valores NaN:\n')
    print(df.isna().sum())


def get_missing_values_by_col(df, col):
    missing_values_mask = df[col].isna()

    df_missing_values = df[missing_values_mask].copy()

    return df_missing_values


def get_cat_cols(df):
    cat_cols = df.select_dtypes(include = 'object').columns.tolist()

    return cat_cols


def get_num_cols(df):
    num_cols = df.select_dtypes(exclude = 'object').columns.tolist()

    return num_cols


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


def create_barplots(df, cat_cols, n_rows, n_cols):
    """
    Genera gráficas de barras para variables categóricas.

    Args:
        df: dataframe de entrada
        cat_cols: lista de nombres de columnas categóricas
        n_rows: # de filas en el grid de subplots
        n_cols: # de columnas en el grid de subplots
    """

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    plt.subplots_adjust(wspace=0.3, hspace=0.6)

    fig.suptitle('Gráficas de Barras: Variables Categóricas', fontsize=12)

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, col in enumerate(cat_cols):
        if i >= len(axes):
            break

        ax = axes[i]
        sns.countplot(x=df[col], ax=ax, palette='rocket', legend=False)
        ax.set_title(f'Conteo: {col}', fontsize=12)
        ax.set_xlabel(None)
        ax.set_ylabel('Frecuencia')
        ax.tick_params(axis='x', rotation=90) 

    # ocultar espacios innecesarios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()


def create_histograms(df, num_cols, n_rows, n_cols):
    """
    Generar histogramas + KDE para variables numéricas

    Params:
        df: dataframe de entrada
        num_cols: lista de columnas numéricas
        n_rows: # de filas en el grid de subplots
        n_cols: # de columnas en el grid de subplots
    """

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    fig.suptitle('Histogramas: Variables Numéricas', fontsize=12)

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, col in enumerate(num_cols):
        if i >= len(axes):
            break

        ax = axes[i]
        sns.histplot(x=df[col], ax=ax, kde=True, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribución de: {col}', fontsize=12)
        ax.set_xlabel(None)
        ax.set_ylabel('Frecuencia/Densidad')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()


def create_boxplots(df, num_cols, n_rows, n_cols):
    """
    Generar Box Plots para variables numéricas & detectar outliers (IQR)

    Args:
        df: dataframe de entrada
        num_cols: lista de nombres de columnas numéricas
        n_rows: # de filas en el grid de subplots
        n_cols: # de columnas en el grid de subplots
    """

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    fig.suptitle('Box Plots / Detección de Outliers (IQR)', fontsize=12)

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, col in enumerate(num_cols):
        if i >= len(axes):
            break

        ax = axes[i]
        sns.boxplot(x=df[col], ax=ax, showmeans=True)
        ax.set_title(f'Box Plot: {col}', fontsize=12)
        ax.set_xlabel(f'{col}')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()


def create_correlation_heatmap(df, num_cols):
    """
    Generar Heatmap para visualizar la matriz de correlación 
    entre las variables numéricas.

    Args:
        df: dataframe de entrada
        num_cols: lista de nombres de columnas numéricas
        figsize: tupla para el tamaño de la figura (width, height)
    """    
    corr_matrix = df[num_cols].corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        corr_matrix, 
        annot=True,              
        fmt=".2f",               
        cmap='rocket',         
        linewidths=.5,           
        linecolor='black',       
        cbar_kws={'label': 'Coeficiente de Correlación'}
    )

    plt.title('Mapa de Calor de Correlación (Variables Numéricas)', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout() 

    plt.show()


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


get_df_info(bike_sharing_df)


get_EDA(bike_sharing_df)


cast_values(bike_sharing_df)


get_NaN_values(bike_sharing_df)


cols_to_drop = get_cols_to_drop(original_bike_sharing_df, bike_sharing_df)
drop_cols(bike_sharing_df, cols_to_drop)


cat_cols = get_cat_cols(bike_sharing_df)
num_cols = get_num_cols(bike_sharing_df)


print(f'\nAnálisis del sesgo de la distribución (skew) & la forma de la distribución (kurtosis):')
distribution_summary_df = get_distribution_summary(bike_sharing_df, num_cols)
distribution_summary_df


print(f'\nAnálisis de Valores Atípicos:')

for col in num_cols:
    outliers_df = get_outliers(bike_sharing_df, col)

    if len(outliers_df) > 0:
        print(f'\n[{col}]\n')
        print(outliers_df) 
    else:
        print(f'\nNo se encontraron resultados para [{col}]')

    print('_' * 40)


create_histograms(bike_sharing_df, num_cols, 6, 3)


create_boxplots(bike_sharing_df, num_cols, 6, 3)


create_correlation_heatmap(bike_sharing_df, num_cols)

