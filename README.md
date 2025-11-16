# MLOps Equipo 26: Bike Sharing


Este repositorio contiene los "notebooks", "scripts" y artefactos de mlflow relacionados al proyecto

## Arquitectura del Proyecto

```
┌──────────────────────────────────────────────────────────────────────┐
│                 Proyecto MLOps de Bike Sharing                       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌───────────────────────────────────────┐
│   Repo GitHub    │◄────────┤      Entorno Local (.venv)            │
│ (Código/Config)  │────────►│                                       │
└──────────────────┘         └────────────┬──────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │   DVC Pipeline   │  │  MLflow Server   │  │   Notebooks      │
         │  (params.yaml)   │  │   (Tracking)     │  │ (Exploración)    │
         └────────┬─────────┘  └────────┬─────────┘  └──────────────────┘
                  │                     │
                  │ ┌───────────────────┘
                  │ │
                  ▼ ▼
         ┌─────────────────────────────────────────┐
         │        Pipeline Stages (DVC)            │
         │                                         │
         │  1. data_cleaning                       │
         │  2. feature_engineering                 │
         │  3. train_* (5 modelos)                 │
         │     • RandomForest                      │
         │     • ElasticNet                        │
         │     • SVR                               │
         │     • XGBoost                           │
         │     • LightGBM                          │
         └────────────────┬────────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐
    │   data/   │  │  models/  │  │  mlruns/  │
    │  (DVC+S3) │  │  (PKL)    │  │ (MLflow)  │
    └───────────┘  └─────┬─────┘  └───────────┘
                         │
                         │
                         ▼
              ┌────────────────────┐
              │   FastAPI Server   │
              │   (api_server.py)  │
              │                    │
              │  • /predict        │
              │  • Docker support  │
              └──────────┬─────────┘
                         │
              ┌──────────┴─────────┐
              │                    │
              ▼                    ▼
       ┌─────────────┐      ┌─────────────┐
       │ Monitoring  │      │    Tests    │
       │ (Drift)     │      │  (Pytest)   │
       └─────────────┘      └─────────────┘


Flujo de Trabajo:
─────────────────
1. dvc pull          ──►  Descargar datos/modelos desde S3
2. dvc repro         ──►  Ejecutar pipeline completo
3. Notebooks         ──►  Experimentación y análisis
4. Entrenamiento     ──►  Registro automático en MLflow
5. Modelos           ──►  Guardados en models/ (PKL files)
6. FastAPI           ──►  Servir predicciones
7. Monitoring        ──►  Detectar data drift
8. dvc push/git push ──►  Versionar código y artefactos


Componentes Clave:
──────────────────
├── bike_sharing/         (Paquete Python principal)
│   ├── api_server.py     (FastAPI server)
│   ├── dataset.py        (Limpieza de datos)
│   ├── features.py       (Feature engineering)
│   ├── modeling/         (Entrenamiento y predicción)
│   └── monitoring/       (Data drift detection)
├── data/                 (Datasets rastreados con DVC)
│   ├── raw/              (Datos originales)
│   ├── processed/        (Datos procesados)
│   ├── interim/          (Datos intermedios)
│   └── external/         (Datos externos)
├── models/               (Modelos entrenados .pkl)
├── mlruns/              (Experimentos de MLflow)
├── notebooks/           (Notebooks de Jupyter)
├── tests/               (Unit & Integration tests)
├── dvc.yaml             (Definición del pipeline)
├── params.yaml          (Parámetros de configuración)
├── Dockerfile           (Containerización)
└── .dvc/                (Configuración de DVC)
```

## Reproducibilidad

Parar reproducir los diferentes "stages" del proyecto se utilizó dvc
```

                         +----------------------------------------+             +----------------------------------------+
                         | data/raw/bike_sharing_modified.csv.dvc |             | data/raw/bike_sharing_original.csv.dvc |
                         +----------------------------------------+             +----------------------------------------+
                                                          ****                       *****
                                                              *****              ****
                                                                   ***        ***
                                                                 +---------------+
                                                                 | data_cleaning |
                                                                 +---------------+
                                                                         *
                                                                         *
                                                                         *
                                                              +---------------------+
                                                          ****| feature_engineering |*****
                                                **********    +---------------------+*    **********
                                      **********          *****            *          ******        ***********
                            **********                ****                  *               *****              **********
                      ******                       ***                      *                    ***                     ******
+---------------------+             +-------------------+             +-----------+             +---------------+             +----------------+
| train_random_forest |             | train_elastic_net |             | train_svr |             | train_xgboost |             | train_lightgbm |
+---------------------+             +-------------------+             +-----------+             +---------------+             +----------------+
```


Para observar el diagrama utiliza

```bash
dvc dag
```

Se puede ejecutar el pipeline entero usando 

```bash
dvc repro
```

También se puede ejecutar un solo tipo de modelo usando 

```bash
dvc repro train_xgboost
```

## Baja el repositorio 

```bash
git clone https://github.com/a00882024/mlops_bike_sharing.git
```


## Setup inicial 

Inicializa el `environment` e instala las dependencias

```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

Configura DVC para obtener los datos 

```bash
dvc remote modify --local origin \
  access_key_id ACCESS_KEY_ID

dvc remote modify --local origin \
  secret_access_key SECRET_KEY
```

Obten los datos y modelos

```
dvc pull
```


## Actualizando codigo, modelos y datos 

```bash
git pull
dvc pull
```

Este se asegura que tienes la ultima version del codigo, y que los datos y modelos se obtienen de la `bucket` en S3

## Trabajando con los notebooks

Los notebooks estan en la carpeta `notebooks/`. Puedes abrirlos con `jupyter` o con `VSCode`


### Para trabajar con jupyter

Inicla el servidor de jupyter lab 

```bash
jupyter-lab
``` 

Da click en la url que aparece en la terminal para abrir jupyter lab en tu navegador

### Para trabajar con VSCode

Desde la terminal, abre VSCode en la carpeta del proyecto

```bash
code .
```

Abre los notebooks desde el explorador de archivos en VSCode

## Para contribuir al proyecto 

1. Crea un fork del repositorio en tu cuenta de GitHub

2. Clona tu fork a tu maquina local

3. Realiza los pasos de setup inicial 

4. Crea una rama para tus cambios

```bash
git checkout -b feat/<nombre-de-tu-rama>
```

5. Realiza tus cambios

6. Sube los datos a DVC

```bash
dvc add <ruta-al-archivo>
git add <ruta-al-archivo>.dvc
git commit -m "Agrega datos a DVC"
git push origin feat/<nombre-de-tu-rama>
dvc pusheee
```

7. Sube tus cambios a tu fork en GitHub

8. Realiza un `pull request` a la rama `main` del repositorio original y espera a que alguien del equipo revise y apruebe tus cambios
