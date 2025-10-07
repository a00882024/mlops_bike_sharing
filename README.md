# MLOps Equipo 26: Bike Sharing


Este repositorio contiene los "notebooks", "scripts" y artefactos de mlflow relacionados al proyecto

## Arquitectura del Proyecto

```
┌─────────────────────────────────────────────────────────────────┐
│              Proyecto MLOps de Bike Sharing                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│   Repo GitHub    │◄────────┤  Entorno Local   │
│ (Código/Config)  │────────►│     (.venv)      │
└──────────────────┘         └────────┬─────────┘
                                      │
                                      │
                         ┌────────────┼────────────┐
                         │            │            │
                         ▼            ▼            ▼
              ┌──────────────┐ ┌──────────┐ ┌──────────────┐
              │     DVC      │ │ MLflow   │ │  Notebooks   │
              │  (Remoto S3) │ │   UI     │ │ (Jupyter/VS) │
              └──────┬───────┘ └────┬─────┘ └──────────────┘
                     │              │
                     │              │
              ┌──────▼───────┐      │
              │              │      │
              │  Datos/      │      │
              │  Artefactos  │      │
              │              │      │
              │  • data/     │      │
              │  • mlruns/   │◄─────┘
              │    mlartifacts/
              └──────────────┘


Flujo de Datos:
───────────────
1. dvc pull  ──►  S3 ──► data/ & mlruns/mlartifacts/ locales
2. Notebooks ──►  Entrenar modelos ──► Registro en MLflow
3. dvc add   ──►  Rastrear nuevos datos/modelos
4. dvc push  ──►  Subir a S3
5. git push  ──►  Versionar archivos .dvc y código


Componentes Clave:
──────────────────
├── data/               (Datasets rastreados con DVC)
├── mlruns/            (Experimentos de MLflow)
│   └── mlartifacts/   (Modelos rastreados con DVC)
├── notebooks/         (Notebooks de Jupyter)
├── src/              (Código fuente)
└── .dvc/             (Configuración de DVC)
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
