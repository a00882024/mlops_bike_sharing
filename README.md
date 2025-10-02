# MLOps Equipo 26: Bike Sharing


Este repositorio contiene los "notebooks", "scripts" y artefactos de mlflow relacionados al proyecto

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

1. Crea una rama nueva para tu contribucion

```bash
git checkout -b feat/<nombre-de-tu-rama>
```

2. Asegura de tener corriendo mlflow

```bash
mlflow ui
```

3. Realiza tus cambios

4. Asegurate de subir los datos y modelos a DVC

```bash
dvc add <ruta-al-archivo>
git add <ruta-al-archivo>.dvc
git commit -m "Agrega datos/modelos a DVC"
git push origin feat/<nombre-de-tu-rama>
dvc pusheee
```

5. Realiza un `pull request` a la rama `main` y espera a que alguien del equipo revise y apruebe tus cambios

6. Una vez aprobado, haz `merge` de tu rama a `main` y elimina tu rama
