# MLOps Equipo 26: Bike Sharing


Este repositorio contiene los "notebooks", "scripts" y artefactos de mlflow relacionados al proyecto


## Setup inicial 

Inicializa el `environment` e instala las dependencias

```bash
python -m venv .venv

pip install -r requirements.txt
```

Configura DVC para obtener los datos 

```bash
dvc remote modify --local origin \
  access_key_id ACCESS_KEY_ID

dvc remote modify --local origin \
  secret_access_key SECRET_KEY
```

Obten los datos 

```
dvc pull
```
