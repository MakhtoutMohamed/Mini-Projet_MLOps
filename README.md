# mlops-ml-project (baseline)

Mini-projet **ML & Git** (baseline reproductible) :
- lecture d’une config **YAML**
- entraînement d’un modèle simple (Iris)
- génération d’artefacts minimaux

> Dataset par défaut : **Iris** (scikit-learn). Une extension CSV est prévue via `data.kind: csv`.

## Structure

```
mlops-ml-project/
  README.md
  requirements.txt
  .gitignore
  config/
    train.yaml
  src/
    __init__.py
    data.py
    features.py
    model.py
  scripts/
    train.py
    evaluate.py
  tests/
    test_config.py
  notebooks/
  data/        # ignoré par git
  artifacts/   # ignoré par git
```

## Installation

### Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Entraînement

```bash
python scripts/train.py
```

## Évaluation

```bash
python scripts/evaluate.py
```

## Artefacts générés

Après `train.py` puis `evaluate.py`, tu dois avoir :
- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`
- `artifacts/report.json`

## Tests

```bash
pytest -q
```

## Git (workflow conseillé)

Exemple de workflow **main/dev/feature** :

```bash
git init

# Commit 1: .gitignore
git add .gitignore
git commit -m "chore: add .gitignore for ML project"

# Commit 2: baseline (code + config + docs + tests)
git add README.md requirements.txt config/ src/ scripts/ tests/
git commit -m "init: baseline ML project (train/eval + artifacts)"

# Branches
git checkout -b dev
git checkout -b feature/preprocessing
```

Dans `feature/preprocessing`, tu peux enrichir le pipeline dans `src/features.py`
(ex: clipping via `FunctionTransformer`), relancer `python scripts/train.py`, puis merge :

```bash
git checkout dev
git merge feature/preprocessing
```

## Tag release baseline

```bash
git checkout main
git merge dev
git tag -a v0.1.0 -m "Baseline model (train/eval OK)"
```
