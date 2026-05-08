# Workflow CI — Muhammad Fadhil Rizki

**Kriteria 3** — Kelas Membangun Sistem Machine Learning (Dicoding).

- **Nama Siswa:** Muhammad Fadhil Rizki
- **Username Dicoding:** `fadhilspooky`
- **DagsHub / Docker Hub:** `thefanaticid`
- **Docker Image:** `thefanaticid/california-housing-gbr`

## Struktur Repository

```
Workflow-CI/
├── .github/workflows/ci.yml          ← CI: train → artifact → Docker push
├── MLProject/
│   ├── MLProject                     ← MLflow project definition
│   ├── conda.yaml                    ← Environment dependencies
│   └── modelling.py                  ← Training script (parameterized)
├── housing_preprocessing/            ← Dataset siap latih (dari Kriteria 1)
└── README.md
```

## Cara Kerja CI (`ci.yml`)

Workflow ter-trigger pada **push ke main**, **pull request ke main**, dan **manual dispatch**.

| Step | Keterangan |
|---|---|
| 1 | Checkout repo |
| 2 | Setup Python 3.12.7 + cache pip |
| 3 | Install dependencies dari `conda.yaml` |
| 4 | Init DagsHub tracking URI → `MLFLOW_TRACKING_URI` di env |
| 5 | `mlflow run MLProject/ --env-manager=local` |
| 6 | Upload `mlruns/` sebagai GitHub artifact (retensi 30 hari) |
| 7 | `mlflow models build-docker -m "runs:/<run_id>/model"` |
| 8 | Push ke Docker Hub: `thefanaticid/california-housing-gbr` |
| 9 | Tulis job summary ke GitHub |

## Secrets yang Harus Dikonfigurasi di GitHub

Buka **repo Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|---|---|
| `DAGSHUB_TOKEN` | Token DagsHub dari Settings → Tokens |
| `DOCKERHUB_USERNAME` | `thefanaticid` |
| `DOCKERHUB_TOKEN` | Access Token dari Docker Hub Settings |

## Menjalankan Secara Lokal

```bash
# Install dependencies
pip install mlflow==2.19.0 dagshub==0.7.0 scikit-learn==1.5.2 pandas==2.2.3 \
            numpy==1.26.4 matplotlib==3.9.2 joblib==1.4.2

# Set env DagsHub (opsional, tanpa ini fallback ke mlflow lokal)
export DAGSHUB_REPO_OWNER=thefanaticid
export DAGSHUB_REPO_NAME=Workflow-CI
export DAGSHUB_TOKEN=<your_token>

# Jalankan
python3 -m mlflow run MLProject/ --env-manager=local

# Build Docker (opsional, butuh docker engine)
RUN_ID=<run_id_dari_output>
mlflow models build-docker -m "runs:/${RUN_ID}/model" -n "thefanaticid/california-housing-gbr"
```

## Parameter MLProject

Parameter bisa di-override via `workflow_dispatch` input atau `-P` flag:

| Parameter | Default | Deskripsi |
|---|---|---|
| `n_estimators` | `300` | Jumlah tree GBR |
| `learning_rate` | `0.1` | Learning rate |
| `max_depth` | `5` | Kedalaman maksimum tree |
| `cv_folds` | `3` | Jumlah fold cross-validation |
| `data_dir` | `../housing_preprocessing` | Path ke dataset |

## Hasil Lokal (Verified)

- Model: GradientBoostingRegressor (best params: n=300, lr=0.1, depth=5)
- **RMSE Test: 0.4544** | MAE: 0.3044 | R²: 0.8416
- Artefak: feature_importance.png, residuals_plot.png, predictions_vs_actual.png, cv_results.csv
