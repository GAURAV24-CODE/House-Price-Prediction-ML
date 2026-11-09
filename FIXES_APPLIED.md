# Fixes Applied — House Price Prediction

## Fix 1: Training and prediction preprocessing consistency

- `src/predictor.py` now loads the exact `StandardScaler` saved during training.
- Raw user inputs are converted into the saved feature order and scaled before model inference.
- Prediction no longer mutates the caller's input dictionary.
- Missing inputs and invalid year relationships produce clear validation errors.
- `train.py` now promotes the newly trained baseline model to `production_model.pkl` so the production model is not left stale after training.

## Fix 2: Consistent date-based feature engineering

- `src/feature_engineering.py` now derives `sale_year` and `sale_month` from the sale date.
- `house_age` and `renovation_age` are calculated relative to the property's sale year, not the machine's current calendar year.
- Training and inference now use the same formulas.
- `src/trainer.py` expects the shared feature engineering step to run before training.

## Validation performed

- Re-trained all eight baseline regression models using the updated features.
- Regenerated the scaler and feature-column artifacts.
- Promoted the newly trained best baseline model to the production model artifact.
- Ran a sample raw-input prediction successfully.
- Added `tests/test_pipeline.py`; all 4 tests passed.

## Run locally

```bash
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

Run the focused pipeline tests with:

```bash
python -m pytest tests/test_pipeline.py -q
```

Note: run `train.py` before launching the app whenever the feature engineering or training data changes. The existing tuned-model artifact has not been regenerated in this first fix pass; use the newly trained baseline production model until hyperparameter tuning is rerun.
