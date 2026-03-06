# SME Risk Navigator

SME Risk Navigator is a practical PoC for **IBK Digital (AI Service Planning)** style workflow: quantitative risk scoring by tree model and qualitative memo generation by LLM copilot.

## 1) Problem Definition
- **Goal**: Predict SME credit risk score (PD-like probability) and grade (A-E), then auto-generate review memo/checklists/scenarios.
- **Target users**: credit reviewer, RM (relationship manager), underwriting support teams.
- **Core concept**: 
  - `Score Engine`: XGBoost + calibration for reproducible quantitative scoring.
  - `Explain`: SHAP drivers for evidence.
  - `Copilot`: LLM outputs JSON memo from model facts only.

## 2) Architecture
- **Data Ingestion**: numeric CSV with label column (`target` by default)
- **Modeling**:
  - Baseline: LogisticRegression (`class_weight='balanced'`)
  - Main: XGBoostClassifier (`scale_pos_weight`, early stopping)
  - Calibration: Isotonic regression using validation set (`cv='prefit'`)
- **Explainability**:
  - Global SHAP importance plot
  - Local top drivers per case
- **Copilot**:
  - FACTS builder + single LLM call
  - JSON schema enforcement with Pydantic
  - Retry (up to 2) + safe fallback JSON
  - Copilot is for documentation/checklist generation only, not for risk prediction
- **UI**: Streamlit demo with manual input / CSV row input, prediction, drivers, memo download

## 3) Data Requirements
- Input file: CSV
- Features: numeric columns only (non-numeric auto-dropped)
- Label column: binary (`target` by default, configurable via CLI)
- Missing values: median imputation
- Split: train/val/test = 70/15/15 (stratified)

## 4) Training & Evaluation
Metrics in `reports/metrics.json`:
- AUROC
- AUPRC
- Brier Score
- KS Statistic

Calibration plots:
- `reports/calibration_before.png`
- `reports/calibration_after.png`

## 5) Grade Policy
Default fallback threshold in `src/config.py`:
- A: PD < 0.05
- B: PD < 0.10
- C: PD < 0.20
- D: PD < 0.35
- E: PD >= 0.35

During training, model-specific thresholds are also generated from train calibrated-score quantiles and saved to `models/grade_thresholds.json` to provide a more balanced grade distribution for the current dataset.

## 6) Operational Flow
1. Train model with historical labeled data.
2. Produce PD and Grade for new case.
3. Retrieve SHAP top drivers as evidence.
4. Generate copilot memo/checklist/action JSON.
5. Reviewer edits final memo for human approval.

Manual input values in Streamlit are **raw feature values** (not normalized/scaled). The pipeline uses median imputation only and no standardization.

## 7) Limitations / Model Card Summary
- PoC assumes binary label and numeric features only.
- SHAP direction is model-relative contribution, not causal proof.
- Copilot does not make credit decisions; it only documents/checklists based on facts.
- If API key is missing or JSON validation fails, fallback JSON is returned safely.
- Domain validation, fairness checks, and policy integration are out-of-scope for 1-week PoC.

## 8) Run Instructions
```bash
pip install -r requirements.txt
python -m src.train --data data/raw/dataset.csv --label target --xgb_device cuda:1
python -m src.evaluate --test_data data/processed/test.csv --label target
python -m src.explain --test_data data/processed/test.csv --label target
streamlit run app/streamlit_app.py
pytest -q
```

If your runtime has a CUDA-linked `xgboost` build without GPU driver, training auto-falls back to a CPU tree model (`random_forest_fallback`) and continues to generate artifacts for the demo.

## 9) Deploy (Streamlit Community Cloud)
1. Push this repository to GitHub (done).
2. Go to `https://share.streamlit.io` and sign in with GitHub.
3. Click `New app` and select:
   - Repository: `eyedicamp/SME-Risk-Navigator`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
4. In app `Settings -> Secrets`, add:
   ```toml
   OPENAI_API_KEY="sk-..."
   OPENAI_MODEL="gpt-4.1-mini"
   OPENAI_BASE_URL=""
   ```
5. Deploy.

Notes:
- `runtime.txt` pins Python to 3.10 for compatibility with the current dependency set.
- GitHub Pages cannot host this app directly because it requires Python model inference at runtime.

## 10) Output Artifacts
- Models: `models/xgb_model.joblib`, `models/calibrated_model.joblib`, `models/feature_schema.json`
- Reports: `reports/metrics.json`, calibration plots, SHAP plot, local explanation JSON
- Demo memos: `reports/demo_cases/memo_case_good.json`, `memo_case_borderline.json`, `memo_case_bad.json`

## 11) Screenshot Placeholders
- `[Placeholder] streamlit_main_screen.png`
- `[Placeholder] streamlit_memo_json.png`
- `[Placeholder] calibration_plots.png`
