# Parkinson's Disease Detection Dashboard

Dark-themed Streamlit dashboard inspired by the interface style in the reference PDF. It adapts the layout to the Parkinson's disease voice-classification research project.

## Features
- Sidebar navigation and Home overview
- Dataset analysis and class distribution
- Voice-feature prediction demonstration
- Model comparison for Logistic Regression, SVM, KNN, Random Forest, and Gradient Boosting
- Performance metrics and runtime table
- Confusion matrix from actual held-out predictions
- Research results tables with CSV download buttons

## Run on Windows / VS Code

Open this folder in VS Code. In the terminal, run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If a `.venv` already exists in the parent folder, use its Python executable or create the environment in this folder.

On first launch, the app downloads the UCI Parkinson's voice dataset to `data/parkinsons.data`, so an internet connection is needed.

## Existing results CSVs
Copy your generated CSV files into this project's `results` folder to make the Research Results page show those exact saved tables. The app also computes current test metrics live for the dashboard.

## Research limitations
This is a voice-only baseline, not a multimodal implementation. Do not join unrelated voice and gait cohorts by row order or claim multimodal performance without compatible participant-linked observations. Predictions are for research demonstration only and are not a clinical diagnosis.
