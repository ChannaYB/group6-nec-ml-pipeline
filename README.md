# NEC ML Pipeline - Smart Power Plant Selection

End-to-end machine learning pipeline for optimizing power plant selection based on generation cost predictions.

**Institution:** Keele University Business School  
**Module:** MAN-40389 - Advanced Data Analytics and Machine Learning  
**Assessment:** Group Assignment - Group 06
**Member Numbers:** 25009641, 24038643, 25027572, 25037027 & 25042861

---

##  Executive Summary

An automated system that predicts generation costs and selects optimal power plants for varying demand scenarios:

- **Dataset:** 26,560 samples (415 demand scenarios × 64 power plants)
- **Final Model:** Tuned Random Forest (200 estimators, unlimited depth)
- **Performance:** 57.83% selection error (42.17% accuracy)
- **Improvement:** 4.0% reduction in selection error vs baseline

---

##  Quick Start

### Prerequisites
```
Python 3.8+
8GB RAM minimum (16GB recommended)
```

### Installation

**1. Setup Environment**
```bash
# Open project folder
Open the project folder in VS Code (or any IDE).
Ensure the terminal is opened in the project root directory.
If the project was downloaded as a ZIP file, ensure you open the inner project directory that contains `requirements.txt` and `main.py`.

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

**Single Command Execution:**
```bash
# Full pipeline with complete hyperparameter tuning (~15-20 min)
python main.py

# Quick mode with reduced parameter grid (~5-8 min)
python main.py --quick

# Baseline evaluation only, no tuning (~2-3 min)
python main.py --no-tune

# Minimal console output
python main.py --quick --quiet
```

### Expected Output
```
======================================================================
NEC ML PIPELINE - COMPLETE EXECUTION
======================================================================

[Step 1/8] Data Loading 
[Step 2/8] Preprocessing 
[Step 3/8] Baseline Model 
[Step 4/8] Baseline Evaluation 
[Step 5/8] Hyperparameter Tuning 
[Step 6/8] Tuned Model Evaluation 
[Step 7/8] Model Comparison 
[Step 8/8] Visualization 

FINAL RESULTS:
  Baseline: 60.24% selection error
  Tuned:    57.83% selection error
  Improvement: 4.0%

 Results saved to results/ directory
======================================================================
```

---

##  Project Structure
```
nec-ml-pipeline/
├── data/
│   ├── raw/                    # Original CSV files
│   └── processed/              # train.csv, test.csv
├── src/                        # Pipeline modules
├── tests/                      # Unit tests
├── results/                    # Generated outputs
│   ├── evaluation_reports/     # Performance summaries
│   ├── selection_tables/       # Prediction results (CSV)
│   └── plots/                  # Visualizations (PNG)
├── models/                     # Saved models (.pkl)
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

##  Testing

**Run all unit tests:**
```bash
# Test individual modules
python -m pytest -q tests/test_data_ingestion.py
python -m pytest -q tests/test_preprocessing.py
python -m pytest -q tests/test_models.py
python -m pytest -q tests/test_evaluation.py
python -m pytest -q tests/test_tuning.py
```

---

##  Key Results

| Metric | Baseline RF | Tuned RF | Improvement |
|--------|-------------|----------|-------------|
| Test RMSE | $13.01 | $12.90 | 0.9% |
| Test R² | 0.332 | 0.344 | 3.5% |
| Selection Error | 60.24% | 57.83% | 4.0% |
| Correct Selections | 33/83 | 35/83 | +2 |

**Tuned Model Configuration:**
- n_estimators: 200
- max_depth: None
- max_features: sqrt
- min_samples_split: 5
- min_samples_leaf: 2

---

##  Environment Details

**Python Dependencies:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

**System Requirements:**
- CPU: 4+ cores recommended for parallel processing
- RAM: 8GB minimum, 16GB for full grid search
- Storage: ~500MB for data and results

**Tested On:**
- Windows 10/11
- macOS 12+
- Ubuntu 20.04+

---

##  Output Files

**After execution, find results in:**
```
results/
├── evaluation_reports/
│   ├── evaluation_Baseline_RF_YYYYMMDD_HHMMSS.txt
│   └── evaluation_Tuned_RF_YYYYMMDD_HHMMSS.txt
├── selection_tables/
│   ├── selection_Baseline_RF_Test_YYYYMMDD_HHMMSS.csv
│   └── selection_Tuned_RF_Test_YYYYMMDD_HHMMSS.csv
└── plots/
    ├── dashboard_Baseline_RF_YYYYMMDD_HHMMSS.png
    ├── dashboard_Tuned_RF_YYYYMMDD_HHMMSS.png
    ├── model_comparison_YYYYMMDD_HHMMSS.png
    ├── logo_cv_Baseline_RF_YYYYMMDD_HHMMSS.png
    └── selection_analysis_*.png

models/
├── tuned_rf_YYYYMMDD_HHMMSS.pkl
└── baseline_rf_YYYYMMDD_HHMMSS.pkl (if saved)
```

---

##  Troubleshooting

**Issue: ModuleNotFoundError**
```bash
# Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

**Issue: Out of Memory**
```bash
# Use quick mode with reduced grid
python main.py --quick
```

**Issue: Slow Execution**
```bash
# Skip hyperparameter tuning
python main.py --no-tune
```

**Issue: Import Errors**
```bash
# Run from project root directory
cd nec-ml-pipeline
python main.py
```

---

##  Documentation

**Code Documentation:**
- All modules include comprehensive docstrings
- Function signatures follow Google style
- Type hints provided for key functions

**For detailed technical information:**
- See `Assessment_2_Brief.pdf` for project requirements
- See `Technical_Summary.docx` for methodology details
- See `Presentation.pptx` for visual overview

---

##  Team Contributions

**Pipeline Components:**
- Data Pipeline & Validation
- Preprocessing & Feature Engineering  
- Model Training & Custom Scorer
- Evaluation & Visualization
- Hyperparameter Tuning & Integration

**Full integration tested and verified.**

---

##  Citation
```bibtex
@software{nec_ml_pipeline_2025,
  title = {NEC ML Pipeline: Smart Power Plant Selection},
  year = {2025},
  institution = {Keele University Business School},
  course = {MAN-40389}
}
```

---
