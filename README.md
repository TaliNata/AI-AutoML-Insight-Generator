# AI AutoML + Insight Generator

End-to-end AutoML pipeline with automatic model selection and LLM-powered business insights.

## Overview

AI AutoML + Insight Generator is a modular machine learning system that:

- Accepts any CSV dataset
- Automatically detects task type (regression / classification)
- Performs preprocessing (encoding + imputation)
- Runs cross-validation across multiple models
- Selects the best model automatically
- Evaluates performance
- Generates feature importance
- Produces LLM-based analytical report
- Saves model and metrics artifacts
- Supports Docker-based reproducible execution

This project demonstrates ML engineering + Prompt engineering + production mindset.

---

## Architecture

```
main.py
  ↓
pipeline.py (orchestrator)
  ↓
data_loader.py
preprocessing.py
model_selection.py
trainer.py
evaluator.py
llm_report.py
config.py
```

### Core Flow

1. Load data
2. Detect task type
3. Preprocess features
4. Split train/test
5. Cross-validation model comparison
6. Select best model
7. Train model
8. Evaluate
9. Save artifacts
10. Generate LLM insights

---

## Features

- Automatic regression/classification detection
- Multi-model comparison:
  - LinearRegression
  - RandomForest
  - GradientBoosting
- 5-fold Cross Validation
- Feature importance extraction (model-aware)
- JSON metrics export
- Model persistence (.pkl)
- LLM Insight generation via OpenRouter
- Structured modular architecture
- Logging support
- Docker containerization
- GitHub Codespaces compatible

---

## Example Run

```bash
python main.py --file data/housing.csv --target median_house_value
```

---

## Example Output

**Metrics**

```json
{
  "task_type": "regression",
  "best_model": "RandomForest",
  "mae": 31921.60,
  "r2": 0.8160
}
```

**Top Features**

- median_income
- ocean_proximity_INLAND
- longitude
- latitude
- housing_median_age

**LLM Report**

Automatically generated structured business interpretation.

---

## Environment Variables

Create `.env` locally (do not commit):

```
OPENROUTER_API_KEY=your_key_here
```

---

## Docker Usage

Build image:

```bash
docker build -t ai-automl .
```

Run container:

```bash
docker run --env-file .env ai-automl --file data/housing.csv --target median_house_value
```

---

## Tech Stack

- Python 3.11
- scikit-learn
- pandas
- OpenRouter API
- Docker
- GitHub Codespaces

---

## Project Purpose

This project demonstrates:

- ML system design
- Automated model comparison
- Clean modular architecture
- Prompt engineering integration
- Reproducible containerized execution

---

## Future Improvements

- Hyperparameter tuning
- Classification metric extension
- MLflow experiment tracking
- REST API deployment
- CI/CD integration
