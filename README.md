# Wine Price Tier Prediction (ID2223 Project)

This project is developed for the course **ID2223 – Scalable Machine Learning and Deep Learning** at KTH.  
It demonstrates an end-to-end machine learning system using a **dynamic external data source**, a **prediction model**, and an **interactive UI**.

---

## Project Overview

We build a classification system that predicts the **price tier** of a wine based on information commonly found on the bottle label (e.g. country, category, alcohol percentage, volume).

Instead of predicting an exact price, wines are grouped into **four price tiers (Q1–Q4)**: [10-120],[120-249],[249-449],[449-1260]SEK within the mainstream market, improving robustness and interpretability.

---

## Data Source (Dynamic, External)

- **Source**: Publicly available Systembolaget product assortment mirror  
- **Access**: Data is fetched programmatically from an external URL (JSON)
- **Dynamic**: The dataset is maintained and updated by a third-party system
- **Not Kaggle / Not static files**

---

## Prediction Task

- **Task type**: Multi-class classification
- **Target**: Price tier (Q1–Q4)
- **Label construction**:
  - Exclude top 10% most expensive products (outlier filtering)
  - Remaining products are divided into four equal-sized price tiers using quantiles
- **Model**: XGBoost classifier
- **Evaluation**: Accuracy on a held-out test set

The task is **not a time series problem** and relies on **external, structured features**.

---

## Features (UI-Oriented)

Selected features are intentionally limited to information that users can realistically provide from a bottle label:

- Country
- Product category (Level 1 & 2)
- Alcohol percentage
- Volume (ml)
- Vintage
- Organic label

This design supports interpretability and a clear prediction workflow.

---

## System Architecture

1. **Data ingestion**: Fetch dynamic product data from external API
2. **Data processing & labeling**: Cleaning, filtering, price tier assignment
3. **Model training**: Performed in Jupyter Notebook, pipeline saved as an artifact
4. **Inference UI**: Streamlit app loads the trained model and performs predictions

---

## User Interface

- Built with **Streamlit**
- Users input bottle-label information
- Output includes:
  - Predicted price tier (Q1–Q4)
  - Corresponding estimated price range (SEK)
- Clean, product-style UI with branding and consistent visual design

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```


## Requirements Checklist
- Dynamic external data source (non-Kaggle)
- No pure time-series modeling
- Clearly defined prediction task
- Machine learning model with evaluation
- Interactive UI demonstrating practical value
All requirements are fulfilled.
