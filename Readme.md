📌 About This Project
This project demonstrates an AI/ML-based IoT sensor data validation pipeline, designed for early fault detection and degradation tracking in newly installed industrial sensors — particularly within an energy production or asset-heavy environment.

The goal is to ensure data quality from new sensors by validating against historical behavior using a multi-model ensemble and surfacing results in a clear monitoring dashboard for operational decision-making.

🎯 Problem Statement
Newly deployed sensors may exhibit anomalies or faulty readings that can lead to inaccurate production reports — potentially impacting compliance, forecasting, and asset performance monitoring.

This project simulates a realistic scenario where:

Data is ingested and enriched with trend-aware features (rolling mean, std, delta)

An ensemble of ML models (XGBoost, Autoencoder, Isolation Forest) classifies sensors as Normal or Faulty

Alerts and anomaly breakdowns are visualized using Power BI

⚙️ Key Features
ETL pipeline in Python for data preprocessing, trend feature engineering, and model inference

Multi-model validation logic, using:

XGBoost residual analysis

Autoencoder reconstruction errors

Isolation Forest outlier scoring

Interactive Power BI dashboard for:

Real-time sensor status

Drift and deviation plots per feature

ML flag breakdown and filter

Designed for modular expansion across multiple sensor types and sites

🔧 Tech Stack
Python (Pandas, Scikit-learn, XGBoost, Autoencoder via Keras)

Power BI for visualization

Mock sensor data with labeled degradation scenarios

Designed to integrate with real-time ingestion layers (e.g., Kafka or REST endpoints)