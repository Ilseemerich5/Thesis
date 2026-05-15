# Modeling the Temporal Persistence of Customer Sentiment from Online Reviews in E-Commerce Sales Forecasting

This repository contains the main code developed for my Master’s thesis in Data Science and Business Intelligence at EDC Paris Business School.

The thesis investigates for how long customer sentiment extracted from online product reviews remains useful when used as an input for sales forecasting models in an e-commerce context. More specifically, it studies whether the predictive value of review sentiment decays over time when integrated into forecasting architectures.

The central idea is that while many studies show that sentiment improves forecasting accuracy, they usually treat review data as static. In practice, however, reviews are time-dependent: a review written months or years ago may not carry the same informational value as a recent one. This project explores that temporal dimension explicitly through multiple look-back windows and forecasting horizons.

Repository structure

This repository includes the main scripts used for:

1. Sentiment analysis (BERT-based classification)
- A BERT-based model (BERTimbau) was used to classify customer reviews into sentiment categories.
- The output sentiment scores were aggregated and used as external features in the forecasting models.

2. Sales forecasting models
- A hybrid forecasting architecture combining:
- Attention-based Bidirectional LSTM (AttBiLSTM) for temporal sequence learning
- XGBoost for residual correction and error refinement
- Multiple experiments were run using different look-back windows (lag 1, 7, 14, 30, 60) and forecasting horizons.

3. Experimental design
- Each configuration tests how far back the model can “see” into the past.
- Forecasting horizons vary depending on the look-back window.
- A full ablation-style structure is used to isolate the contribution of sentiment features.

4. Analysis and visualization
- Training history comparison plots across look-back configurations
- Absolute error analysis across models and horizons
- Attention heatmaps to interpret which time steps are most important for predictions
- Performance metrics across all experiments (MAE, RMSE, WMAPE, NSE, ME)

Data:

The repository includes:
- The Olist e-commerce dataset (raw data) used for training and evaluation

Olist is a Brazilian e-commerce dataset containing real transaction data, product information, and customer reviews.

Survey data:

A primary survey was also conducted as part of the thesis to study consumer perception of review relevance over time.

The repository includes:
- Survey questionnaire (all questions)
- Google Forms link used for data collection
- Google Sheets link with collected responses

This survey complements the forecasting analysis by adding a behavioral perspective on how users perceive the temporal value of online reviews.


Important note:

This repository does not include all code developed during the thesis.
Several additional notebooks were used for:
- Data cleaning and preprocessing
- Exploratory analysis
- Intermediate experiments and model testing

Only the main final pipelines and core experimental scripts are included here.
