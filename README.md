# Fraud Detection using Machine Learning

## Introduction
Financial fraud is a significant challenge in the digital era, where millions of transactions occur every day. Detecting fraudulent activities in real time is essential to minimizing financial losses and ensuring user security. This project focuses on developing a machine learning model to identify fraudulent transactions using real-world data.

## Project Overview
This repository contains the code and resources for my fraud detection application, developed during my Ironhack Data Analytics Bootcamp. Built entirely by me, this project uses machine learning to identify fraudulent transactions in a banking scenario and won the top award at the Hackshow for its practical impact. The system processes a dataset of 144,000 transactions with 434 features (e.g., amounts, card details, devices) to predict fraud in real time, featuring a pipeline with data preprocessing, feature engineering, a LightGBM model, and a Streamlit app for user interaction.

## Repository Structure

- `data/`: Contains information about the dataset (see below for details).
- `results/`: Stores presentation materials, visualizations, and model performance outputs.
- `src/`: Includes the source code for data processing, model training, and the app.

## Project Highlights

- Real-Time Detection: Uses LightGBM to flag fraud instantly.
- Feature Engineering: Custom features (e.g., D2, C1-C14) enhance accuracy.
- Impact: Won the Hackshow for its practical banking application.

## Technologies Used

- Python: Core language (Pandas, NumPy, Scikit-learn).
- LightGBM: For fast, scalable predictions.
- Tableau: For data visualization.
- Streamlit: For the app interface.
- Git: For version control.
