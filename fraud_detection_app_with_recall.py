import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
import numpy as np
from cleaners import TransactionCleaner, IdentityCleaner

# Definir las columnas categóricas (basado en tu script original)
categorical_features = [
    'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'id_01', 'id_02', 'id_04', 'id_11', 'id_12', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19',
    'id_20', 'id_22', 'id_23', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35',
    'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo'
]

# Configuración de la página
st.set_page_config(
    page_title="Fraud Detection App with Recall Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS (adaptado del script original)
st.markdown(
    """
    <style>
    body {
        background-color: #F5F6F5;
        color: #333333;
    }
    .stApp {
        background-color: #F5F6F5;
    }
    .stButton>button {
        background-color: #003087;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #005566;
    }
    h1, h2, h3 {
        color: #003087;
    }
    .stFileUploader {
        background-color: #FFFFFF;
        border: 1px solid #D3D3D3;
        border-radius: 5px;
        padding: 10px;
    }
    .stDataFrame {
        border: 1px solid #D3D3D3;
        border-radius: 5px;
    }
    .stAlert {
        background-color: #E6F0FA;
        color: #003087;
        border: 1px solid #003087;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Título y descripción
st.title("🏦 Fraud Detection App with Recall Analysis")
st.markdown("""
    This application detects fraudulent transactions using a pre-trained LightGBM model.  
    Upload your transaction and identity data, adjust the threshold, and analyze how recall and other metrics change in real-time.  
    **Instructions:** Upload `test_transaction_sample.csv` and `test_identity_sample.csv` with matching `TransactionID`s.
""")

# Barra lateral para cargar archivos y configurar el umbral
st.sidebar.header("Upload Data Files")
transaction_file = st.sidebar.file_uploader("Upload Transaction Data (CSV)", type="csv", key="transaction")
identity_file = st.sidebar.file_uploader("Upload Identity Data (CSV)", type="csv", key="identity")

# Control deslizante para ajustar el umbral
st.sidebar.header("Adjust Fraud Detection Threshold")
default_threshold = 0.30  # Basado en tu último modelo (recall > 90%)
threshold = st.sidebar.slider(
    "Fraud Probability Threshold",
    min_value=0.1,
    max_value=0.9,
    value=default_threshold,
    step=0.01,
    help="Adjust the threshold to classify transactions as fraudulent. Lower values increase recall but may raise false positives."
)

# Cargar datos de prueba para análisis de métricas
try:
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')['isFraud']
    # Preparar X_test con las columnas categóricas
    for col in categorical_features:
        if col in X_test.columns:
            X_test.loc[:, col] = X_test[col].astype('category')
except FileNotFoundError:
    st.warning("X_test.csv or y_test.csv not found. Please ensure these files are available for recall analysis.")

# Cargar el modelo y predecir probabilidades para X_test
if 'X_test' in locals() and 'y_test' in locals():
    model_recall = lgb.Booster(model_file='lightgbm_model_high_recall.txt')
    y_pred_proba_test = model_recall.predict(X_test)

    # Calcular métricas para el umbral seleccionado
    y_pred_test = (y_pred_proba_test >= threshold).astype(int)
    recall = recall_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Mostrar métricas en la barra lateral
    st.sidebar.subheader("Metrics on Test Data")
    st.sidebar.write(f"Threshold: {threshold:.2f}")
    st.sidebar.write(f"Recall: {recall:.4f}")
    st.sidebar.write(f"Precision: {precision:.4f}")
    st.sidebar.write(f"F1-Score: {f1:.4f}")
    st.sidebar.write(f"False Positive Rate (FPR): {fpr:.4f}")

# Botón para procesar los datos cargados
if st.sidebar.button("Process and Predict"):
    if transaction_file is None or identity_file is None:
        st.error("Please upload both transaction and identity data files.")
    else:
        try:
            # Cargar los datos subidos
            df_transaction = pd.read_csv(transaction_file)
            df_identity = pd.read_csv(identity_file)

            # Renombrar columnas de identity
            df_identity.columns = df_identity.columns.str.replace('-', '_')

            # Verificar que los TransactionID coincidan
            transaction_ids = set(df_transaction['TransactionID'])
            identity_ids = set(df_identity['TransactionID'])
            common_ids = transaction_ids.intersection(identity_ids)

            if len(common_ids) == 0:
                st.error("No matching TransactionIDs found between the two files.")
            elif len(common_ids) != len(df_transaction) or len(common_ids) != len(df_identity):
                st.warning(f"Only {len(common_ids)} TransactionIDs match. Processing only matching transactions.")
                df_transaction = df_transaction[df_transaction['TransactionID'].isin(common_ids)]
                df_identity = df_identity[df_identity['TransactionID'].isin(common_ids)]

            # Cargar los pipelines entrenados
            with st.spinner("Loading pipelines..."):
                transaction_cleaner = joblib.load('transaction_cleaner_trained.pkl')
                identity_cleaner = joblib.load('identity_cleaner_trained.pkl')

            # Procesar los datos con los pipelines
            with st.spinner("Cleaning transaction data..."):
                df_transaction_cleaned = transaction_cleaner.transform(df_transaction)
            with st.spinner("Cleaning identity data..."):
                df_identity_cleaned = identity_cleaner.transform(df_identity)

            # Unir los datos
            with st.spinner("Merging data..."):
                df_merged = df_transaction_cleaned.merge(df_identity_cleaned, on='TransactionID', how='inner')

            # Preparar los datos para predicción
            X = df_merged.drop(columns=['TransactionID'])
            for col in categorical_features:
                if col in X.columns:
                    X.loc[:, col] = X[col].astype('category')

            # Cargar el modelo LightGBM
            with st.spinner("Loading model..."):
                model = lgb.Booster(model_file='lightgbm_model_high_recall.txt')

            # Hacer predicciones
            with st.spinner("Predicting fraud..."):
                y_pred_proba = model.predict(X)
                y_pred = (y_pred_proba >= threshold).astype(int)

            # Agregar predicciones al DataFrame
            new_cols = {
                'Predicted_isFraud': y_pred,
                'Fraud_Probability': y_pred_proba
            }
            new_cols_df = pd.DataFrame(new_cols, index=df_merged.index)
            df_merged = pd.concat([df_merged, new_cols_df], axis=1)

            # Mostrar resultados
            st.header("Prediction Results")
            st.markdown(f"**Total Transactions Processed:** {len(df_merged)}")
            st.markdown(f"**Fraudulent Transactions Detected (Threshold = {threshold}):** {df_merged['Predicted_isFraud'].sum()}")

            # Gráfico de barras: Transacciones fraudulentas vs no fraudulentas
            st.subheader("Fraud Detection Summary")
            fraud_counts = df_merged['Predicted_isFraud'].value_counts()
            fraud_labels = ['Non-Fraudulent', 'Fraudulent']
            fraud_values = [fraud_counts.get(0, 0), fraud_counts.get(1, 0)]

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=fraud_values, y=fraud_labels, hue=fraud_labels, palette=['#003087', '#D32F2F'], legend=False)
            ax.set_title('Fraud Detection Results', fontsize=14, color='#003087')
            ax.set_xlabel('Number of Transactions', fontsize=12)
            ax.set_ylabel('Fraud Status', fontsize=12)
            for i, v in enumerate(fraud_values):
                ax.text(v + 0.5, i, str(v), color='#333333', va='center', fontweight='bold')
            st.pyplot(fig)

            # Histograma de probabilidades de fraude
            st.subheader("Distribution of Fraud Probabilities")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df_merged['Fraud_Probability'], bins=30, kde=True, color='#003087', ax=ax)
            ax.axvline(x=threshold, color='#D32F2F', linestyle='--', label=f'Threshold = {threshold}')
            ax.set_title('Distribution of Fraud Probabilities', fontsize=14, color='#003087')
            ax.set_xlabel('Fraud Probability', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend()
            st.pyplot(fig)

            # Gráfico de Recall vs Threshold (usando X_test y y_test)
            if 'X_test' in locals() and 'y_test' in locals():
                st.subheader("Recall vs Threshold Analysis")
                thresholds = np.arange(0.1, 1.0, 0.01)
                recalls = [recall_score(y_test, (y_pred_proba_test >= t).astype(int)) for t in thresholds]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(thresholds, recalls, label='Recall', color='#003087')
                ax.axvline(x=threshold, color='#D32F2F', linestyle='--', label=f'Current Threshold = {threshold}')
                ax.set_xlabel('Threshold', fontsize=12)
                ax.set_ylabel('Recall', fontsize=12)
                ax.set_title('Recall vs Threshold', fontsize=14, color='#003087')
                ax.legend()
                ax.grid()
                st.pyplot(fig)

            # Mostrar transacciones fraudulentas
            st.subheader("Fraudulent Transactions")
            fraudulent_transactions = df_merged[df_merged['Predicted_isFraud'] == 1][['TransactionID', 'Fraud_Probability']]
            if len(fraudulent_transactions) > 0:
                st.dataframe(fraudulent_transactions.style.format({'Fraud_Probability': '{:.4f}'}))
            else:
                st.info("No fraudulent transactions detected with the current threshold. Try lowering the threshold.")

            # Descargar resultados
            st.subheader("Download Results")
            csv = df_merged[['TransactionID', 'Predicted_isFraud', 'Fraud_Probability']].to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.info("Upload the transaction and identity data files, then click 'Process and Predict' to start.")

# Footer
st.markdown("---")
st.markdown("**Fraud Detection App with Recall Analysis** | Developed for Final Project | 2025", unsafe_allow_html=True)