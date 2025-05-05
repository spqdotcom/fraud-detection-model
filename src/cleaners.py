# cleaners.py
import pandas as pd
import numpy as np

class TransactionCleaner:
    def __init__(self):
        self.avg_amt_by_card1 = None
        self.amt_percentile_90_by_card1 = None
        self.fraud_rate_by_remail = None
        self.c14_threshold = None

    def group_email_domain(self, domain):
        if pd.isna(domain):
            return 'nan'
        domain = domain.lower()
        if 'gmail' in domain:
            return 'gmail'
        elif 'yahoo' in domain:
            return 'yahoo'
        elif 'hotmail' in domain:
            return 'hotmail'
        elif 'outlook' in domain:
            return 'outlook'
        elif 'aol' in domain:
            return 'aol'
        elif 'anonymous' in domain:
            return 'anonymous'
        elif 'icloud' in domain:
            return 'icloud'
        elif 'comcast' in domain:
            return 'comcast'
        elif 'verizon' in domain:
            return 'verizon'
        elif 'prodigy' in domain:
            return 'prodigy'
        elif 'servicios' in domain:
            return 'servicios'
        else:
            return 'other'

    def fit(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but got {type(df)}")

        df = df.copy()
        df.columns = df.columns.str.replace('-', '_')

        email_mapping = {
            'gmail': 0, 'yahoo': 1, 'hotmail': 2, 'outlook': 3, 'aol': 4,
            'anonymous': 5, 'icloud': 6, 'comcast': 7, 'verizon': 8,
            'prodigy': 9, 'servicios': 10, 'other': 11, 'nan': float('nan')
        }
        df['R_emaildomain'] = df['R_emaildomain'].apply(self.group_email_domain)
        df['R_emaildomain'] = df['R_emaildomain'].map(email_mapping).astype('float32')

        self.avg_amt_by_card1 = df.groupby('card1')['TransactionAmt'].mean()
        self.amt_percentile_90_by_card1 = df.groupby('card1')['TransactionAmt'].quantile(0.90)
        self.fraud_rate_by_remail = df.groupby('R_emaildomain')['isFraud'].mean().to_dict()
        self.c14_threshold = df['C14'].quantile(0.90, interpolation='nearest')

        print("\nEstadísticas calculadas en fit:")
        print("avg_amt_by_card1:")
        print(self.avg_amt_by_card1.describe())
        print("\namt_percentile_90_by_card1:")
        print(self.amt_percentile_90_by_card1.describe())
        print("\nfraud_rate_by_remail:")
        print(self.fraud_rate_by_remail)

        return self

    def transform(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but got {type(df)}")

        df = df.copy()
        df.columns = df.columns.str.replace('-', '_')

        # Convertir columnas float64 a float32
        float64_cols = df.select_dtypes(include=['float64']).columns
        df[float64_cols] = df[float64_cols].astype('float32')

        # Transformaciones iniciales
        df['ProductCD'] = df['ProductCD'].map({
            'W': 0, 'H': 1, 'C': 2, 'S': 3, 'R': 4
        }).astype('float32')

        df['card4'] = df['card4'].map({
            'discover': 0, 'mastercard': 1, 'visa': 2, 'american express': 3
        }).astype('float32')

        df['card6'] = df['card6'].map({
            'credit': 0, 'debit': 1, 'debit or credit': 2, 'charge card': 3
        }).astype('float32')

        email_mapping = {
            'gmail': 0, 'yahoo': 1, 'hotmail': 2, 'outlook': 3, 'aol': 4,
            'anonymous': 5, 'icloud': 6, 'comcast': 7, 'verizon': 8,
            'prodigy': 9, 'servicios': 10, 'other': 11, 'nan': float('nan')
        }
        df['P_emaildomain'] = df['P_emaildomain'].apply(self.group_email_domain).map(email_mapping).astype('float32')
        df['R_emaildomain'] = df['R_emaildomain'].apply(self.group_email_domain).map(email_mapping).astype('float32')

        # Crear columnas binarias para presencia de email domains
        new_cols = {}
        new_cols['P_emaildomain_present'] = df['P_emaildomain'].notnull().astype('int')
        new_cols['R_emaildomain_present'] = df['R_emaildomain'].notnull().astype('int')

        # Transformar columnas M
        m_columns_tf = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
        for col in m_columns_tf:
            if col in df.columns:
                df[col] = df[col].map({'T': 1, 'F': 0}).astype('float32')

        if 'M4' in df.columns:
            df['M4'] = df['M4'].map({'M0': 0, 'M1': 1, 'M2': 2}).astype('float32')

        # Crear columnas binarias para grupos de M
        m7_m9_cols = ['M7', 'M8', 'M9']
        if all(col in df.columns for col in m7_m9_cols):
            new_cols['M7_M9_present'] = df[m7_m9_cols].notnull().any(axis=1).astype('int')

        m1_m3_cols = ['M1', 'M2', 'M3']
        if all(col in df.columns for col in m1_m3_cols):
            new_cols['M1_M3_present'] = df[m1_m3_cols].notnull().any(axis=1).astype('int')

        # Eliminar columnas redundantes si existen
        redundant_cols = ['M1_present', 'M2_M3_present', 'M3_present']
        df = df.drop(columns=[col for col in redundant_cols if col in df.columns])

        # Crear columnas binarias para grupos de Vxxx
        v1_v11 = [f'V{i}' for i in range(1, 12) if f'V{i}' in df.columns]
        if v1_v11:
            new_cols['V1_V11_present'] = df[v1_v11].notnull().any(axis=1).astype('int')

        v12_v34 = [f'V{i}' for i in range(12, 35) if f'V{i}' in df.columns]
        if v12_v34:
            new_cols['V12_V34_present'] = df[v12_v34].notnull().any(axis=1).astype('int')

        v35_v36 = [f'V{i}' for i in range(35, 37) if f'V{i}' in df.columns]
        if v35_v36:
            new_cols['V35_V36_present'] = df[v35_v36].notnull().any(axis=1).astype('int')

        v37_v52 = [f'V{i}' for i in range(37, 53) if f'V{i}' in df.columns]
        if v37_v52:
            new_cols['V37_V52_present'] = df[v37_v52].notnull().any(axis=1).astype('int')

        v53_v74 = [f'V{i}' for i in range(53, 75) if f'V{i}' in df.columns]
        if v53_v74:
            new_cols['V53_V74_present'] = df[v53_v74].notnull().any(axis=1).astype('int')

        v75_v94 = [f'V{i}' for i in range(75, 95) if f'V{i}' in df.columns]
        if v75_v94:
            new_cols['V75_V94_present'] = df[v75_v94].notnull().any(axis=1).astype('int')

        v95_v137 = [f'V{i}' for i in range(95, 138) if f'V{i}' in df.columns]
        if v95_v137:
            new_cols['V95_V137_present'] = df[v95_v137].notnull().any(axis=1).astype('int')

        v138_v339 = [f'V{i}' for i in range(138, 340) if f'V{i}' in df.columns]
        if v138_v339:
            new_cols['V138_V339_present'] = df[v138_v339].notnull().any(axis=1).astype('int')

        # Transformaciones de tiempo (TransactionDT y D1-D15)
        min_transaction_dt = df['TransactionDT'].min()
        time_features = {
            'TransactionDT_days': (df['TransactionDT'] / 86400).astype('float32'),
            'TransactionDT_normalized': (df['TransactionDT'] - min_transaction_dt).astype('float32')
        }
        time_features['TransactionDT_hour'] = (time_features['TransactionDT_normalized'] % 86400 // 3600).astype('float32')
        time_features['TransactionDT_day_of_week'] = (time_features['TransactionDT_normalized'] // 86400 % 7).astype('float32')
        time_features['TransactionDT_day_relative'] = (time_features['TransactionDT_normalized'] // 86400).astype('float32')

        new_cols.update(time_features)

        # Diferencias y frecuencias
        new_cols['D2_minus_D1'] = (df['D2'] - df['D1']).astype('float32')
        new_cols['D2_over_D1'] = (df['D2'] / df['D1'].replace(0, np.nan)).astype('float32')
        new_cols['D1_frequency'] = (1 / df['D1'].replace(0, np.nan)).astype('float32')
        new_cols['D2_frequency'] = (1 / df['D2'].replace(0, np.nan)).astype('float32')

        # Columnas binarias para presencia de D
        d6_d9 = ['D6', 'D8', 'D9']
        if all(col in df.columns for col in d6_d9):
            new_cols['D6_D9_present'] = df[d6_d9].notnull().any(axis=1).astype('int')

        d12_d14 = ['D12', 'D13', 'D14']
        if all(col in df.columns for col in d12_d14):
            new_cols['D12_D14_present'] = df[d12_d14].notnull().any(axis=1).astype('int')

        d2_d3_d11 = ['D2', 'D3', 'D11']
        if all(col in df.columns for col in d2_d3_d11):
            new_cols['D2_D3_D11_present'] = df[d2_d3_d11].notnull().any(axis=1).astype('int')

        for col in ['D1', 'D4', 'D5', 'D7', 'D10', 'D15', 'dist2']:
            if col in df.columns:
                new_cols[f'{col}_present'] = df[col].notnull().astype('int')

        # Crear addr_present
        if all(col in df.columns for col in ['addr1', 'addr2']):
            new_cols['addr_present'] = df[['addr1', 'addr2']].notnull().any(axis=1).astype('int')

        # Eliminar columnas Vxxx de baja importancia
        feature_importances = {
            'R_emaildomain': 115, 'TransactionAmt': 111, 'C1': 110, 'C14': 96, 'card2': 93,
            'card3': 83, 'card1': 81, 'D8': 79, 'C13': 75, 'ProductCD': 69, 'P_emaildomain': 69,
            'TransactionDT': 63, 'id_02': 57, 'TransactionDT_day_relative': 57, 'D2': 55,
            'id_20': 53, 'V156': 49, 'card6': 43, 'id_30': 42, 'V87': 37, 'V258': 36,
            'V165': 34, 'C11': 33, 'addr1': 31, 'id_06': 28, 'id_14': 28, 'D4': 28, 'C3': 26,
            'id_09': 25, 'V45': 25, 'D13': 24, 'id_01': 23, 'dist2': 22, 'id_17': 21,
            'id_33': 21, 'D15': 20, 'V55': 20, 'TransactionDT_hour': 19, 'D3': 18, 'D14': 18,
            'V189': 18, 'id_31': 17, 'DeviceInfo': 17, 'D6': 17, 'id_19': 16, 'C2': 16,
            'card5': 15, 'D2_frequency': 15, 'id_03': 13, 'id_07': 13, 'id_18': 13, 'V308': 13,
            'V56': 12, 'V58': 12, 'V310': 12, 'V314': 12, 'V315': 12, 'TransactionDT_days': 12,
            'card4': 10, 'V139': 10, 'V251': 10, 'V261': 10, 'C12': 9, 'V67': 9, 'V74': 9,
            'V78': 9, 'V207': 9, 'id_36': 8, 'V37': 8, 'V234': 8, 'id_10': 7, 'V149': 7,
            'V152': 7, 'C4': 6, 'C8': 6, 'V143': 6, 'V158': 6, 'V169': 6, 'V206': 6, 'V245': 6,
            'V332': 6, 'V333': 6, 'V338': 6, 'id_04': 5, 'id_32': 5, 'V57': 5, 'V137': 5,
            'V145': 5, 'V159': 5, 'V162': 5, 'V166': 5, 'V217': 5, 'V221': 5, 'V224': 5,
            'V263': 5, 'V280': 5, 'V309': 5, 'V336': 5, 'D1_frequency': 5, 'id_13': 4, 'D5': 4,
            'V38': 4, 'V150': 4, 'V157': 4, 'V160': 4, 'V170': 4, 'V172': 4, 'V187': 4,
            'V192': 4, 'V205': 4, 'V209': 4, 'V259': 4, 'V266': 4, 'V281': 4, 'V293': 4,
            'V294': 4, 'V335': 4, 'V339': 4, 'P_emaildomain_present': 4, 'D7': 3, 'D9': 3,
            'D10': 3, 'D12': 3, 'V44': 3, 'V52': 3, 'V77': 3, 'V79': 3, 'V99': 3, 'V109': 3,
            'V130': 3, 'V161': 3, 'V164': 3, 'V201': 3, 'V208': 3, 'V229': 3, 'V243': 3,
            'V257': 3, 'V271': 3, 'V312': 3, 'V313': 3, 'V324': 3, 'V327': 3,
            'TransactionDT_day_of_week': 3, 'id_34': 2, 'C6': 2, 'C7': 2, 'V25': 2, 'V42': 2,
            'V43': 2, 'V47': 2, 'V51': 2, 'V63': 2, 'V64': 2, 'V66': 2, 'V71': 2, 'V72': 2,
            'V73': 2, 'V81': 2, 'V83': 2, 'V84': 2, 'V127': 2, 'V128': 2, 'V135': 2, 'V140': 2,
            'V178': 2, 'V213': 2, 'V228': 2, 'V230': 2, 'V232': 2, 'V256': 2, 'V264': 2,
            'V265': 2, 'V267': 2, 'V268': 2, 'V270': 2, 'V279': 2, 'V283': 2, 'V285': 2,
            'V290': 2, 'V311': 2, 'V317': 2, 'V323': 2, 'V330': 2, 'D2_over_D1': 2, 'id_08': 1,
            'id_11': 1, 'id_21': 1, 'id_24': 1, 'DeviceType': 1, 'V15': 1, 'V18': 1, 'V23': 1,
            'V39': 1, 'V40': 1, 'V53': 1, 'V94': 1, 'V102': 1, 'V123': 1, 'V133': 1, 'V136': 1,
            'V151': 1, 'V154': 1, 'V155': 1, 'V175': 1, 'V181': 1, 'V182': 1, 'V184': 1,
            'V185': 1, 'V188': 1, 'V204': 1, 'V212': 1, 'V223': 1, 'V225': 1, 'V231': 1,
            'V233': 1, 'V238': 1, 'V242': 1, 'V244': 1, 'V246': 1, 'V248': 1, 'V262': 1,
            'V272': 1, 'V274': 1, 'V278': 1, 'V282': 1, 'V296': 1, 'V300': 1, 'V303': 1,
            'V319': 1, 'V328': 1, 'V331': 1, 'V334': 1
        }
        low_importance_v_cols = [f'V{i}' for i in range(1, 340) if f'V{i}' in df.columns and f'V{i}' in feature_importances and feature_importances[f'V{i}'] <= 2]
        df = df.drop(columns=low_importance_v_cols)

        # Convertir float64 a float32 e int64 a int32
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        # Características derivadas
        new_cols['Fraud_Rate_R_emaildomain'] = df['R_emaildomain'].map(self.fraud_rate_by_remail).astype('float32')
        new_cols['TransactionAmt_Diff_Avg_card1'] = (df['TransactionAmt'] - df['card1'].map(self.avg_amt_by_card1)).astype('float32')
        new_cols['TransactionAmt_High_card1'] = (df['TransactionAmt'] > df['card1'].map(self.amt_percentile_90_by_card1)).astype('int')
        new_cols['C14_High'] = (df['C14'] > self.c14_threshold).astype('int')

        # Frecuencia de transacciones por card1 en el último día
        transactions_last_day = np.zeros(len(df), dtype=np.float32)
        grouped = df.groupby('card1')
        for card, group in grouped:
            group = group.sort_values('TransactionDT')
            transaction_dt = group['TransactionDT'].values
            indices = group.index
            counts = np.zeros(len(group), dtype=np.float32)
            for i in range(len(group)):
                window_start = transaction_dt[i] - 86400
                counts[i] = np.sum((transaction_dt >= window_start) & (transaction_dt <= transaction_dt[i]))
            transactions_last_day[indices] = counts
        new_cols['Transactions_Last_Day'] = transactions_last_day

        new_cols['D8_minus_D1'] = (df['D8'] - df['D1']).astype('float32')

        v156_166 = [f'V{i}' for i in range(156, 167) if f'V{i}' in df.columns]
        if v156_166:
            new_cols['V156_V166_Sum'] = df[v156_166].sum(axis=1, skipna=True).astype('float32')

        # Concatenar todas las columnas nuevas al DataFrame
        new_cols_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_cols_df], axis=1)

        # Debug: Imprimir estadísticas de columnas clave
        cols_to_check = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
                         'TransactionAmt', 'Fraud_Rate_R_emaildomain', 'TransactionAmt_Diff_Avg_card1',
                         'TransactionAmt_High_card1', 'C14_High', 'Transactions_Last_Day', 'D8_minus_D1',
                         'V156_V166_Sum']
        print("\nEstadísticas de columnas clave después de la transformación:")
        for col in cols_to_check:
            if col in df.columns:
                print(f"\nColumna: {col}")
                print(f"Mean: {df[col].mean()}")
                print(f"Median: {df[col].median()}")
                print(f"% Null: {df[col].isna().mean() * 100:.5f}%")
                print(f"Valores únicos: {df[col].nunique()}")
                print(f"Contenido único: {df[col].unique()[:10]}")

        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)

class IdentityCleaner:
    def __init__(self, category_mappings=None, top_categories=None):
        self.category_mappings = category_mappings if category_mappings is not None else {}
        self.top_categories = top_categories if top_categories is not None else {}

    def group_resolution(self, res):
        if pd.isna(res):
            return float('nan')
        try:
            width, height = map(int, res.split('x'))
            area = width * height
            if area <= 1_000_000:
                return 'small'
            elif area <= 2_500_000:
                return 'medium'
            else:
                return 'large'
        except:
            return 'other'

    def group_device_info(self, device):
        if pd.isna(device):
            return 'nan'
        device = device.lower()
        if 'samsung' in device:
            return 'Samsung'
        elif 'ios' in device:
            return 'iOS'
        elif 'windows' in device:
            return 'Windows'
        elif 'mac' in device:
            return 'Mac'
        else:
            return 'Other'

    def fit(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but got {type(df)}")

        df = df.copy()
        df.columns = df.columns.str.replace('-', '_')

        top_categories_limits = {
            'id_30': 5,
            'id_31': 5,
            'id_33': 3,
            'DeviceInfo': 5
        }

        label_cols = ['id_30', 'id_31', 'id_33', 'DeviceInfo']
        for col in label_cols:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in DataFrame")
            
            if col in ['id_33', 'DeviceInfo']:
                continue
                
            top_n = top_categories_limits[col]
            top_cats = df[col].value_counts(dropna=True).head(top_n).index.tolist()
            self.top_categories[col] = top_cats

            if col == 'id_30':
                mapping = {
                    'Windows 10': 2.0,
                    'Windows 7': 2.0,
                    'iOS 11.2.1': 1.0,
                    'iOS 11.1.2': 1.0,
                    'Android 7.0': 0.0,
                    'other': 3.0
                }
            elif col == 'id_31':
                mapping = {
                    'chrome 63.0': 0.0,
                    'mobile safari 11.0': 1.0,
                    'mobile safari generic': 1.0,
                    'ie 11.0 for desktop': 4.0,
                    'safari generic': 1.0,
                    'other': 3.0
                }
            self.category_mappings[col] = mapping

        return self

    def transform(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, but got {type(df)}")

        df = df.copy()
        df.columns = df.columns.str.replace('-', '_')

        float64_cols = df.select_dtypes(include=['float64']).columns
        df[float64_cols] = df[float64_cols].astype('float32')

        print("Porcentaje de NaN por columna en id_11 a id_38:")
        for i in range(11, 39):
            col = f'id_{i}'
            if col in df.columns:
                null_percentage = df[col].isna().mean() * 100
                print(f"{col}: {null_percentage:.5f}%")

        print("\nPorcentaje de NaN por columna en id_30 a id_34:")
        for i in range(30, 35):
            col = f'id_{i}'
            if col in df.columns:
                null_percentage = df[col].isna().mean() * 100
                print(f"{col}: {null_percentage:.5f}%")

        if all(f'id_{i}' in df.columns for i in [11, 15, 28, 29, 35, 36, 37, 38]):
            include_cols = [11, 15, 28, 29, 35, 36, 37, 38]
            id_11_38_cols = df[[f'id_{i}' for i in include_cols]]
            all_nan_11_38 = id_11_38_cols.isna().all(axis=1).sum()
            print(f"\nRows with all NaN in id_11 to id_38 (using id_11, id_15, id_28, id_29, id_35, id_36, id_37, id_38): {all_nan_11_38}")

        if all(f'id_{i}' in df.columns for i in range(30, 35)):
            id_30_34_cols = df[[f'id_{i}' for i in range(30, 35) if i != 31]]
            all_nan_30_34 = id_30_34_cols.isna().all(axis=1).sum()
            print(f"Rows with all NaN in id_30 to id_34 (excluding id_31): {all_nan_30_34}")

        # Recolectar columnas binarias nuevas
        new_cols = {}
        if all(col in df.columns for col in ['id_35', 'id_36', 'id_37', 'id_38']):
            new_cols['id_35_38_present'] = df[['id_35', 'id_36', 'id_37', 'id_38']].notnull().any(axis=1).astype('int32')

        if all(f'id_{i}' in df.columns for i in [11, 15, 28, 29, 35, 36, 37, 38]):
            include_cols = [11, 15, 28, 29, 35, 36, 37, 38]
            new_cols['id_11_38_present'] = df[[f'id_{i}' for i in include_cols]].notnull().any(axis=1).astype('int32')

        if all(col in df.columns for col in ['id_07', 'id_27']):
            new_cols['id_07_27_present'] = df[['id_07', 'id_27']].notnull().any(axis=1).astype('int32')

        if all(f'id_{i}' in df.columns for i in range(30, 35)):
            new_cols['id_30_34_present'] = df[[f'id_{i}' for i in range(30, 35) if i != 31]].notnull().any(axis=1).astype('int32')

        if all(col in df.columns for col in ['id_02', 'DeviceType']):
            new_cols['id_02_DeviceType_present'] = df[['id_02', 'DeviceType']].notnull().any(axis=1).astype('int32')

        if 'id_31' in df.columns:
            new_cols['id_31_present'] = df['id_31'].notnull().astype('int32')

        if all(f'id_{i}' in df.columns for i in range(17, 21)):
            new_cols['id_17_20_present'] = df[[f'id_{i}' for i in range(17, 21)]].notnull().any(axis=1).astype('int32')

        if all(col in df.columns for col in ['id_05', 'id_06']):
            new_cols['id_05_06_present'] = df[['id_05', 'id_06']].notnull().any(axis=1).astype('int32')

        if 'id_13' in df.columns:
            new_cols['id_13_present'] = df['id_13'].notnull().astype('int32')

        if 'id_16' in df.columns:
            new_cols['id_16_present'] = df['id_16'].notnull().astype('int32')

        if all(col in df.columns for col in ['id_03', 'id_04']):
            new_cols['id_03_04_present'] = df[['id_03', 'id_04']].notnull().any(axis=1).astype('int32')

        if all(col in df.columns for col in ['id_09', 'id_10']):
            new_cols['id_09_10_present'] = df[['id_09', 'id_10']].notnull().any(axis=1).astype('int32')

        if 'id_14' in df.columns:
            new_cols['id_14_present'] = df['id_14'].notnull().astype('int32')

        if 'id_18' in df.columns:
            new_cols['id_18_present'] = df['id_18'].notnull().astype('int32')

        if 'id_24' in df.columns:
            new_cols['id_24_present'] = df['id_24'].notnull().astype('int32')

        # Transformar otras columnas categóricas
        if 'id_12' in df.columns:
            df['id_12'] = df['id_12'].map({'NotFound': 0, 'Found': 1}).astype('float32')

        if 'id_15' in df.columns:
            df['id_15'] = df['id_15'].map({'New': 0, 'Found': 1, 'Unknown': 2}).astype('float32')

        if 'id_16' in df.columns:
            df['id_16'] = df['id_16'].map({'NotFound': 0, 'Found': 1}).astype('float32')

        if 'id_23' in df.columns:
            df['id_23'] = df['id_23'].map({
                'IP_PROXY:TRANSPARENT': 0,
                'IP_PROXY:ANONYMOUS': 1,
                'IP_PROXY:HIDDEN': 2
            }).astype('float32')

        if 'id_27' in df.columns:
            df['id_27'] = df['id_27'].map({'NotFound': 0, 'Found': 1}).astype('float32')

        if 'id_28' in df.columns:
            df['id_28'] = df['id_28'].map({'New': 0, 'Found': 1}).astype('float32')

        if 'id_29' in df.columns:
            df['id_29'] = df['id_29'].map({'NotFound': 0, 'Found': 1}).astype('float32')

        if 'id_34' in df.columns:
            df['id_34'] = df['id_34'].map({
                'match_status:-1': -1,
                'match_status:0': 0,
                'match_status:1': 1,
                'match_status:2': 2
            }).astype('float32')

        for col in ['id_35', 'id_36', 'id_37', 'id_38']:
            if col in df.columns:
                df[col] = df[col].map({'T': 1, 'F': 0}).astype('float32')

        if 'DeviceType' in df.columns:
            df['DeviceType'] = df['DeviceType'].map({'mobile': 0, 'desktop': 1}).astype('float32')

        # Transformar id_30, id_31, id_33, DeviceInfo
        label_cols = ['id_30', 'id_31', 'id_33', 'DeviceInfo']
        nan_masks = {col: df[col].isna() for col in label_cols if col in df.columns}

        for col in label_cols:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in DataFrame")
            
            if col == 'id_33':
                df_col = df[col].apply(self.group_resolution)
                df[col] = df_col.map({
                    'small': 0,
                    'medium': 1,
                    'large': 2,
                    'other': 3,
                    float('nan'): float('nan')
                }).astype('float32')
            elif col == 'DeviceInfo':
                df_col = df[col].apply(self.group_device_info)
                df[col] = df_col.map({
                    'Samsung': 0,
                    'iOS': 1,
                    'Windows': 2,
                    'Mac': 3,
                    'Other': 4,
                    'nan': float('nan')
                }).astype('float32')
            else:
                df_col = df[col].where(df[col].isin(self.top_categories[col]), 'other')
                df[col] = df_col.map(self.category_mappings[col]).astype('float32')
                df[col] = df[col].where(~nan_masks[col], np.nan)

        # Concatenar columnas binarias nuevas
        new_cols_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_cols_df], axis=1)

        print("\nDistribución de valores en DeviceInfo después de la transformación:")
        print(df['DeviceInfo'].value_counts(dropna=False))

        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
