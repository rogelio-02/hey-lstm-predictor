import os
import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from flask import Flask, request, jsonify
import io # Import io module for string-based CSV reading
from flask_cors import CORS # Importa CORS



# --- Configuration Parameters (must match training CONFIG) ---
CONFIG = {
    'sequence_length': 10,
    'latent_dim': 16,
    'recurrent_error_percentile': 95,
    'recurring_interval_tolerance_days': 7,
    'min_recurring_occurrences': 3,
    'monto_grouping_tolerance': 0.05,
    'min_transactions_for_sequence_creation': 10,
    'batch_size_pred': 64
}

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Global variables for the model and preprocessor ---
model = None
preprocessor = None

# --- Preprocessing Functions (Identical to Training) ---
# These functions MUST be identical to the ones used during model training
# to ensure consistent feature engineering.
def clean_commerce_col(s: str):
    s = str(s).strip().replace(' ', '').replace('-', '').replace('&', '')
    return s

def calculate_time_features(df):
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['dia_de_semana'] = df['fecha'].dt.dayofweek
    df['dia'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['sin_dia'] = np.sin(2 * np.pi * df['dia'] / 31)
    df['cos_dia'] = np.cos(2 * np.pi * df['dia'] / 31)
    df['sin_mes'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['cos_mes'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['sin_de_dia_semana'] = np.sin(2 * np.pi * df['dia_de_semana'] / 7)
    df['cos_de_dia_semana'] = np.cos(2 * np.pi * df['dia_de_semana'] / 7)
    df['entre_semana'] = df['fecha'].dt.dayofweek.isin([5, 6]).astype(int)
    return df

def calculate_interval_features(df):
    df = df.sort_values(by=['id', 'comercio', 'fecha'])
    df['dias_desde_ultima_transaccion_general'] = df.groupby(by=['id', 'comercio'])['fecha'].diff().dt.days.fillna(0).astype(float)

    df = df.sort_values(by=['id', 'comercio', 'fecha'])
    df['dias_desde_ultima_transaccion_mismo_comercio'] = df.groupby(by=['id', 'comercio'])['fecha'].diff().dt.days.fillna(9999.0).astype(float)

    df = df.sort_values(by=['id', 'giro_comercio', 'fecha'])
    df['dias_desde_ultima_transaccion_mismo_giro'] = df.groupby(by=['id', 'giro_comercio'])['fecha'].diff().dt.days.fillna(9999.0).astype(float)
    return df

def process_raw_input_data(transactions_df):
    df_final = transactions_df

    if df_final.empty:
        print("DataFrame is empty after merge. Check IDs and input data.")
        return pd.DataFrame()

    key_cols_for_dropna = ['id', 'fecha', 'comercio', 'monto', 'giro_comercio', 'tipo_persona',
                           'actividad_empresarial', 'tipo_venta']
    actual_key_cols = [col for col in key_cols_for_dropna if col in df_final.columns]
    df_final.dropna(subset=actual_key_cols, inplace=True)

    df_final.dropna(inplace=True)
    df_final = df_final[df_final['giro_comercio'] != '4121']
    if 'comercio' in df_final.columns:
        df_final['comercio'] = df_final['comercio'].astype(str).apply(clean_commerce_col)

    print(f"Shape of data after merge and basic cleaning: {df_final.shape}")
    return df_final

def apply_feature_engineering(general_df: pd.DataFrame):
    new_df = general_df.copy()
    cols_to_drop = ['fecha_nacimiento', 'fecha_alta', 'id_municipio', 'id_estado', 'genero']
    existing_cols_to_drop = [col for col in cols_to_drop if col in new_df.columns]
    if existing_cols_to_drop:
        new_df = new_df.drop(labels=existing_cols_to_drop, axis=1)

    new_df = new_df.sort_values(by=['id', 'fecha'])
    new_df = calculate_time_features(new_df)
    new_df = calculate_interval_features(new_df)
    print(f"Shape after feature engineering: {new_df.shape}")
    return new_df

def prepare_flat_features(df_full_original, preprocessor_to_use):
    df_copy = df_full_original.copy()
    numerical_features = [
        'monto', 'dia_de_semana', 'dia', 'mes',
        'sin_dia', 'cos_dia', 'sin_mes', 'cos_mes',
        'sin_de_dia_semana', 'cos_de_dia_semana', 'entre_semana',
        'dias_desde_ultima_transaccion_general',
        'dias_desde_ultima_transaccion_mismo_comercio',
        'dias_desde_ultima_transaccion_mismo_giro'
    ]
    categorical_features = [
        'tipo_persona', 'actividad_empresarial', 'comercio', 'giro_comercio', 'tipo_venta'
    ]
    numerical_features = [col for col in numerical_features if col in df_copy.columns]
    categorical_features = [col for col in categorical_features if col in df_copy.columns]

    if not numerical_features and not categorical_features:
        print("Error: No numerical or categorical features found for processing.")
        return None, df_copy[['id', 'fecha', 'comercio', 'monto', 'dias_desde_ultima_transaccion_mismo_comercio']].copy()

    X_flat_processed = preprocessor_to_use.transform(df_copy)

    X_flat_processed = X_flat_processed.astype(np.float32)
    if np.isnan(X_flat_processed).any():
        print(f"WARNING: NaNs found. Shape: {X_flat_processed.shape}. Replacing with 0.")
        X_flat_processed = np.nan_to_num(X_flat_processed, nan=0.0)

    meta_cols = ['id', 'fecha', 'comercio', 'monto', 'dias_desde_ultima_transaccion_mismo_comercio']
    existing_meta_cols = [col for col in meta_cols if col in df_copy.columns]
    original_data_for_meta = df_copy[existing_meta_cols].copy()

    return X_flat_processed, original_data_for_meta

def identify_recurrent_patterns_unsupervised(model, X_sequences_dataset_for_prediction, original_sequences_meta, original_transactions_df_meta, X_sequences_np_array_full_original):
    print("Predicting reconstructions for all sequences...")
    reconstructions = model.predict(X_sequences_dataset_for_prediction, verbose=0)
    X_original_sequences_for_mse = X_sequences_np_array_full_original
    mse_per_sequence = np.mean(np.power(X_original_sequences_for_mse - reconstructions, 2), axis=(1, 2))

    for i, mse_val in enumerate(mse_per_sequence):
        if i < len(original_sequences_meta):
            original_sequences_meta[i]['mse'] = mse_val

    if len(mse_per_sequence) == 0:
        print("Warning: mse_per_sequence is empty. Cannot calculate threshold.")
        mse_recurrent_threshold = float('inf')
    else:
        mse_recurrent_threshold = np.percentile(mse_per_sequence, CONFIG['recurrent_error_percentile'])

    print(f"MSE Threshold for 'recurrent' sequences (errors <= this value are candidates): {mse_recurrent_threshold:.4f}")
    recurrent_candidates_meta = [s for s in original_sequences_meta if 'mse' in s and s['mse'] <= mse_recurrent_threshold]

    transaction_mse_map = {}
    for seq_meta in recurrent_candidates_meta:
        start_idx = seq_meta['sequence_start_idx_in_processed_df']
        end_idx = seq_meta['sequence_end_idx_in_processed_df']

        for original_df_idx in range(start_idx, end_idx + 1):
             if original_df_idx not in transaction_mse_map or seq_meta['mse'] < transaction_mse_map[original_df_idx]:
                transaction_mse_map[original_df_idx] = seq_meta['mse']

    original_transactions_df_meta['reconstruction_error'] = original_transactions_df_meta.index.map(transaction_mse_map)

    if original_transactions_df_meta['reconstruction_error'].dropna().empty:
        original_transactions_df_meta['reconstruction_error'] = float('inf')
    else:
        max_error_fill = original_transactions_df_meta['reconstruction_error'].replace([np.inf, -np.inf], np.nan).max()
        if pd.isna(max_error_fill): max_error_fill = 1.0
        original_transactions_df_meta['reconstruction_error'] = original_transactions_df_meta['reconstruction_error'].fillna(max_error_fill + 0.1)

    recurrent_transaction_original_indices = set(transaction_mse_map.keys())
    predicted_recurrent_transactions_df = original_transactions_df_meta.loc[list(recurrent_transaction_original_indices)].copy()

    if predicted_recurrent_transactions_df.empty:
        print("No individual transactions derived from recurrent candidate sequences.")
        return []

    final_recurrent_patterns = []
    predicted_recurrent_transactions_df['fecha'] = pd.to_datetime(predicted_recurrent_transactions_df['fecha'])
    predicted_recurrent_transactions_df = predicted_recurrent_transactions_df.sort_values(by=['id', 'comercio', 'fecha'])

    max_monto_val = predicted_recurrent_transactions_df['monto'].max()
    if pd.isna(max_monto_val) or max_monto_val == 0:
        predicted_recurrent_transactions_df['monto_group'] = 0
    else:
        grouping_val = max_monto_val * CONFIG['monto_grouping_tolerance']
        if grouping_val == 0: grouping_val = 0.01
        predicted_recurrent_transactions_df['monto_group'] = (
            predicted_recurrent_transactions_df['monto'] // grouping_val
        ).astype(int)

    for (person_id, comercio, monto_group), group_df in predicted_recurrent_transactions_df.groupby(['id', 'comercio', 'monto_group']):
        intervals = group_df['dias_desde_ultima_transaccion_mismo_comercio'].tolist()
        valid_intervals = [i for i in intervals if i > 0 and i < 9000]

        if len(group_df) >= CONFIG['min_recurring_occurrences'] and valid_intervals:
            mean_interval = np.mean(valid_intervals)
            std_interval = np.std(valid_intervals)
            mean_group_monto = group_df['monto'].mean()
            is_consistent_interval = std_interval < CONFIG['recurring_interval_tolerance_days']
            is_consistent_amount = (group_df['monto'].std() / mean_group_monto) < CONFIG['monto_grouping_tolerance'] if mean_group_monto != 0 else True

            if is_consistent_interval and is_consistent_amount and mean_interval > 0:
                final_recurrent_patterns.append({
                    'id': person_id,
                    'comercio': comercio,
                    'estimated_monto': round(mean_group_monto, 4),
                    'estimated_interval_days': round(mean_interval, 4),
                    'num_occurrences': len(group_df),
                    'first_occurrence_date': pd.to_datetime(group_df['fecha']).min().strftime('%Y-%m-%d'),
                    'last_occurrence_date': pd.to_datetime(group_df['fecha']).max().strftime('%Y-%m-%d'),
                    'avg_reconstruction_error_in_group': round(group_df['reconstruction_error'].mean(), 6)
                })
    return final_recurrent_patterns

def predict_recurrent_transactions(model_obj: keras.Model, trained_preprocessor_obj: ColumnTransformer, transactions_data: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Starting Prediction from provided DataFrames ---")

    df_merged_data = process_raw_input_data(transactions_data)
    if df_merged_data.empty:
        return pd.DataFrame()

    df_engineered_data = apply_feature_engineering(df_merged_data)
    df_engineered_data_indexed = df_engineered_data.reset_index(drop=True)
    print(f"Engineered data for prediction. Shape: {df_engineered_data_indexed.shape}")

    X_flat_processed_predict, original_data_for_meta_predict = prepare_flat_features(
        df_engineered_data_indexed,
        preprocessor_to_use=trained_preprocessor_obj
    )
    if X_flat_processed_predict is None:
        print("Error in prepare_flat_features for prediction data. Aborting.")
        return pd.DataFrame()
    print(f"Flat features prepared for prediction. Shape: {X_flat_processed_predict.shape}")

    all_X_sequences_predict = []
    all_sequences_meta_predict = []

    grouped_predict_data = original_data_for_meta_predict.groupby('id')

    for person_id, person_df_meta_predict in grouped_predict_data:
        if len(person_df_meta_predict) < CONFIG['min_transactions_for_sequence_creation']:
            continue

        person_features_array_predict = X_flat_processed_predict[person_df_meta_predict.index.values]

        for i in range(len(person_df_meta_predict) - CONFIG['sequence_length'] + 1):
            start_original_idx = person_df_meta_predict.index[i]
            end_original_idx = person_df_meta_predict.index[i + CONFIG['sequence_length'] - 1]
            current_sequence_data = person_features_array_predict[i : i + CONFIG['sequence_length']]

            if current_sequence_data.shape[0] == CONFIG['sequence_length']:
                all_X_sequences_predict.append(current_sequence_data)
                all_sequences_meta_predict.append({
                    'id': person_id,
                    'sequence_start_idx_in_processed_df': start_original_idx,
                    'sequence_end_idx_in_processed_df': end_original_idx,
                    'first_fecha_in_seq': person_df_meta_predict.loc[start_original_idx, 'fecha'],
                    'last_fecha_in_seq': person_df_meta_predict.loc[end_original_idx, 'fecha']
                })

    if not all_X_sequences_predict:
        print("No sequences created for prediction from the new data.")
        return pd.DataFrame()

    X_sequences_predict_np_array_full = np.array(all_X_sequences_predict, dtype=np.float32)
    print(f"Sequences created from new data for prediction. Shape: {X_sequences_predict_np_array_full.shape}")

    prediction_dataset = tf.data.Dataset.from_tensor_slices(
        X_sequences_predict_np_array_full
    ).batch(CONFIG['batch_size_pred']).prefetch(tf.data.AUTOTUNE)

    identified_patterns = identify_recurrent_patterns_unsupervised(
        model_obj,
        prediction_dataset,
        all_sequences_meta_predict,
        original_data_for_meta_predict,
        X_sequences_predict_np_array_full
    )

    if identified_patterns:
        final_df = pd.DataFrame(identified_patterns)
        final_df = final_df.sort_values(by=['id', 'first_occurrence_date'])
        return final_df
    else:
        return pd.DataFrame()

# --- Flask Route for Prediction ---
@app.route('/predict_recurrent', methods=['POST'])
def predict_recurrent_endpoint():
    global model, preprocessor

    if model is None or preprocessor is None:
        return jsonify({"error": "Model or preprocessor not loaded. Server is not ready."}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Expecting 'clients_csv' and 'transactions_csv' keys in the JSON data,
    # each containing a string of CSV data.
    if 'transactions_csv' not in data:
        return jsonify({"error": "Missing 'clients_csv' or 'transactions_csv' data in JSON payload."}), 400

    transactions_csv_string = data['transactions_csv']

    try:
        # Read CSV strings into pandas DataFrames
        # Using io.StringIO to treat the string as a file
        transactions_data = pd.read_csv(io.StringIO(transactions_csv_string), parse_dates=True, date_format='%d/%m/%Y')
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV strings into DataFrames. Check CSV format and headers: {e}"}), 400

    # Run the prediction
    try:
        identified_patterns_df = predict_recurrent_transactions(
            model,
            preprocessor,
            transactions_data
        )

        if not identified_patterns_df.empty:
            response_data = identified_patterns_df.to_dict(orient='records')
            return jsonify({"status": "success", "recurrent_patterns": response_data}), 200
        else:
            return jsonify({"status": "success", "message": "No recurrent patterns identified."}), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        # In a real application, you might want to log the full traceback for debugging
        return jsonify({"error": f"An internal error occurred during prediction: {e}"}), 500

# --- Server Startup Logic ---
def load_models():
    """Loads the pre-trained Keras model and scikit-learn preprocessor."""
    global model, preprocessor
    MODEL_PATH = "models/model.keras"
    PREPROCESSOR_PATH = "models/fitted_preprocessor.joblib"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Ensure it's in the same directory.")
        return False
    if not os.path.exists(PREPROCESSOR_PATH):
        print(f"Error: Preprocessor file '{PREPROCESSOR_PATH}' not found. Ensure it's in the same directory.")
        return False

    try:
        model = keras.models.load_model(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Model and preprocessor loaded successfully.")
        return True
    except Exception as e:
        print(f"Failed to load model or preprocessor: {e}")
        return False

if __name__ == "__main__":
    if load_models():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=3000)
    else:
        print("Server will not start due to model/preprocessor loading errors.")