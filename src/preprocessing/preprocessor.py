import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    """
    Handles cleaning, scaling, and categorical encoding for the 5G-NIDD dataset 
    to prepare it for the C-TimeGAN generation pipeline.
    """
    def __init__(self):
        self.continuous_scalers = {}
        self.categorical_encoders = {}
        self.categorical_cols = []
        self.continuous_cols = []
        self.feature_columns = []
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes identifier columns, handles mixed data types, and deals with NaN values.
        """
        logging.info("Cleaning dataset (removing empty rows, filling strict Nulls)...")
        time_cols = [
            col for col in ['Seq', 'StartTime', 'StartTimeMs', 'StartTimeMillis', 'Timestamp', 'ts']
            if col in df.columns
        ]
        if time_cols:
            df = df.sort_values(time_cols[0]).reset_index(drop=True)
            logging.info(f"Sorted rows by chronological column: {time_cols[0]}")
        # Drop irrelevant id/index columns if they exist
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        if 'Seq' in df.columns:
            df = df.drop(columns=['Seq'])
            
        # These are commonly string attributes or discrete features in Argus flow data
        self.categorical_cols = ['Proto', 'Cause', 'State', 'Label', 'Attack Type', 'Attack Tool', 'sDSb', 'dDSb']
        # Keeping only columns that exist in the provided dataframe
        self.categorical_cols = [col for col in self.categorical_cols if col in df.columns]
        
        # All other columns are assumed continuous/numerical
        self.continuous_cols = [col for col in df.columns if col not in self.categorical_cols]
        
        # Resolve mixed datatypes (Force categorical columns to string type)
        for col in self.categorical_cols:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace('nan', 'Unknown')
            
        # Convert numerical fields safely
        for col in self.continuous_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Fill strictly computational NaNs with 0 (since in network flow, missing count/bytes is generally 0)
        df[self.continuous_cols] = df[self.continuous_cols].fillna(0)
        
        logging.info(f"Identified {len(self.continuous_cols)} continuous and {len(self.categorical_cols)} categorical features.")
        return df

    def scale_continuous(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies MinMax scaling to continuous variables (Log transformation could also be applied here for byte counts)."""
        logging.info("Scaling continuous features using MinMaxScaler...")
        for col in self.continuous_cols:
            # We use MinMaxScaler because GANs (with Tanh/Sigmoid activations) generally prefer data in [0,1] or [-1,1]
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.continuous_scalers[col] = scaler
        return df
        
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label encodes categorical strings into integers. 
        Note: The C-TimeGAN's embedding layer will take these integers and output dense vectors later."""
        logging.info("Encoding categorical features...")
        for col in self.categorical_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            self.categorical_encoders[col] = encoder
            logging.info(f"Column '{col}' maps to {len(encoder.classes_)} unique categories.")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the entire preprocessing pipeline."""
        df = self.clean_data(df)
        df = self.scale_continuous(df)
        df = self.encode_categorical(df)
        self.feature_columns = df.columns.tolist()
        return df

    def save_pipeline(self, output_dir: str):
        """Saves scalers and encoders so we can inverse_transform the generated synthetic data later."""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.continuous_scalers, os.path.join(output_dir, 'continuous_scalers.pkl'))
        joblib.dump(self.categorical_encoders, os.path.join(output_dir, 'categorical_encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(output_dir, 'feature_columns.pkl'))
        joblib.dump(self.continuous_cols, os.path.join(output_dir, 'continuous_cols.pkl'))
        joblib.dump(self.categorical_cols, os.path.join(output_dir, 'categorical_cols.pkl'))
        logging.info(f"Preprocessing pipeline artifacts saved to {output_dir}.")

if __name__ == "__main__":
    from data_parser import NIDDParser
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # 1. Load small subset for quick test
    dataset_path = os.path.join(project_root, "data", "raw", "Combined.csv")
    parser = NIDDParser(dataset_path)
    df = parser.load_data()
    
    # Let's take just the first 10k rows to verify the pipeline doesn't crash 
    df_sample = df.head(10000).copy()
    
    # 2. Run Preprocessor
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df_sample)
    
    # 3. Save the scalers (mock path)
    artifact_dir = os.path.join(project_root, "models", "artifacts")
    preprocessor.save_pipeline(artifact_dir)
    
    logging.info("Preprocessing complete. Output sample:")
    logging.info(df_processed[['TotBytes', 'SrcBytes', 'Proto', 'Attack Type']].head())
