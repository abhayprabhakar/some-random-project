import numpy as np
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SequenceGenerator:
    """
    Converts independent, preprocessed tabular network flows into
    time-series sequences for LSTM ingestion.
    """
    def __init__(self, sequence_length: int = 20, stride: int = 10):
        self.sequence_length = sequence_length
        self.stride = stride

    def create_windows(self, df: pd.DataFrame, label_col: str = 'Attack Type') -> tuple:
        """
        Slides a window across the dataframe to create 3D tensors.
        
        Args:
            df (pd.DataFrame): Fully preprocessed, numerical DataFrame.
            label_col (str): The column name representing the conditional class/label.
            
        Returns:
            X (np.ndarray): Tensor of shape (num_samples, seq_length, num_features)
            y (np.ndarray): Array of labels for each sequence
        """
        logging.info(f"Generating sliding windows (Length: {self.sequence_length}, Stride: {self.stride})")
        
        dataset_values = df.values
        
        # Identify index of the conditional label
        if label_col in df.columns:
            label_idx = df.columns.get_loc(label_col)
        else:
            logging.warning(f"Label column '{label_col}' not found. Cannot extract targets.")
            label_idx = -1

        X_sequences = []
        y_labels = []

        num_records = len(df)
        for i in range(0, num_records - self.sequence_length + 1, self.stride):
            window = dataset_values[i : i + self.sequence_length]
            X_sequences.append(window)
            
            if label_idx != -1:
                # Target conditional label: we assign the "majority" attack class within this time window
                # or simply take the most aggressive class. For C-TimeGAN, picking the mode is standard.
                window_labels = window[:, label_idx]
                # Cast to integer just in case, then find the most frequent label
                mode_label = np.bincount(window_labels.astype(int)).argmax()
                y_labels.append(mode_label)

        X = np.array(X_sequences)
        y = np.array(y_labels)

        logging.info(f"Success! Generated X shape: {X.shape}, y shape: {y.shape}")
        return X, y

if __name__ == "__main__":
    from data_parser import NIDDParser
    from preprocessor import DataPreprocessor
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    dataset_path = os.path.join(project_root, "data", "raw", "Combined.csv")
    
    # 1. Parse Data
    parser = NIDDParser(dataset_path)
    df = parser.load_data()
    
    # Take a 50,000 line sample for the sequence testing
    df_sample = df.head(50000).copy()
    
    # 2. Preprocess (Scale & Encode)
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df_sample)
    
    # 3. Generate Sequences!
    seq_gen = SequenceGenerator(sequence_length=20, stride=5)
    X, y = seq_gen.create_windows(df_processed, label_col='Attack Type')
    
    # Ready for PyTorch / TensorFlow LSTMs
    logging.info(f"First sequence matrix snippet:\n{X[0][:2, :5]}")
