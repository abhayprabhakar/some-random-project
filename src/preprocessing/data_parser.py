import pandas as pd
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NIDDParser:
    """
    Handles the ingestion and preliminary cleaning of the raw 5G-NIDD tabular data.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Loads the dataset into a Pandas DataFrame."""
        try:
            logging.info(f"Loading data from {self.file_path}...")
            # For demonstration, we assume a CSV format for the extracted NetFlow/tabular 5G-NIDD data
            self.raw_data = pd.read_csv(self.file_path)
            logging.info(f"Data successfully loaded. Shape: {self.raw_data.shape}")
            return self.raw_data
        except FileNotFoundError:
            logging.error(f"Dataset not found at {self.file_path}. Please place the 5G-NIDD dataset there.")
            raise
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def get_basic_stats(self):
        """Returns basic distribution of labels to check for class imbalance."""
        if self.raw_data is None:
            logging.warning("Data not loaded yet. Call load_data() first.")
            return None
        
        # We assume the label column is traditionally named 'Label' or 'Attack_Type'
        label_col = [col for col in self.raw_data.columns if 'label' in col.lower() or 'attack' in col.lower()]
        if label_col:
            stats = self.raw_data[label_col[0]].value_counts()
            logging.info(f"Class distribution:\n{stats}")
            return stats
        else:
            logging.warning("Could not automatically locate the target label column.")
            return None

if __name__ == "__main__":
    import os
    
    # Get the absolute path to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    # Target our newly available Combined.csv file
    dataset_path = os.path.join(project_root, "data", "raw", "Combined.csv")
    
    parser = NIDDParser(dataset_path)
    try:
        df = parser.load_data()
        
        # Display the found columns to verify everything looks correct
        logging.info("Dataset Columns:")
        logging.info(list(df.columns))
        
        # Show class distribution
        parser.get_basic_stats()
    except Exception as e:
        logging.error("Failed to execute parser script.")

