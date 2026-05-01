from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MLEfficacyTester:
    """
    Implements Train on Synthetic, Test on Real (TSTR) and 
    Train on Real, Test on Real (TRTR) baseline metric.
    """
    
    def __init__(self):
        # We use a standard Random Forest for robust benchmark classification
        self.clf = RandomForestClassifier(n_estimators=50, random_state=42)

    def evaluate_tstr(self, X_synth_train: np.ndarray, y_synth_train: np.ndarray, 
                      X_real_test: np.ndarray, y_real_test: np.ndarray) -> float:
        """
        Train Machine Learning model (IDS) on the AI GENERATED data, 
        but strictly test it on the unseen REAL 5G-NIDD data.
        If generation is good, F1-score will be high.
        """
        logging.info("Training IDS classifier purely on SYNTHETIC data...")
        self.clf.fit(X_synth_train, y_synth_train)
        
        logging.info("Evaluating IDS classifier on REAL 5G-NIDD data...")
        y_pred = self.clf.predict(X_real_test)
        
        score = f1_score(y_real_test, y_pred, average='weighted')
        logging.info(f"TSTR Weighted F1-Score: {score:.4f}")
        
        logging.info("\n" + classification_report(y_real_test, y_pred))
        return score

if __name__ == "__main__":
    # Quick Test Stub
    synth_x = np.random.rand(500, 10)
    synth_y = np.random.randint(0, 2, 500)
    
    real_x = np.random.rand(200, 10)
    real_y = np.random.randint(0, 2, 200)
    
    tester = MLEfficacyTester()
    tester.evaluate_tstr(synth_x, synth_y, real_x, real_y)
