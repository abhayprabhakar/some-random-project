import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import logging

# Ensure an interactive backend is used on Linux
try:
    matplotlib.use('TkAgg') # Often works out of the box on Linux Python installations
except Exception:
    pass # Fallback to default if Tkinter isn't installed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def watch_and_plot():
    """
    Background script to constantly read the real-time metrics generated 
    by `train.py` and output an updating graph of the AI's learning curve.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    metrics_path = os.path.join(project_root, 'checkpoints', 'training_metrics.csv')
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    output_image = os.path.join(docs_dir, 'training_progress.png')
    
    logging.info(f"Watching for changes in: {metrics_path}")
    logging.info(f"Graph will dynamically render to: {output_image} and POP UP on screen.")
    logging.info("Keep this terminal open to watch the live-rendering GUI window!")
    
    last_modified = 0
    
    # Initialize interactive plotting window for native Linux GUI popup
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title('C-TimeGAN Real-Time Training Progress')
    
    while True:
        try:
            if not os.path.exists(metrics_path):
                plt.pause(2)
                continue
                
            current_modified = os.path.getmtime(metrics_path)
            
            # Only redraw if the trainer actually appended a new epoch
            if current_modified > last_modified:
                last_modified = current_modified
                
                df = pd.read_csv(metrics_path)
                
                # Minimum wait until we have at least 2 points to draw a curve
                if len(df) < 2:
                    plt.pause(2)
                    continue
                
                # Clear previous plot lines before re-drawing
                ax.clear()
                
                # WGAN Discriminator Loss (D_Loss) normally hovers/oscillates near 0 as it converges
                ax.plot(df['epoch'], df['d_loss'], label='Discriminator Loss', color='blue', linewidth=2)
                
                # Generator Loss (G_Loss)
                ax.plot(df['epoch'], df['g_loss'], label='Generator Loss', color='red', linewidth=2)
                
                # Autoencoder Loss
                if 'ae_loss' in df.columns:
                    ax.plot(df['epoch'], df['ae_loss'], label='Autoencoder Recon Loss', color='green', linewidth=1, linestyle='--')

                # Supervisor Loss
                if 'sup_loss' in df.columns:
                    ax.plot(df['epoch'], df['sup_loss'], label='Supervisor Loss', color='orange', linewidth=1, linestyle=':')

                # Generator Supervisor Loss
                if 'g_sup_loss' in df.columns:
                    ax.plot(df['epoch'], df['g_sup_loss'], label='Generator Sup Loss', color='purple', linewidth=1, linestyle='-.')
                
                ax.set_title('C-TimeGAN Real-Time Training Progress')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Adversarial Loss')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
                
                plt.tight_layout()
                
                # Update the interactive GUI window immediately
                plt.draw()
                # Still save the static backup image to docs/
                plt.savefig(output_image) 
                
                logging.info(f"Chart Rendered & Screen Updated! (Epoch {df['epoch'].max()})")
                
            # Important: Use plt.pause() instead of time.sleep() so the GUI window stays responsive and doesn't freeze
            plt.pause(5)
            
        except KeyboardInterrupt:
            logging.info("Plot watcher closed by user.")
            break
        except Exception as e:
            # If file is locked for a millisecond by train.py, just loop again safely
            plt.pause(2)

if __name__ == "__main__":
    watch_and_plot()