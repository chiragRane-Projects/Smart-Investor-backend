import matplotlib.pyplot as plt
import pandas as pd
import uuid
import os
import matplotlib
matplotlib.use('Agg') 

def plot_predictions(y_test: pd.Series, predictions, ticker: str) -> str:
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(predictions, label="Predicted", linewidth=2)
    plt.title(f"{ticker.upper()} – Actual vs Predicted Closing Price")
    plt.xlabel("Days")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)

    # Save image to static dir with unique name
    plot_id = str(uuid.uuid4())
    filepath = f"static/{plot_id}.png"
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath
