import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def plot_sweep():
    results_dir = "results"
    data = []
    for filename in os.listdir(results_dir):
        if filename.startswith("improved_") and filename.endswith(".json"):
            with open(os.path.join(results_dir, filename), 'r') as f:
                data.append(json.load(f))
    df = pd.DataFrame(data).sort_values(['strategy', 'compression_rate'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    strategies = df['strategy'].unique()
    colors = {'entropy': '#e74c3c', 'sentence': '#3498db', 'safety_aware': '#2ecc71', 'hybrid': '#9b59b6'}
    for strat in strategies:
        strat_data = df[df['strategy'] == strat]
        ax1.plot(strat_data['compression_rate'], strat_data['asr'] * 100, marker='o', label=strat, color=colors.get(strat))
        ax2.plot(strat_data['compression_rate'], strat_data['brr'] * 100, marker='s', label=strat, color=colors.get(strat))
    ax1.set_title("Safety Loss (ASR %)"); ax1.set_xlabel("Compression"); ax1.grid(True); ax1.legend()
    ax2.set_title("Usability Cost (BRR %)"); ax2.set_xlabel("Compression"); ax2.grid(True); ax2.legend()
    plt.savefig("final_results_plot.png")
    print("📈 Plot saved as 'final_results_plot.png'")

if __name__ == "__main__":
    plot_sweep()