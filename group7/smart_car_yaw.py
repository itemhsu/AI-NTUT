import os
import pandas as pd
import matplotlib.pyplot as plt

data_folder = r"c:\Users\user\Desktop\College\機器學習\train_data"
csv_files = [f for f in os.listdir(data_folder) if f.endswith('-iteration.csv')]

for file in csv_files:
    file_path = os.path.join(data_folder, file)
    data = pd.read_csv(file_path)
    
    # 取得所有不同的 episode
    unique_episodes = data['episode'].unique()
    num_plots = len(unique_episodes)
    cols = 5
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten()

    for i, episode in enumerate(unique_episodes):
        episode_data = data[data['episode'] == episode]
        axes[i].plot(episode_data['steps'], episode_data['yaw'], label=f"Episode {episode}")
        axes[i].set_title(f"Episode {episode}")
        axes[i].set_xlabel("Steps")
        axes[i].set_ylabel("Yaw")
        axes[i].legend(loc="upper left", fontsize="small")
        axes[i].grid(True)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f"Yaw Values for All Episodes in {file}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()