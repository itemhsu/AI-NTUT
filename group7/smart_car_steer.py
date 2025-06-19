import os
import pandas as pd

data_dir = "c:/Users/user/Desktop/College/機器學習/train_data"
min_steer, max_steer = -30, 30

for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)
        if 'steer' not in df.columns:
            print(f"{filename}: 沒有 'steer' 欄位")
            continue
        out_of_bounds = df[(df['steer'] < min_steer) | (df['steer'] > max_steer)]
        if not out_of_bounds.empty:
            print(f"{filename}: 有 {len(out_of_bounds)} 筆 steer 超出範圍")
            for idx, row in out_of_bounds.iterrows():
                episode = row['episode'] if 'episode' in row else 'N/A'
                steps = row['steps'] if 'steps' in row else 'N/A'
                print(f"[超出範圍] episode: {episode}, steps: {steps}, steer: {row['steer']}")
        else:
            print(f"{filename}: 所有 steer 都在範圍內")