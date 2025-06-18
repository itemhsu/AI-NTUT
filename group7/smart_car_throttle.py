import os
import pandas as pd

data_dir = r'c:\Users\user\Desktop\College\機器學習\train_data'
min_throttle = 1.3
max_throttle = 4.0

for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)
        
        if 'throttle' not in df.columns:
            print(f"{filename}: 沒有 'throttle' 欄位")
            continue
        
        out_of_bounds = df[(df['throttle'] < min_throttle) | (df['throttle'] > max_throttle)]
        if not out_of_bounds.empty:
            print(f"{filename}: 有 {len(out_of_bounds)} 筆 throttle 超出範圍")
            for idx, row in out_of_bounds.iterrows():
                episode = row['episode'] if 'episode' in row else 'N/A'
                steps = row['steps'] if 'steps' in row else 'N/A'
                print(f"[超出範圍] episode: {episode}, steps: {steps}, throttle: {row['throttle']}")
        else:
            print(f"{filename}: 所有 throttle 都在範圍內")