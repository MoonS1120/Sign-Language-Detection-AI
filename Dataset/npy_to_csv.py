import numpy as np
import pandas as pd
import os

dfs = []

for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
    folder = f"train_data/{letter}"
    files = os.listdir(folder)
    
    for numpy_file in files:
        data = np.load(f"train_data/{letter}/{numpy_file}")
        data = data.reshape(1, -1)
        labels = [letter]
        columns = [f"ver.{coord}{i}" for i in range(21) for coord in ["x", "y", "z"]]
        
        df = pd.DataFrame(
            data=data, 
            columns= columns
        )
        df["label"] = labels
        df = df[["label"] + [col for col in df.columns if col != "label"]]

        dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv('train_data.csv', index=False)