import pandas as pd
import sys
import numpy as np
import os.path as path

if __name__ == '__main__':
    file_path: str = sys.argv[1]
    if not path.isfile(file_path) or not file_path.endswith('.csv'):
        raise ValueError('Invalid file path! Must be a csv that exists.')

    df = pd.read_csv(file_path)
    print(f'Adding ptp\'s to {file_path}, with {len(df)} rows!')

    num_columns = 10
    for width in range(2, num_columns + 1):
        for i in range(num_columns - width + 1):
            key_name = f'ptp({",".join(f"col{j}" for j in range(i, i + width))})'
            print(f'Adding column: {key_name}')
            df[key_name] = df.loc[:, (f'col{j}' for j in range(i, i + width))].apply(np.ptp, axis=1)

    save_path = file_path.replace('.csv', '_local_ptp.csv')
    df.to_csv(save_path)
    print(f'Saved file with ptp\'s at {save_path}!')

else:
    raise Exception("This file should only be used as a script!")
