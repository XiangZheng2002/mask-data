import pandas as pd
import os
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    app_path = '/zhengxiang/control_dataset/PDE-Control/legacy/apps/'
    parser.add_argument('--path', type=str, default='Debug_1118/smoke_out_compare', help='Directory containing the CSV files')
    parser.add_argument('--save_path', type=str, default='analysis_param', help='Place to save JSON files')
    
    
    args = parser.parse_args()
    csv_dir = os.path.join(app_path, args.path)
    save_dir = os.path.join(app_path, args.save_path)

    mean_ground_sum = {}

    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_dir, filename)
            
            df = pd.read_csv(file_path)
            
            mean_value = df['ground rate'].mean()
            if mean_value > 0.15:
                mean_ground_sum[filename] = mean_value

    # Display the results
    for filename, mean_value in mean_ground_sum.items():
        print(f"{filename}: Mean of ground rate = {mean_value}")

    save_path = os.path.join(save_dir, 'mean_ground rate.json')
    with open(save_path, 'w') as json_file:
        json.dump(mean_ground_sum, json_file)

