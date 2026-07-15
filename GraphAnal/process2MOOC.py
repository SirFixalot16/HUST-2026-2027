import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def convert_with_rich_features(input_csv, output_prefix):
    # 1. Load Mendeley Dataset
    df = pd.read_csv(input_csv)
    
    # 2. Map Nodes to Integers (Continuous-Time Dynamic Graph Format) [2]
    all_entities = pd.concat([df['customer_id'], df['merchant_id']]).unique()
    id_map = {original_id: i for i, original_id in enumerate(all_entities)}
    df['src'] = df['customer_id'].map(id_map)
    df['dst'] = df['merchant_id'].map(id_map)
    
    # 3. Convert ISO 8601 Timestamps to Continuous Floats [2, 3]
    df['time_dt'] = pd.to_datetime(df['timestamp'])
    start_time = df['time_dt'].min()
    df['time'] = (df['time_dt'] - start_time).dt.total_seconds()
    
    # 4. Sort Chronologically (Required for Sequential ODE Aggregator) [4, 5]
    df = df.sort_values('time').reset_index(drop=True)
    
    # 5. Extract Rich Features (Behavioral & Geographic) [3]
    # We'll encode 'amount', 'category', 'lat', and 'long'
    scaler = StandardScaler()
    numeric_features = df[['amount', 'merchant_latitude', 'merchant_longitude']].values
    numeric_features = scaler.fit_transform(numeric_features)
    
    cat_enc = LabelEncoder()
    category_features = cat_enc.fit_transform(df['merchant_category']).reshape(-1, 1)
    
    # Combine features
    combined = np.hstack([numeric_features, category_features])
    
    # 6. Pad to 128 Dimensions (to match MOOC project baseline) [1]
    feature_dim = 128
    num_links = len(df)
    final_features = np.zeros((num_links, feature_dim), dtype=np.float32)
    # Fill the first few dimensions with actual data, leave rest as 0 or small noise
    final_features[:, :combined.shape[1]] = combined
    
    # 7. Format edges.csv (Topological Stream)
    # int_roll maps to fraud label; ext_roll maps to category index
    edges_df = pd.DataFrame({
        'src': df['src'],
        'dst': df['dst'],
        'time': df['time'],
        'int_roll': df['is_fraud'],
        'ext_roll': category_features.flatten()
    })
    
    # 8. Export Files
    edges_df.to_csv(f"{output_prefix}_edges.csv") # Includes index as in MOOC sample
    np.save(f"{output_prefix}_features.npy", final_features)
    
    print(f"Success: Exported {num_links} links with {feature_dim}-dim features.")

convert_with_rich_features('FDD.csv', 'MOOC_0.3')