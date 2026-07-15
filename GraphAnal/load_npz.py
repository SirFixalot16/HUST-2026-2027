import numpy as np

# 1. Load the archive safely using a context manager
with np.load('GSNOP/code/DATA/MOOC_0.3/int_train.npz') as data:
    
    # 2. View all available array names (keys) inside the file
    print("Arrays in file:", data.files)
    
    # 3. Access a specific array using its key
    arr = ['indptr', 'indices', 'ts', 'eid']
    my_array = data[arr[3]]  # Replace 'arr_0' with your actual key
    
    # 4. View properties and contents of the array
    print("Shape:", my_array.shape)
    print("Data Type:", my_array.dtype)
    print("Contents:\n", my_array)
