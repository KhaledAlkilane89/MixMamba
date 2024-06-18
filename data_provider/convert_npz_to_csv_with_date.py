import numpy as np
import pandas as pd

def convert_npz_to_csv_with_datetime_index(npz_file_path, data_key, start_date, timestep_minutes, csv_file_path):
    # Load the NPZ file
    npz_file = np.load(npz_file_path)

    # Extract the data array
    data = npz_file[data_key]

    # Reshape the data array to 2D if it has more than 2 dimensions
    if data.ndim > 2 and data.shape[2] > 0:
        data = data[:, :, 0]

    # Number of rows in the data
    num_rows = data.shape[0]

    # Generate a date range with the specified start date and timestep
    timestamps = pd.date_range(start=start_date, periods=num_rows, freq=f'{timestep_minutes}T')

    # Create a DataFrame with the timestamps as index
    df = pd.DataFrame(data, index=timestamps)

    # Print shape and Index
    # Print DataFrame details
    print(f"DataFrame shape: {df.shape}")
    print(f"Index range: {df.index.min()} to {df.index.max()}")
    print(f"Index type: {type(df.index)}")

    # Reset the index to make the datetime a column, and rename it to 'date'
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)

    # Convert 'date' column to string (object type)
    df['date'] = df['date'].astype(str)

    # Save to CSV
    df.to_csv(csv_file_path, index=False)

    print(f"File saved as '{csv_file_path}'.")


# Example usage
# convert_npz_to_csv_with_datetime_index('./dataset/PEMS/PEMS03.npz', 'data', '2012-01-05', 5, './dataset/PEMS/PEMS03.csv')
# convert_npz_to_csv_with_datetime_index('./dataset/PEMS/PEMS04.npz', 'data', '2017-01-07', 5, './dataset/PEMS/PEMS04.csv')
# convert_npz_to_csv_with_datetime_index('./dataset/PEMS/PEMS07.npz', 'data', '2017-01-05', 5, './dataset/PEMS/PEMS07.csv')
# convert_npz_to_csv_with_datetime_index('./dataset/PEMS/PEMS08.npz', 'data', '2012-01-03', 5, './dataset/PEMS/PEMS08.csv')
