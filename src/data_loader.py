import pandas as pd
import os

class DataLoader:
    """
    The 'Storyteller' for our data. This class is responsible for opening the books 
    (CSV files) and reading the raw digital history of our users.
    """
    def __init__(self, data_dir='data'):
        # We start our journey by looking for the 'data' folder.
        # To make this robust for different environments (like Jupyter), we 
        # check if 'data' is in the current folder or hiding in the parent directory.
        if not os.path.exists(data_dir) and os.path.exists(os.path.join('..', data_dir)):
            self.data_dir = os.path.join('..', data_dir) # Path correction for Notebooks
        else:
            self.data_dir = data_dir

    def load_data(self):
        """
        Gathers the three pillars of our study: Train, Val, and Test sets.
        """
        try:
            # We load the dataframes into memory to begin the analysis.
            train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
            val_df = pd.read_csv(os.path.join(self.data_dir, 'val.csv'))
            test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
            
            # Standardization: We rename usage time to a cleaner format to ensure 
            # all following features can refer to it with a consistent 'name'.
            for df in [train_df, val_df, test_df]:
                df.rename(columns={'Daily_Usage_Time (minutes)': 'Daily_Usage_Time'}, inplace=True)

            # Verification: Before we move forward, we ensure the 'Protagonists' of 
            # our story (mandatory columns) are present in the dataset.
            required_columns = ['User_ID', 'Dominant_Emotion', 'Daily_Usage_Time']
            for col in required_columns:
                if col not in train_df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            return train_df, val_df, test_df
        except FileNotFoundError as e:
            # If the data files aren't found, the story cannot begin.
            print(f"Error loading data: {e}")
            raise
        except Exception as e:
            # Catching unexpected twists in the data loading process.
            print(f"Unexpected error: {e}")
            raise
