# scripts/data_loader.py
import pandas as pd

def load_data(path: str, frac: float = 1.0) -> pd.DataFrame:
    """
    Loads data from a CSV file and optionally samples it.
    
    Args:
        path (str): The path to the CSV file.
        frac (float): The fraction of the dataset to sample. Defaults to 1.0 (full dataset).
    
    Returns:
        pd.DataFrame: The loaded (and sampled) DataFrame.
    """
    df = pd.read_csv(path)
    if frac < 1.0:
        df = df.sample(frac=frac, random_state=42)
    df.reset_index(drop=True, inplace=True)
    return df 
