import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple

def remove_columns(df: pd.DataFrame, cols_to_drop: List[str]) -> pd.DataFrame:
    """
    Remove specified columns from the DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame.
        cols_to_drop (List[str]): List of column names to drop.

    Returns:
        pd.DataFrame: DataFrame with specified columns removed.
    """
    return df.drop(columns=cols_to_drop, inplace=False)


def split_train_val(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training and validation sets.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        target_col (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the validation set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing training and validation DataFrames.
    """
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df[target_col])
    return train_df, val_df

def create_inputs_targets(train_df: pd.DataFrame, val_df: pd.DataFrame,) -> Tuple[List[str], str, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Define input, target columns.
    Create input and target DataFrames from the given DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        Tuple[List[str], str, pd.DataFrame pd.Series, pd.DataFrame, pd.Series]: Input columns, target column, inputs and targets.
    """
    input_cols = list(train_df.columns)[:-1]
    target_col = list(train_df.columns)[-1]

    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()

    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()

    return input_cols, target_col, train_inputs, train_targets, val_inputs, val_targets

def identify_numeric_categorical(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns in the given DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        Tuple[List[str], List[str]]: Numeric columns and categorical columns.
    """
    numeric_cols = df.select_dtypes(include = np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include = 'object').columns.tolist()
    
    return numeric_cols, categorical_cols

def scale_numeric_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, numeric_cols: List[str], scaler: MinMaxScaler) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale numeric columns using MinMaxScaler.
    
    Args:
        train_inputs (pd.DataFrame): Training inputs.
        val_inputs (pd.DataFrame): Validation inputs.
        numeric_cols (List[str]): List of numeric columns.
        scaler (MinMaxScaler): Scaler instance.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]: Scaled training and validation inputs, and the scaler used.
    """
    scaler.fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    
    return train_inputs, val_inputs, scaler

def encode_categorical_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, categorical_cols: List[str], encoder: OneHotEncoder) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], OneHotEncoder]:
    """
    Encode categorical columns using OneHotEncoder.
    
    Args:
        train_inputs (pd.DataFrame): Training inputs.
        val_inputs (pd.DataFrame): Validation inputs.
        categorical_cols (List[str]): List of categorical columns.
        encoder (OneHotEncoder): Encoder instance.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str], OneHotEncoder]: Encoded training and validation inputs, list of encoded column names, and the encoder used.
    """
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    return train_inputs, val_inputs, encoded_cols, encoder

def preprocess_data(raw_df: pd.DataFrame, drop_cols: List[str], scaler_numeric: bool) -> Tuple[Dict[str, pd.DataFrame], List[str], MinMaxScaler, OneHotEncoder]:
    """
    Preprocess the raw DataFrame for machine learning.
    
    Args:
        raw_df (pd.DataFrame): The raw input DataFrame.
        drop_cols (List[str]): List of columns to drop.
        scaler_numeric (bool): Flag to determine if numeric features should be scaled.
    
    Returns:
        Tuple[Dict[str, pd.DataFrame], List[str], MinMaxScaler, OneHotEncoder]: A dictionary containing preprocessed training and validation data, the list of input columns, the scaler used and the encoder used.
    """
   
    # Removing unnecessary columns
    raw_df = remove_columns(raw_df, drop_cols)

    # Create training and validation sets
    train_df, val_df = split_train_val(raw_df, target_col="Exited")
        
    # Create inputs and targets
    input_cols, target_col, train_inputs, train_targets, val_inputs, val_targets = create_inputs_targets(train_df, val_df)
        
    # Identify numeric and categorical columns
    numeric_cols, categorical_cols = identify_numeric_categorical(train_inputs)
            
    # Scale numeric features
    if scaler_numeric:
        scaler = MinMaxScaler()
        train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols, scaler)
    else:
        scaler = None
  
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    train_inputs, val_inputs, encoded_cols, encoder = encode_categorical_features(train_inputs, val_inputs, categorical_cols, encoder)
    
    # Final training and validation sets
    X_train = train_inputs[numeric_cols + encoded_cols]
    X_val = val_inputs[numeric_cols + encoded_cols]
    
    return {
        'train_X': X_train,
        'train_y': train_targets,
        'val_X': X_val,
        'val_y': val_targets
    }, input_cols, scaler, encoder

def preprocess_new_data(new_raw_df: pd.DataFrame, drop_cols: List[str], scaler_numeric: bool, scaler: MinMaxScaler, encoder: OneHotEncoder) -> Dict[str, pd.DataFrame]:
    """
    Preprocess new data by removing columns, scaling numeric features, and encoding categorical features.
    
    Args:
        new_raw_df (pd.DataFrame): The new raw input DataFrame.
        drop_cols (List[str]): List of columns to drop.
        scaler_numeric (bool): Flag to determine if numeric features should be scaled.
        scaler (MinMaxScaler): Fitted scaler to apply to numeric features.
        encoder (OneHotEncoder): Fitted encoder to apply to categorical features.
    
    Returns:
        Dict[str, pd.DataFrame]: Processed test data.
    """
    # Removing unnecessary columns
    new_raw_df = remove_columns(new_raw_df, drop_cols)
    
    # Identify numeric and categorical columns
    numeric_cols = new_raw_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = new_raw_df.select_dtypes(include='object').columns.tolist()

    # Scale numeric features
    if scaler_numeric:
        new_raw_df[numeric_cols] = scaler.transform(new_raw_df[numeric_cols])
    
    # One-hot encode categorical features
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    new_raw_df[encoded_cols] = encoder.transform(new_raw_df[categorical_cols])

    # Final X_test set
    X_test = new_raw_df[numeric_cols + encoded_cols]
    
    return {'test_X': X_test}


# import warnings
# warnings.filterwarnings("ignore")

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import OneHotEncoder

# def preprocess_data(raw_df, drop_cols, scaler_numeric):
    
#     # Removing unnecessary columns    
#     for col in drop_cols:
#         raw_df.drop(columns = col, inplace = True)
    
#     # Create training and validation sets
#     train_df, val_df = train_test_split(raw_df, test_size = 0.25, random_state = 42, stratify = raw_df.Exited)

#     # Create inputs and targets
#     input_cols = list(train_df.columns)[:-1]
#     target_col = list(train_df.columns)[-1]

#     train_inputs = train_df[input_cols].copy()
#     train_targets = train_df[target_col].copy()

#     val_inputs = val_df[input_cols].copy()
#     val_targets = val_df[target_col].copy()

#     # Identify numeric and categorical columns
#     numeric_cols = train_inputs.select_dtypes(include = np.number).columns.tolist()
#     categorical_cols = train_inputs.select_dtypes(include = 'object').columns.tolist()

#     # Scale numeric features
#     if scaler_numeric == True:
#         scaler = MinMaxScaler()
#         scaler.fit(train_inputs[numeric_cols])
#         train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
#         val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
#     else:
#         scaler = None

#     # One-hot encode categorical features
#     encoder = OneHotEncoder(sparse = False, handle_unknown = 'ignore')
#     encoder.fit(train_inputs[categorical_cols])
#     encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
#     train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
#     val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

#     # Final training and validation sets
#     X_train = train_inputs[numeric_cols + encoded_cols]
#     X_val = val_inputs[numeric_cols + encoded_cols]
    
#     result = {
#         'train_X': X_train,
#         'train_y': train_targets,
#         'val_X': X_val,
#         'val_y': val_targets
#     }
#     return result, input_cols, scaler, encoder

# def preprocess_new_data(new_raw_df, drop_cols, scaler_numeric, scaler, encoder):
    
#     # Removing unnecessary columns    
#     for col in drop_cols:
#         new_raw_df.drop(columns = col, inplace = True)
    
#     # Identify numeric and categorical columns
#     numeric_cols = new_raw_df.select_dtypes(include = np.number).columns.tolist()
#     categorical_cols = new_raw_df.select_dtypes(include = 'object').columns.tolist()

#     # Scale numeric features
#     if scaler_numeric == True:
#         new_raw_df[numeric_cols] = scaler.transform(new_raw_df[numeric_cols])
    
#     # One-hot encode categorical features
#     encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
#     new_raw_df[encoded_cols] = encoder.transform(new_raw_df[categorical_cols])
    
#     #Final X_test set
#     X_test = new_raw_df[numeric_cols + encoded_cols]
    
#     result = {
#         'test_X': X_test
#     }
    
#     return result