import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(filepath):

    # --------------------------------
    # Load dataset (auto separator)
    # --------------------------------
    try:
        df = pd.read_csv("data/bank-additional.csv", sep=';')

        # If only one column, try comma
        if df.shape[1] == 1:
            df = pd.read_csv(filepath)

    except Exception as e:
        print("Error loading dataset:", e)
        return None, None, None

    print("\nFirst 5 Rows")
    print(df.head())

    print("\nDataset Shape:", df.shape)

    print("\nColumn Names")
    print(df.columns)

    print("\nMissing Values")
    print(df.isnull().sum())

    # --------------------------------
    # Encode categorical columns
    # --------------------------------
    label_encoders = {}

    categorical_columns = df.select_dtypes(
        include=['object']
    ).columns

    for col in categorical_columns:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

        label_encoders[col] = le

    # --------------------------------
    # Features and Target
    # --------------------------------
    X = df.drop("y", axis=1)
    y = df["y"]

    return X, y, label_encoders