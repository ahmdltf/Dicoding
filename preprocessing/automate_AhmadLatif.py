# ======================================
# AUTOMATE DATA PREPROCESSING
# ======================================

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ======================================
# LOAD DATASET
# ======================================
def load_data(path):

    # jika file tidak ada → download otomatis
    if not os.path.exists(path):
        print("Dataset tidak ditemukan. Mengunduh dataset...")

        url = "https://archive.ics.uci.edu/static/public/352/data.csv"

        os.makedirs(os.path.dirname(path), exist_ok=True)

        df = pd.read_csv(url)
        df.to_csv(path, index=False)

        print("Dataset berhasil diunduh.")

    # load dataset
    df = pd.read_csv(path)

    return df

# ======================================
# DATA CLEANING
# ======================================
def clean_data(df):
    """Membersihkan dataset"""

    # Hapus transaksi cancel
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    # Hapus missing customer
    df = df.dropna(subset=["CustomerID"])

    # Quantity valid
    df = df[df["Quantity"] > 0]

    return df


# ======================================
# FEATURE ENGINEERING
# ======================================
def feature_engineering(df):
    """Membuat fitur baru"""

    df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

    threshold = df["TotalSpend"].median()
    df["HighValueCustomer"] = (
        df["TotalSpend"] > threshold
    ).astype(int)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day

    df.drop("InvoiceDate", axis=1, inplace=True)

    return df


# ======================================
# ENCODING
# ======================================
def encode_data(df):
    """Encoding kategori"""

    le = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    return df


# ======================================
# SCALING
# ======================================
def scale_data(df):
    """Standarisasi fitur"""

    df.drop(
        ["InvoiceNo", "StockCode", "CustomerID"],
        axis=1,
        inplace=True
    )

    target = "HighValueCustomer"
    X_cols = df.columns.drop(target)

    scaler = StandardScaler()
    df[X_cols] = scaler.fit_transform(df[X_cols])

    return df


# ======================================
# SAVE DATA
# ======================================
def save_data(df, output_path):
    """Menyimpan dataset baru"""
    os.makedirs(output_path, exist_ok=True)

    df.to_csv(
        f"{output_path}/online_retail_clean.csv",
        index=False
    )


# ======================================
# MAIN PIPELINE
# ======================================
def main():

    input_path = "../online_retail_raw/online_retail.csv"
    output_path = "online_retail_preprocessing"

    df = load_data(input_path)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_data(df)
    df = scale_data(df)

    save_data(df, output_path)

    print("Preprocessing selesai!")


if __name__ == "__main__":
    main()
