# data.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def _data():
    # Đọc dữ liệu từ tệp CSV
    df = pd.read_csv(r'F:\Đồ Án\kag_risk_factors_cervical_cancer.csv', sep=';')

    # Thay thế "?" bằng NaN
    df.replace('?', np.nan, inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm thử
    X_data = df[['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs',
                  'STDs: Number of diagnosis', 'STDs:HPV']]
    y_data = df['Biopsy']
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, shuffle=True)
    print("bản ghi", df)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    _data()
