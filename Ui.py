from tkinter import *
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv(r'F:\Đồ Án\kag_risk_factors_cervical_cancer.csv', sep=';')

    # Thay thế "?" bằng NaN
df.replace('?', np.nan, inplace=True)

    # Drop rows with missing values
# df.dropna(inplace=True)
# df.replace('?', np.nan, inplace=True)
    # Xử lý dữ liệu thiếu
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean())
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm thử
X_data = df[['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs',
                  'STDs: Number of diagnosis', 'STDs:HPV']]
y_data = df['Biopsy']
    # scaler = StandardScaler()
    # normalized_features = scaler.fit_transform(X_data)
scaler = StandardScaler()
normalized_features = scaler.fit_transform(X_data)    
X_train, X_test, y_train, y_test = train_test_split(normalized_features, y_data, test_size=0.2, random_state=42)
# svc = SVC( kernel='linear')
# svc.fit(X_train, y_train)

svc = SVC( kernel="linear")
svc.fit(X_train,y_train)


ID3= tree.DecisionTreeClassifier(criterion='entropy',max_depth=13,random_state=0)
ID3.fit(X_train,y_train)

CART = tree.DecisionTreeClassifier(criterion='gini',max_depth=13, random_state=0)
CART.fit(X_train,y_train)

DuDoanSVM=svc.predict(X_test)
DuDoanID3=ID3.predict(X_test)
DuDoanCART=CART.predict(X_test)
form = Tk()
form.title("Phân loại nguy cơ ung thư cổ tử cung ")
form.geometry("1150x800")
form.configure(bg='#52D3D8')

# GUI components
lable_TieuDe = Label(form, text="THÔNG TIN VỀ BỆNH", fg="black", font='Times 15 bold', bg='#52D3D8')
lable_TieuDe.grid(row=0, column=4, pady=20)

lable_age = Label(form, text="  Age  ", bg='#52D3D8', font='serif 12')
lable_age.grid(row=1, column=2, pady=20)
Cb_age = ttk.Entry(form, width=20, font='Times 15 bold')
Cb_age.grid(row=1, column=3)

lable_nospartners = Label(form, text="  Number of sexual partners   ", bg='#52D3D8', font='serif 12')
lable_nospartners.grid(row=2, column=2, pady=20)
Cb_nospartners = ttk.Entry(form, width=18, font='Times 15 bold')
Cb_nospartners.grid(row=2, column=3)

lable_fsintercourse = Label(form, text=" First sexual intercourse   ", bg='#52D3D8', font='serif 12')
lable_fsintercourse.grid(row=3, column=2, pady=20)
Cb_fsintercourse = ttk.Entry(form, width=20, font='Times 15 bold')
Cb_fsintercourse.grid(row=3, column=3)

lable_smoking = Label(form, text="   Smoking   ", bg='#52D3D8', font='serif 12')
lable_smoking.grid(row=4, column=2, pady=20)
Cb_smoking = ttk.Combobox(form, width=18, values=["0", "1"], font='Times 15 bold')
Cb_smoking.grid(row=4, column=3)

lable_nopregnancies = Label(form, text=" Num of pregnancies   ", bg='#52D3D8', font='serif 12')
lable_nopregnancies.grid(row=5, column=2, pady=20)
Cb_nopregnancies = ttk.Entry(form, width=20, font='Times 15 bold')
Cb_nopregnancies.grid(row=5, column=3)

lable_hcontraceptives = Label(form, text="Hormonal Contraceptives ", bg='#52D3D8', font='serif 12')
lable_hcontraceptives.grid(row=1, column=5, pady=20)
Cb_hcontraceptives = ttk.Combobox(form, width=18, values=["0", "1"], font='Times 15 bold')
Cb_hcontraceptives.grid(row=1, column=7)

lable_iud = Label(form, text="IUD", bg='#52D3D8', font='serif 12').grid(row=2, column=5, pady=20)
Cb_iud = ttk.Entry(form, width=20, font='Times 15 bold')
Cb_iud.grid(row=2, column=7)

lable_stds = Label(form, text="STDs ", bg='#52D3D8', font='serif 12').grid(row=3, column=5, pady=20)
Cb_stds = ttk.Entry(form, width=20, font='Times 15 bold')
Cb_stds.grid(row=3, column=7)

lable_nodiagnoses = Label(form, text="STDs (Number of diagnoses) ", bg='#52D3D8', font='serif 12').grid(row=4, column=5, pady=20)
Cb_nodiagnoses = ttk.Entry(form, width=20, font='Times 15 bold')
Cb_nodiagnoses.grid(row=4, column=7)

lable_stdsHPV = Label(form, text="STDs:HPV ", bg='#52D3D8', font='serif 12')
lable_stdsHPV.grid(row=5, column=5, pady=20)
Cb_stdsHPV = ttk.Combobox(form, width=18, values=["0", "1"], font='Times 15 bold')
Cb_stdsHPV.grid(row=5, column=7)

lable_KQSVM = Label(form, text="", bg='#52D3D8', font='serif 12 bold', pady=10)
lable_KQSVM.grid(row=9, column=3)
lable_KQNSVM = Label(form, text="", bg='#52D3D8', font='serif 12 bold', pady=10)
lable_KQNSVM.grid(row=11, column=3)

lable_KQID3 = Label(form, text="", bg='#52D3D8', font='serif 12 bold', pady=10)
lable_KQID3.grid(row=9, column=5)
lable_KQNID3 = Label(form, text="", bg='#52D3D8', font='serif 12 bold', pady=10)
lable_KQNID3.grid(row=11, column=5)

lable_KQCart = Label(form, text="", bg='#52D3D8', font='serif 12 bold', pady=10)
lable_KQCart.grid(row=9, column=9)
lable_KQNCart = Label(form, text="", bg='#52D3D8', font='serif 12 bold', pady=10)
lable_KQNCart.grid(row=11, column=9)

def DuDoanTestSVM():
    # DuDoanSVM = svc.predict(X_test)
    for i in range(len(y_test)):
        print(i, '   Dự đoán:  ', DuDoanSVM[i], "  \tThực tế:   ", y_test.iloc[i])

    # Display evaluation metrics
    lable_KQSVM.configure(text="Tỉ lệ dự đoán đúng của SVM theo các độ đo: " + '\n'
                               "Precision :  " + str(precision_score(y_test,DuDoanSVM,average='macro',zero_division=1)) + '\n'
                               "F1-Score  :  " + str(f1_score(y_test,DuDoanSVM,average='macro')) + '\n'
                               "Recall-Score  :  " + str(recall_score(y_test,DuDoanSVM,average='macro')) + '\n'
                               "accuracy  :  " + str(accuracy_score(y_test,DuDoanSVM)) + '\n')

button_DuDoanTestSVM = Button(form, text='Dự đoán test với SVM', command=DuDoanTestSVM, bg="#87CEEB",
                              font='serif 12').grid(row=8, column=3, pady=10)
def DuDoanTestID3():
    for i in range(len(y_test)):
        print(i, '   Dự đoán:  ', DuDoanSVM[i], "  \tThực tế:   ", y_test.iloc[i])


    # Display evaluation metrics
    lable_KQID3.configure(text="Tỉ lệ dự đoán đúng của ID3 theo các độ đo: " + '\n'
                               "Precision :  " + str(precision_score(y_test,DuDoanID3,average='macro',zero_division=1)) + '\n'
                               "F1-Score  :  " + str(f1_score(y_test,DuDoanID3,average='macro')) + '\n'
                               "Recall-Score  :  " + str(recall_score(y_test,DuDoanID3,average='macro')) + '\n'
                               "accuracy  :  " + str(accuracy_score(y_test,DuDoanID3)) + '\n')

button_DuDoanTest = Button(form, text='Dự đoán test với ID3', command=DuDoanTestID3, bg="#87CEEB",
                           font='serif 12').grid(row=8, column=5, pady=15)


def DuDoanTestCART():
    y_test_reset = y_test.reset_index(drop=True)  # Reset the index to avoid the KeyError
    count_cart = 0
    for i in range(0,len(DuDoanCART)):
        if y_test_reset[i] == DuDoanCART[i]:
            count_cart += 1
    predictCorrectCart = count_cart / len(DuDoanCART)
    print('Tỉ lệ dự đoán đúng của CART:', predictCorrectCart)

    # Display evaluation metrics
    lable_KQCart.configure(text="Tỉ lệ dự đoán đúng của CART theo các độ đo: " + '\n'
                               "Precision :  " + str(precision_score(y_test_reset, DuDoanCART,average='macro',zero_division=1)) + '\n'
                               "F1-Score  :  " + str(f1_score(y_test_reset, DuDoanCART,average='macro')) + '\n'
                               "Recall-Score  :  " + str(recall_score(y_test_reset, DuDoanCART,average='macro')) + '\n'
                               "Accuracy  :  " + str(predictCorrectCart) + '\n')

button_DuDoanTestCART = Button(form, text='Dự đoán test với CART', command=DuDoanTestCART, bg="#87CEEB",
                               font='serif 12').grid(row=8, column=9, pady=20)




def DuDoanNhapSVM():
    X_nhap = np.array(
        [Cb_age.get(), Cb_nospartners.get(), Cb_fsintercourse.get(), Cb_smoking.get(), Cb_nopregnancies.get(),
         Cb_hcontraceptives.get(),
         Cb_iud.get(), Cb_stds.get(), Cb_nodiagnoses.get(), Cb_stdsHPV.get()], dtype=float)
    print("Input Values:", X_nhap)  # Check the input values
    normalized_X = scaler.transform([X_nhap])
    print("Normalized Input:", normalized_X)  # Check the normalized input
    DuDoanNhapSVM = svc.predict(normalized_X)
    print("Prediction:", DuDoanNhapSVM)
    if DuDoanNhapSVM == 1:
        DuDoanSVM = 'bị ung thư cổ tử cung'
    else:
        DuDoanSVM = 'không bị ung thư cổ tử cung'

    lable_KQNSVM.configure(text="Dự đoán:  " + str(DuDoanSVM))


button_load = Button(form, text='Dự đoán nhập với SVM', command=DuDoanNhapSVM, bg="#87CEEB",
                     font='serif 12').grid(row=10, column=3, pady=10)


def DuDoanNhapID3():
    X_nhap = np.array(
        [Cb_age.get(), Cb_nospartners.get(), Cb_fsintercourse.get(), Cb_smoking.get(), Cb_nopregnancies.get(),
         Cb_hcontraceptives.get(),
         Cb_iud.get(), Cb_stds.get(), Cb_nodiagnoses.get(), Cb_stdsHPV.get()], dtype=float)
    print("Input Values:", X_nhap)  # Check the input values
    normalized_X = scaler.transform([X_nhap])
    print("Normalized Input:", normalized_X)  # Check the normalized input
    DuDoanNhapID3 = ID3.predict(normalized_X)
    print("Prediction:", DuDoanNhapID3)
    if DuDoanNhapID3 == 1:
        DuDoanNhapID3 = 'bị ung thư cổ tử cung'
    else:
        DuDoanNhapID3 = 'không bị ung thư cổ tử cung'

    lable_KQNID3.configure(text="Dự đoán:  " + str(DuDoanNhapID3))


button_load_ID3 = Button(form, text='Dự đoán nhập với ID3', command=DuDoanNhapID3, bg="#87CEEB",
                         font='serif 12').grid(row=10, column=5, pady=10)


def DuDoanNhapCart():
    X_nhap = np.array(
        [Cb_age.get(), Cb_nospartners.get(), Cb_fsintercourse.get(), Cb_smoking.get(), Cb_nopregnancies.get(),
         Cb_hcontraceptives.get(),
         Cb_iud.get(), Cb_stds.get(), Cb_nodiagnoses.get(), Cb_stdsHPV.get()], dtype=float)
    print("Input Values:", X_nhap)  # Check the input values
    normalized_X = scaler.transform([X_nhap])
    print("Normalized Input:", normalized_X)  # Check the normalized input
    DuDoanNhapCart = CART.predict(normalized_X)
    print("Prediction:", DuDoanNhapCart)
    if DuDoanNhapCart == 1:
        DuDoanNhapCart = 'bị ung thư cổ tử cung'
    else:
        DuDoanNhapCart = 'không bị ung thư cổ tử cung'

    lable_KQNCart.configure(text="Dự đoán:  " + str(DuDoanNhapCart))


button_load_Cart = Button(form, text='Dự đoán nhập với Cart', command=DuDoanNhapCart, bg="#87CEEB",
                         font='serif 12').grid(row=10, column=9, pady=10)
form.mainloop()