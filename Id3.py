
from sklearn import tree
import pandas as pd
from data import _data
from sklearn.metrics import confusion_matrix

# Huấn luyện mô hình ID3
# X_train, X_test, y_train, y_test = _data()
# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf.fit(X_train, y_train)

# # Dự đoán trên tập kiểm thử
# y_pred_ID3 = clf.predict(X_test)

# # Tính Confusion Matrix
# cm_ID3 = confusion_matrix(y_test, y_pred_ID3)

# print("Confusion Matrix ID3:")
# print(cm_ID3)

def _ID3():
    X_train, X_test, y_train, y_test = _data()
    clf = tree.DecisionTreeClassifier(criterion='entropy', random_state= 0) #gọi hàm với hàm sai số là entropy
    clf.fit(X_train,y_train)
    y_pred_ID3=clf.predict(X_test)
    
    y_test = y_test.reset_index(drop=True)
    y_pred_ID3 = pd.Series(y_pred_ID3)
    cm_ID3 = confusion_matrix(y_test, y_pred_ID3)
    print("Confusion Matrix ID3:")
    print(cm_ID3)
    count_ID3 = 0
    for i in range(0,len(y_pred_ID3)):
        if(y_test[i] == y_pred_ID3[i]):
            count_ID3 = count_ID3 +1
    predictCorrectID3 = count_ID3/len(y_pred_ID3)
    print('Ty le du doan dung ID3: ',predictCorrectID3)
    print("zo",y_test)
    # Các độ đo 
    from sklearn.metrics import precision_score
   
    precision_score_ID3=precision_score(y_test, y_pred_ID3,average='macro')
    print('Độ chính xác Perceptron tính theo ID3: ',precision_score_ID3)

    from sklearn.metrics import recall_score
    recall_score_ID3=recall_score(y_test, y_pred_ID3,average='macro')
    print('Độ chính xác Recall tính theo ID3: ',recall_score_ID3)

    from sklearn.metrics import f1_score
    f1_score_ID3=f1_score(y_test, y_pred_ID3,average='macro')
    print('Độ chính xác F1 tính theo ID3: ',f1_score_ID3)
    return predictCorrectID3,precision_score_ID3, recall_score_ID3, f1_score_ID3
if __name__ == "__main__":
    _ID3()
