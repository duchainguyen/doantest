
from sklearn import tree
import pandas as pd
from data import _data
from sklearn.metrics import confusion_matrix

def _Cart():
    X_train, X_test, y_train, y_test = _data()
    #Cart
    clf1 = tree.DecisionTreeClassifier(criterion='gini')
    clf1.fit(X_train,y_train)
    y_pred_Cart=clf1.predict(X_test)
    y_test = y_test.reset_index(drop=True)
    y_pred_Cart = pd.Series(y_pred_Cart)
    cm_ID3 = confusion_matrix(y_test, y_pred_Cart)
    print("Confusion Matrix CART:")
    print(cm_ID3)
    count_cart = 0
    for i in range(0,len(y_pred_Cart)):
        if(y_test[i] == y_pred_Cart[i]):    
            count_cart = count_cart +1
    predictCorrectCart= count_cart/len(y_pred_Cart)
    print('Ty le du doan dung Cart: ',predictCorrectCart )

    # Các độ đo 
    from sklearn.metrics import precision_score
   
    precision_score_Cart=precision_score(y_test, y_pred_Cart,average='macro')
    print('Độ chính xác Perceptron tính theo Cart: ',precision_score_Cart)

    from sklearn.metrics import recall_score
    recall_score_Cart=recall_score(y_test, y_pred_Cart,average='macro')
    print('Độ chính xác Recall tính theo Cart: ',recall_score_Cart)

    from sklearn.metrics import f1_score
    f1_score_Cart=f1_score(y_test, y_pred_Cart,average='macro')
    print('Độ chính xác F1 tính theo Cart: ',f1_score_Cart)
    return predictCorrectCart,precision_score_Cart, recall_score_Cart, f1_score_Cart
if __name__ == "__main__":
    _Cart()
