from Svc import _Svc
from Id3 import _ID3
from Cart import _Cart
from tabulate import tabulate

def main():
    predictCorrect,precision_score_svc, recall_score_svc, f1_score_svc = _Svc()
    predictCorrectID3,precision_score_ID3, recall_score_ID3, f1_score_ID3 = _ID3()
    predictCorrectCart,precision_score_Cart, recall_score_Cart, f1_score_Cart = _Cart()
    head=["Tên phương pháp","Tỷ lệ dự đoán đúng","presicion","Recall","F1_Score"]
    Mydata=[
        ["SVC", predictCorrect,precision_score_svc, recall_score_svc, f1_score_svc],
        ["ID3", predictCorrectID3,precision_score_ID3, recall_score_ID3, f1_score_ID3],
        ["Cart", predictCorrectCart,precision_score_Cart, recall_score_Cart, f1_score_Cart],
    ]

    print(tabulate(Mydata, headers=head,tablefmt="grids"))
if __name__ == "__main__":
    main()