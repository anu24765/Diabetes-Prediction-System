from django.shortcuts import render
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def home(request):
    return render(request, "home.html")


def predict(request):
    return render(request, "predict.html")


def result(request):
    data = pd.read_csv(r"/Users/anubhavk/Desktop/ML-Project/diabetes.csv")

    X = data.drop("Outcome", axis=1)
    y = data['Outcome']

    X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train.values, y_train.values)
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    res = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    if res == [1]:
        result1 = "positive"
    else:
        result1 = "negative"

    return render(request, "predict.html", {"result2": result1})
