from django.db.models import Avg
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
import os

from Remote_User.models import (
    ClientRegister_Model,
    cyber_attack_detection,
    detection_ratio,
    detection_accuracy
)

# ---------------------------------------------
# ADMIN LOGIN
# ---------------------------------------------
def serviceproviderlogin(request):
    if request.method == "POST":
        admin = request.POST.get("username")
        password = request.POST.get("password")
        if admin == "Admin" and password == "Admin":
            detection_accuracy.objects.all().delete()
            return redirect("View_Remote_Users")
    return render(request, "SProvider/serviceproviderlogin.html")


# ---------------------------------------------
# USER LIST
# ---------------------------------------------
def View_Remote_Users(request):
    return render(
        request,
        "SProvider/View_Remote_Users.html",
        {"objects": ClientRegister_Model.objects.all()}
    )


# ---------------------------------------------
# PREDICTION RATIO
# ---------------------------------------------
def View_Prediction_Of_Cyber_Attack_Type_Ratio(request):

    detection_ratio.objects.all().delete()

    def calc_ratio(keyword):
        total = cyber_attack_detection.objects.count()
        if total == 0:
            return
        count = cyber_attack_detection.objects.filter(Prediction=keyword).count()
        ratio = (count / total) * 100
        if ratio > 0:
            detection_ratio.objects.create(names=keyword, ratio=ratio)

    for key in ["DDoS", "Intrusion", "Malware"]:
        calc_ratio(key)

    return render(
        request,
        "SProvider/View_Prediction_Of_Cyber_Attack_Type_Ratio.html",
        {"objs": detection_ratio.objects.all()}
    )


# ---------------------------------------------
# CHARTS
# ---------------------------------------------
def charts(request, chart_type):
    chart = detection_ratio.objects.values("names").annotate(dcount=Avg("ratio"))
    return render(request, "SProvider/charts.html", {"form": chart, "chart_type": chart_type})


def charts1(request, chart_type):
    chart = detection_accuracy.objects.values("names").annotate(dcount=Avg("ratio"))
    return render(request, "SProvider/charts1.html", {"form": chart, "chart_type": chart_type})


def likeschart(request, like_chart):
    chart = detection_accuracy.objects.values("names").annotate(dcount=Avg("ratio"))
    return render(request, "SProvider/likeschart.html", {"form": chart, "like_chart": like_chart})


# ---------------------------------------------
# VIEW PREDICTIONS
# ---------------------------------------------
def View_Prediction_Of_Cyber_Attack_Type(request):
    return render(
        request,
        "SProvider/View_Prediction_Of_Cyber_Attack_Type.html",
        {"list_objects": cyber_attack_detection.objects.all()}
    )


# ---------------------------------------------
# DOWNLOAD PREDICTED DATA
# ---------------------------------------------
def Download_Predicted_DataSets(request):

    from openpyxl import Workbook  # safe here

    response = HttpResponse(
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response["Content-Disposition"] = 'attachment; filename="Predicted_Datasets.xlsx"'

    wb = Workbook()
    ws = wb.active
    ws.title = "Predicted"

    headers = [
        field.name for field in cyber_attack_detection._meta.fields
    ]
    ws.append(headers)

    for obj in cyber_attack_detection.objects.all():
        ws.append([getattr(obj, field) for field in headers])

    wb.save(response)
    return response


# ---------------------------------------------
# TRAIN MODEL (FAST after first run)
# ---------------------------------------------
def train_model(request):

    if detection_accuracy.objects.exists():
        return render(
            request,
            "SProvider/train_model.html",
            {"objs": detection_accuracy.objects.all()}
        )

    # heavy imports only inside
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(os.path.join(settings.BASE_DIR, "Datasets.csv"))

    mapping = {"Malware": 0, "DDoS": 1, "Intrusion": 2}
    df["results"] = df["Attack_Type"].apply(lambda x: mapping[x])

    cv = CountVectorizer()
    X = cv.fit_transform(df["Payload_Data"])
    y = df["results"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    models = [
        ("Artificial Neural Network (ANN)", MLPClassifier(max_iter=500)),
        ("SVM", SVC()),
        ("Logistic Regression", LogisticRegression(max_iter=500)),
        ("Decision Tree Classifier", DecisionTreeClassifier()),
        ("Gradient Boosting Classifier", GradientBoostingClassifier()),
    ]

    objs = []

    for name, model in models:
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test)) * 100
        objs.append(detection_accuracy(names=name, ratio=acc))

    detection_accuracy.objects.bulk_create(objs)

    return render(
        request,
        "SProvider/train_model.html",
        {"objs": detection_accuracy.objects.all()}
    )
