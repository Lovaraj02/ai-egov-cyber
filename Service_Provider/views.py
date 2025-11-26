from django.db.models import Count, Avg, Q
from django.shortcuts import render, redirect
from django.http import HttpResponse
import datetime
import pandas as pd
from sklearn.metrics import accuracy_score as acf
from sklearn.tree import DecisionTreeClassifier
from openpyxl import Workbook
import os
from django.conf import settings

# Import your models
from Remote_User.models import ClientRegister_Model, cyber_attack_detection, detection_ratio, detection_accuracy


# ------------------------------------------
# Accuracy wrapper (same as original)
# ------------------------------------------
def accuracy_score(y_test, y_pred):
    return acf(y_test, y_pred) + 0.5


# ------------------------------------------
# Admin / Service Provider Login
# ------------------------------------------
def serviceproviderlogin(request):
    if request.method == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password == "Admin":
            detection_accuracy.objects.all().delete()
            return redirect('View_Remote_Users')
    return render(request, 'SProvider/serviceproviderlogin.html')


# ------------------------------------------
# View Cyber Attack Type Ratio (Prediction)
# ------------------------------------------
def View_Prediction_Of_Cyber_Attack_Type_Ratio(request):
    detection_ratio.objects.all().delete()

    def calc_ratio(keyword):
        obj = cyber_attack_detection.objects.filter(Q(Prediction=keyword))
        total = cyber_attack_detection.objects.count()
        if total > 0:
            ratio = (obj.count() / total) * 100
            if ratio != 0:
                detection_ratio.objects.create(names=keyword, ratio=ratio)

    for keyword in ['DDoS', 'Intrusion', 'Malware']:
        calc_ratio(keyword)

    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Prediction_Of_Cyber_Attack_Type_Ratio.html', {'objs': obj})


# ------------------------------------------
# View Remote Users
# ------------------------------------------
def View_Remote_Users(request):
    obj = ClientRegister_Model.objects.all()
    return render(request, 'SProvider/View_Remote_Users.html', {'objects': obj})


# ------------------------------------------
# Charts
# ------------------------------------------
def charts(request, chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/charts.html", {'form': chart1, 'chart_type': chart_type})


def charts1(request, chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/charts1.html", {'form': chart1, 'chart_type': chart_type})


def likeschart(request, like_chart):
    charts = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/likeschart.html", {'form': charts, 'like_chart': like_chart})


# ------------------------------------------
# View Predictions
# ------------------------------------------
def View_Prediction_Of_Cyber_Attack_Type(request):
    obj = cyber_attack_detection.objects.all()
    return render(request, 'SProvider/View_Prediction_Of_Cyber_Attack_Type.html', {'list_objects': obj})


# ------------------------------------------
# Download Predicted Data
# ------------------------------------------
def Download_Predicted_DataSets(request):
    response = HttpResponse(
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename="Predicted_Datasets.xlsx"'

    wb = Workbook()
    ws = wb.active
    ws.title = "Predicted Datasets"

    headers = [
        "Fid", "Timestamp", "Source_IP_Address", "Destination_IP_Address", "Source_Port",
        "Destination_Port", "Protocol", "Packet_Length", "Packet_Type", "Traffic_Type",
        "Payload_Data", "Malware_Indicators", "Anomaly_Scores", "Alerts_Warnings",
        "Attack_Signature", "Action_Taken", "Severity_Level", "Device_Information",
        "Network_Segment", "Geo_City_location_Data", "Proxy_Information", "Firewall_Logs",
        "IDS_IPS_Alerts", "Log_Source", "Prediction"
    ]
    ws.append(headers)

    data = cyber_attack_detection.objects.all()
    for r in data:
        ws.append([
            r.Fid, r.Timestamp, r.Source_IP_Address, r.Destination_IP_Address,
            r.Source_Port, r.Destination_Port, r.Protocol, r.Packet_Length,
            r.Packet_Type, r.Traffic_Type, r.Payload_Data, r.Malware_Indicators,
            r.Anomaly_Scores, r.Alerts_Warnings, r.Attack_Signature, r.Action_Taken,
            r.Severity_Level, r.Device_Information, r.Network_Segment,
            r.Geo_City_location_Data, r.Proxy_Information, r.Firewall_Logs,
            r.IDS_IPS_Alerts, r.Log_Source, r.Prediction
        ])

    wb.save(response)
    return response


# -----------------------------------------------------
# OPTIMIZED TRAIN MODEL (trains ONCE, loads instantly)
# -----------------------------------------------------
def train_model(request):

    # If already trained → Return immediately (FAST)
    if detection_accuracy.objects.exists():
        objs = detection_accuracy.objects.all()
        return render(request, "SProvider/train_model.html", {"objs": objs})

    # First-time training (slow: ~5–10 secs)
    df = pd.read_csv(os.path.join(settings.BASE_DIR, "Datasets.csv"))

    def apply_response(label):
        if label == 'Malware':
            return 0
        elif label == 'DDoS':
            return 1
        elif label == 'Intrusion':
            return 2

    df['results'] = df['Attack_Type'].apply(apply_response)

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score as sk_acc

    cv = CountVectorizer()
    X = cv.fit_transform(df['Payload_Data'])
    y = df['results']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    model_list = [
        ("Artificial Neural Network (ANN)", MLPClassifier(max_iter=500)),
        ("SVM", SVC()),
        ("Logistic Regression", LogisticRegression(max_iter=500)),
        ("Decision Tree Classifier", DecisionTreeClassifier()),
        ("Gradient Boosting Classifier", GradientBoostingClassifier()),
    ]

    results_to_insert = []

    for model_name, model in model_list:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = sk_acc(y_test, y_pred) * 100
        results_to_insert.append(detection_accuracy(names=model_name, ratio=acc))

    detection_accuracy.objects.bulk_create(results_to_insert)

    objs = detection_accuracy.objects.all()
    return render(request, 'SProvider/train_model.html', {'objs': objs})
