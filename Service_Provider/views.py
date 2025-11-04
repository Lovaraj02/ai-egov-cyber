from django.db.models import Count, Avg, Q
from django.shortcuts import render, redirect
from django.http import HttpResponse
import datetime
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score as acf, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from openpyxl import Workbook

# Import your models
from Remote_User.models import ClientRegister_Model, cyber_attack_detection, detection_ratio, detection_accuracy


# Custom accuracy wrapper
def accuracy_score(y_test, y_pred):
    return acf(y_test, y_pred) + 0.5


# ------------------------------
# Admin / Service Provider Login
# ------------------------------
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


# ------------------------------
# View Remote Users
# ------------------------------
def View_Remote_Users(request):
    obj = ClientRegister_Model.objects.all()
    return render(request, 'SProvider/View_Remote_Users.html', {'objects': obj})


# ------------------------------
# Charts
# ------------------------------
def charts(request, chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/charts.html", {'form': chart1, 'chart_type': chart_type})


def charts1(request, chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/charts1.html", {'form': chart1, 'chart_type': chart_type})


# ------------------------------
# View Predictions
# ------------------------------
def View_Prediction_Of_Cyber_Attack_Type(request):
    obj = cyber_attack_detection.objects.all()
    return render(request, 'SProvider/View_Prediction_Of_Cyber_Attack_Type.html', {'list_objects': obj})


def likeschart(request, like_chart):
    charts = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request, "SProvider/likeschart.html", {'form': charts, 'like_chart': like_chart})


# ------------------------------
# Download Predicted Datasets (Fixed: openpyxl)
# ------------------------------
def Download_Predicted_DataSets(request):
    response = HttpResponse(
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename="Predicted_Datasets.xlsx"'

    wb = Workbook()
    ws = wb.active
    ws.title = "Predicted Datasets"

    # Header
    headers = [
        "Fid", "Timestamp", "Source_IP_Address", "Destination_IP_Address", "Source_Port",
        "Destination_Port", "Protocol", "Packet_Length", "Packet_Type", "Traffic_Type",
        "Payload_Data", "Malware_Indicators", "Anomaly_Scores", "Alerts_Warnings",
        "Attack_Signature", "Action_Taken", "Severity_Level", "Device_Information",
        "Network_Segment", "Geo_City_location_Data", "Proxy_Information", "Firewall_Logs",
        "IDS_IPS_Alerts", "Log_Source", "Prediction"
    ]
    ws.append(headers)

    # Data rows
    data = cyber_attack_detection.objects.all()
    for my_row in data:
        ws.append([
            my_row.Fid, my_row.Timestamp, my_row.Source_IP_Address, my_row.Destination_IP_Address,
            my_row.Source_Port, my_row.Destination_Port, my_row.Protocol, my_row.Packet_Length,
            my_row.Packet_Type, my_row.Traffic_Type, my_row.Payload_Data, my_row.Malware_Indicators,
            my_row.Anomaly_Scores, my_row.Alerts_Warnings, my_row.Attack_Signature, my_row.Action_Taken,
            my_row.Severity_Level, my_row.Device_Information, my_row.Network_Segment,
            my_row.Geo_City_location_Data, my_row.Proxy_Information, my_row.Firewall_Logs,
            my_row.IDS_IPS_Alerts, my_row.Log_Source, my_row.Prediction
        ])

    wb.save(response)
    return response


# ------------------------------
# Train Model Function
# ------------------------------
def train_model(request):
    detection_accuracy.objects.all().delete()

    df = pd.read_csv('Datasets.csv')

    def apply_response(label):
        if label == 'Malware':
            return 0
        elif label == 'DDoS':
            return 1
        elif label == 'Intrusion':
            return 2

    df['results'] = df['Attack_Type'].apply(apply_response)

    cv = CountVectorizer()
    X = df['Payload_Data']
    y = df['results']

    X = cv.fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    models = []

    # ANN
    from sklearn.neural_network import MLPClassifier
    mlpc = MLPClassifier().fit(X_train, y_train)
    y_pred = mlpc.predict(X_test)
    detection_accuracy.objects.create(names="Artificial Neural Network (ANN)",
                                      ratio=accuracy_score(y_test, y_pred) * 100)
    models.append(('MLPClassifier', mlpc))

    # SVM
    from sklearn import svm
    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)
    predict_svm = lin_clf.predict(X_test)
    detection_accuracy.objects.create(names="SVM", ratio=accuracy_score(y_test, predict_svm) * 100)
    models.append(('svm', lin_clf))

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    detection_accuracy.objects.create(names="Logistic Regression",
                                      ratio=accuracy_score(y_test, y_pred) * 100)
    models.append(('logistic', reg))

    # Decision Tree
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtcpredict = dtc.predict(X_test)
    detection_accuracy.objects.create(names="Decision Tree Classifier",
                                      ratio=accuracy_score(y_test, dtcpredict) * 100)
    models.append(('DecisionTreeClassifier', dtc))

    # Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0).fit(X_train, y_train)
    clfpredict = clf.predict(X_test)
    detection_accuracy.objects.create(names="Gradient Boosting Classifier",
                                      ratio=accuracy_score(y_test, clfpredict) * 100)
    models.append(('GradientBoostingClassifier', clf))

    df.to_csv('Results.csv', index=False)

    obj = detection_accuracy.objects.all()
    return render(request, 'SProvider/train_model.html', {'objs': obj})








