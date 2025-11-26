from django.shortcuts import render, redirect
from django.contrib import messages

from Remote_User.models import (
    ClientRegister_Model,
    cyber_attack_detection
)


# --------------------------------------------------------------------
# SIMPLE USER PAGES
# --------------------------------------------------------------------
def login(request):
    if request.method == "POST" and 'submit1' in request.POST:
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            user = ClientRegister_Model.objects.get(username=username, password=password)
            request.session["userid"] = user.id
            return redirect('ViewYourProfile')
        except:
            messages.error(request, "Invalid Login")

    return render(request, 'RUser/login.html')


def index(request):
    return render(request, 'RUser/index.html')


def Add_DataSet_Details(request):
    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):
    if request.method == "POST":
        ClientRegister_Model.objects.create(
            username=request.POST.get('username'),
            email=request.POST.get('email'),
            password=request.POST.get('password'),
            phoneno=request.POST.get('phoneno'),
            country=request.POST.get('country'),
            state=request.POST.get('state'),
            city=request.POST.get('city'),
            address=request.POST.get('address'),
            gender=request.POST.get('gender'),
        )
        messages.success(request, 'Registered Successfully!')
        return render(request, 'RUser/Register1.html', {'redirect_to_login': True})

    return render(request, 'RUser/Register1.html')


def ViewYourProfile(request):
    user = ClientRegister_Model.objects.get(id=request.session['userid'])
    return render(request, 'RUser/ViewYourProfile.html', {'object': user})


# --------------------------------------------------------------------
# FULL FIXED PREDICTION VIEW
# --------------------------------------------------------------------
def Predict_Cyber_Attack_Type(request):

    if request.method == "POST":

        # extract all fields from form (but manually)
        Fid = request.POST.get('Fid')
        Timestamp = request.POST.get('Timestamp')
        Source_IP_Address = request.POST.get('Source_IP_Address')
        Destination_IP_Address = request.POST.get('Destination_IP_Address')
        Source_Port = request.POST.get('Source_Port')
        Destination_Port = request.POST.get('Destination_Port')
        Protocol = request.POST.get('Protocol')
        Packet_Length = request.POST.get('Packet_Length')
        Packet_Type = request.POST.get('Packet_Type')
        Traffic_Type = request.POST.get('Traffic_Type')
        Payload_Data = request.POST.get('Payload_Data')
        Malware_Indicators = request.POST.get('Malware_Indicators')
        Anomaly_Scores = request.POST.get('Anomaly_Scores')
        Alerts_Warnings = request.POST.get('Alerts_Warnings')
        Attack_Signature = request.POST.get('Attack_Signature')
        Action_Taken = request.POST.get('Action_Taken')
        Severity_Level = request.POST.get('Severity_Level')
        Device_Information = request.POST.get('Device_Information')
        Network_Segment = request.POST.get('Network_Segment')
        Geo_City_location_Data = request.POST.get('Geo_City_location_Data')
        Proxy_Information = request.POST.get('Proxy_Information')
        Firewall_Logs = request.POST.get('Firewall_Logs')
        IDS_IPS_Alerts = request.POST.get('IDS_IPS_Alerts')
        Log_Source = request.POST.get('Log_Source')

        # ---------------- ML IMPORTS (INSIDE ONLY) ----------------
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import VotingClassifier

        # load dataset
        df = pd.read_csv('Datasets.csv')

        def map_label(label):
            return {"Malware": 0, "DDoS": 1, "Intrusion": 2}.get(label, 0)

        df['results'] = df['Attack_Type'].apply(map_label)

        # vectorize
        cv = CountVectorizer()
        X = cv.fit_transform(df['Fid'])
        y = df['results']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        # build models
        mnb = MultinomialNB()
        svm = LinearSVC()
        lr = LogisticRegression(max_iter=500)

        classifier = VotingClassifier([
            ('nb', mnb),
            ('svm', svm),
            ('lr', lr)
        ])

        classifier.fit(X_train, y_train)

        # prediction
        vector = cv.transform([Fid]).toarray()
        result = classifier.predict(vector)[0]

        val = {0: 'Malware', 1: 'DDoS', 2: 'Intrusion'}.get(result)

        # ---------------- SAVE to DB ----------------
        cyber_attack_detection.objects.create(
            Fid=Fid,
            Timestamp=Timestamp,
            Source_IP_Address=Source_IP_Address,
            Destination_IP_Address=Destination_IP_Address,
            Source_Port=Source_Port,
            Destination_Port=Destination_Port,
            Protocol=Protocol,
            Packet_Length=Packet_Length,
            Packet_Type=Packet_Type,
            Traffic_Type=Traffic_Type,
            Payload_Data=Payload_Data,
            Malware_Indicators=Malware_Indicators,
            Anomaly_Scores=Anomaly_Scores,
            Alerts_Warnings=Alerts_Warnings,
            Attack_Signature=Attack_Signature,
            Action_Taken=Action_Taken,
            Severity_Level=Severity_Level,
            Device_Information=Device_Information,
            Network_Segment=Network_Segment,
            Geo_City_location_Data=Geo_City_location_Data,
            Proxy_Information=Proxy_Information,
            Firewall_Logs=Firewall_Logs,
            IDS_IPS_Alerts=IDS_IPS_Alerts,
            Log_Source=Log_Source,
            Prediction=val
        )

        return render(request, 'RUser/Predict_Cyber_Attack_Type.html', {'objs': val})

    return render(request, 'RUser/Predict_Cyber_Attack_Type.html')
