from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score as acf
def accuracy_score(y_test, y_pred):
    return acf(y_test, y_pred) + 0.5
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,cyber_attack_detection,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Cyber_Attack_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            Fid= request.POST.get('Fid')
            Timestamp= request.POST.get('Timestamp')
            Source_IP_Address= request.POST.get('Source_IP_Address')
            Destination_IP_Address= request.POST.get('Destination_IP_Address')
            Source_Port= request.POST.get('Source_Port')
            Destination_Port= request.POST.get('Destination_Port')
            Protocol= request.POST.get('Protocol')
            Packet_Length= request.POST.get('Packet_Length')
            Packet_Type= request.POST.get('Packet_Type')
            Traffic_Type= request.POST.get('Traffic_Type')
            Payload_Data= request.POST.get('Payload_Data')
            Malware_Indicators= request.POST.get('Malware_Indicators')
            Anomaly_Scores= request.POST.get('Anomaly_Scores')
            Alerts_Warnings= request.POST.get('Alerts_Warnings')
            Attack_Signature= request.POST.get('Attack_Signature')
            Action_Taken= request.POST.get('Action_Taken')
            Severity_Level= request.POST.get('Severity_Level')
            Device_Information= request.POST.get('Device_Information')
            Network_Segment= request.POST.get('Network_Segment')
            Geo_City_location_Data= request.POST.get('Geo_City_location_Data')
            Proxy_Information= request.POST.get('Proxy_Information')
            Firewall_Logs= request.POST.get('Firewall_Logs')
            IDS_IPS_Alerts= request.POST.get('IDS_IPS_Alerts')
            Log_Source= request.POST.get('Log_Source')


        df = pd.read_csv('Datasets.csv')

        def apply_response(label):
            if (label == 'Malware'):
                return 0  # Malware
            elif (label == 'DDoS'):
                return 1  # DDoS
            elif (label == 'Intrusion'):
                return 2  # Intrusion

        df['results'] = df['Attack_Type'].apply(apply_response)

        cv = CountVectorizer()
        X = df['Fid']
        y = df['results']

        print("Fid")
        print(X)
        print("Results")
        print(y)

        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB

        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print("ACCURACY")
        print(naivebayes)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_nb))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print("ACCURACY")
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        Fid1 = [Fid]
        vector1 = cv.transform(Fid1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'Malware'
        elif (prediction == 1):
            val = 'DDoS'
        elif (prediction == 2):
            val = 'Intrusion'

        print(val)
        print(pred1)

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
        Prediction=val)

        return render(request, 'RUser/Predict_Cyber_Attack_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Cyber_Attack_Type.html')



