from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=20)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=10)
    address= models.CharField(max_length=20)
 




class cyber_attack_detection(models.Model):
    Fid = models.TextField(null=True, blank=True)
    Timestamp = models.TextField(null=True, blank=True)
    Source_IP_Address = models.TextField(null=True, blank=True)
    Destination_IP_Address = models.TextField(null=True, blank=True)
    Source_Port = models.TextField(null=True, blank=True)
    Destination_Port = models.TextField(null=True, blank=True)
    Protocol = models.TextField(null=True, blank=True)
    Packet_Length = models.TextField(null=True, blank=True)
    Packet_Type = models.TextField(null=True, blank=True)
    Traffic_Type = models.TextField(null=True, blank=True)
    Payload_Data = models.TextField(null=True, blank=True)
    Malware_Indicators = models.TextField(null=True, blank=True)
    Anomaly_Scores = models.TextField(null=True, blank=True)
    Alerts_Warnings = models.TextField(null=True, blank=True)
    Attack_Signature = models.TextField(null=True, blank=True)
    Action_Taken = models.TextField(null=True, blank=True)
    Severity_Level = models.TextField(null=True, blank=True)
    Device_Information = models.TextField(null=True, blank=True)
    Network_Segment = models.TextField(null=True, blank=True)
    Geo_City_location_Data = models.TextField(null=True, blank=True)
    Proxy_Information = models.TextField(null=True, blank=True)
    Firewall_Logs = models.TextField(null=True, blank=True)
    IDS_IPS_Alerts = models.TextField(null=True, blank=True)
    Log_Source = models.TextField(null=True, blank=True)
    Prediction = models.TextField(null=True, blank=True)





class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



