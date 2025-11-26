from django.contrib import admin
from django.urls import path, re_path
from django.conf import settings
from django.conf.urls.static import static

from Remote_User import views as remoteuser
from Service_Provider import views as serviceprovider

urlpatterns = [
    # Remote User
    path('admin/', admin.site.urls),
    path('', remoteuser.index, name="index"),
    path('login/', remoteuser.login, name="login"),
    path('Register1/', remoteuser.Register1, name="Register1"),
    path('Predict_Cyber_Attack_Type/', remoteuser.Predict_Cyber_Attack_Type, name="Predict_Cyber_Attack_Type"),
    path('ViewYourProfile/', remoteuser.ViewYourProfile, name="ViewYourProfile"),

    # Service Provider
    path('serviceproviderlogin/', serviceprovider.serviceproviderlogin, name="serviceproviderlogin"),
    path('View_Remote_Users/', serviceprovider.View_Remote_Users, name="View_Remote_Users"),
    path('View_Prediction_Of_Cyber_Attack_Type/', serviceprovider.View_Prediction_Of_Cyber_Attack_Type, name="View_Prediction_Of_Cyber_Attack_Type"),
    path('View_Prediction_Of_Cyber_Attack_Type_Ratio/', serviceprovider.View_Prediction_Of_Cyber_Attack_Type_Ratio, name="View_Prediction_Of_Cyber_Attack_Type_Ratio"),

    path('charts/<str:chart_type>/', serviceprovider.charts, name="charts"),
    path('charts1/<str:chart_type>/', serviceprovider.charts1, name="charts1"),
    path('likeschart/<str:like_chart>/', serviceprovider.likeschart, name="likeschart"),

    path('train_model/', serviceprovider.train_model, name="train_model"),
    path('Download_Predicted_DataSets/', serviceprovider.Download_Predicted_DataSets, name="Download_Predicted_DataSets"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
