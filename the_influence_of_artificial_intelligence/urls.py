"""the_influence_of_artificial_intelligence URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import re_path
from django.contrib import admin
from Remote_User import views as remoteuser
from the_influence_of_artificial_intelligence import settings
from Service_Provider import views as serviceprovider
from django.conf.urls.static import static


urlpatterns = [
    re_path('admin/', admin.site.urls),
    re_path(r'^$', remoteuser.index, name="index"),
    re_path(r'^login/$', remoteuser.login, name="login"),
    re_path(r'^Register1/$', remoteuser.Register1, name="Register1"),
    re_path(r'^Predict_Cyber_Attack_Type/$', remoteuser.Predict_Cyber_Attack_Type, name="Predict_Cyber_Attack_Type"),
    re_path(r'^ViewYourProfile/$', remoteuser.ViewYourProfile, name="ViewYourProfile"),
    re_path(r'^serviceproviderlogin/$',serviceprovider.serviceproviderlogin, name="serviceproviderlogin"),
    re_path(r'View_Remote_Users/$',serviceprovider.View_Remote_Users,name="View_Remote_Users"),
    re_path(r'^charts/(?P<chart_type>\w+)', serviceprovider.charts,name="charts"),
    re_path(r'^charts1/(?P<chart_type>\w+)', serviceprovider.charts1, name="charts1"),
    re_path(r'^likeschart/(?P<like_chart>\w+)', serviceprovider.likeschart, name="likeschart"),
    re_path(r'^View_Prediction_Of_Cyber_Attack_Type_Ratio/$', serviceprovider.View_Prediction_Of_Cyber_Attack_Type_Ratio, name="View_Prediction_Of_Cyber_Attack_Type_Ratio"),
    re_path(r'^train_model/$', serviceprovider.train_model, name="train_model"),
    re_path(r'^View_Prediction_Of_Cyber_Attack_Type/$', serviceprovider.View_Prediction_Of_Cyber_Attack_Type, name="View_Prediction_Of_Cyber_Attack_Type"),
    re_path(r'^Download_Predicted_DataSets/$', serviceprovider.Download_Predicted_DataSets, name="Download_Predicted_DataSets"),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
