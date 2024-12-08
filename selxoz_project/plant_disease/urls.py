# plant_disease/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.PredictLeafDiseaseView.as_view(), name='predict-leaf-disease'),
]
