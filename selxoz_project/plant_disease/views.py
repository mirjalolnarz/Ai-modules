from django.shortcuts import render

# Create your views here.

# plant_disease/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictLeafDiseaseSerializer

class PredictLeafDiseaseView(APIView):
    def post(self, request):
        serializer = PredictLeafDiseaseSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']
            algorithm = serializer.validated_data['algorithm']
            # Tanlangan algoritm orqali bashoratni olish jarayoni
            # result = model.predict(image)
            return Response({"message": "Bashorat qilingan natijalar"}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
