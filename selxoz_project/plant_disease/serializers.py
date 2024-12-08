# plant_disease/serializers.py

from rest_framework import serializers

class PredictLeafDiseaseSerializer(serializers.Serializer):
    image = serializers.ImageField()  # Barg tasviri
    algorithm = serializers.ChoiceField(choices=["KNN", "SVM", "CNN", "Bayes"])  # Algoritm tanlovi
