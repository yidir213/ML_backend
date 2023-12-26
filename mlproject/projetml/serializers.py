from rest_framework import serializers

class csvSerializer(serializers.Serializer):
    csv_file = serializers.FileField()
