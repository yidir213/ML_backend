from rest_framework import serializers

class csvSerializer(serializers.Serializer):
    csv_file = serializers.FileField()
    model = serializers.CharField()

class predSerializer(serializers.Serializer):
    toPred = serializers.ListField()






# class predSerializer(serializers.Serializer):
#     sepal_length = serializers.FloatField()
#     sepal_width = serializers.FloatField()
#     petal_length = serializers.FloatField()
#     petal_width = serializers.FloatField()