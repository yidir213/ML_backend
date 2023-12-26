import pandas as pd
import numpy as np
from io import TextIOWrapper
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from django.core.cache import cache
from .serializers import csvSerializer,predSerializer


def scale_dataset(dataframe, oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1, 1))))

  return data, X, y


model=None

class TrainAPIView(APIView):
    model = None
    def post(self, request):
        if request.path == '/train/':
            return self.post_train( request)
        elif request.path == '/train/predict/':
            return self.post_pred( request)
        
    def post_train(self, request):
       serializer = csvSerializer(data=request.data)
       if serializer.is_valid():
            csv_file = serializer.validated_data['csv_file']

            df = pd.read_csv(csv_file)
            train, test = np.split(df.sample(frac=1), [int(0.7*len(df))])
            train, x_train, y_train = scale_dataset(train, oversample=True)
            test, x_test, y_test = scale_dataset(test, oversample=False)
            svm_model = SVC(kernel='linear')
            svm_model = svm_model.fit(x_train, y_train)

            cache.set('model', svm_model)
            #model=svm_model
            
            y_pred = svm_model.predict(x_test)
            accuracy1=classification_report(y_test, y_pred)
            accuracy=accuracy_score(y_test, y_pred)
            acc=f"{accuracy}\n\n {accuracy1}"
            return Response({'accuracy':acc }, status=status.HTTP_200_OK)
        
       return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def post_pred(self,request):
        if cache.get('model') is None:
            return Response({'error': 'Model not trained yet'}, status=status.HTTP_400_BAD_REQUEST)
        else:
            serializer=predSerializer(data=request.data)
            if serializer.is_valid():
                toPred=serializer.validated_data['toPred']
                predicte=cache.get('model').predict(toPred)
                return Response({'predicted':predicte }, status=status.HTTP_200_OK)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# class PredictAPIView(APIView):
    
#     def post(self,request):
#         if model is None:
#             return Response({'error': 'Model not trained yet'}, status=status.HTTP_400_BAD_REQUEST)
#         else:
#             serializer=predSerializer(data=request.data)
#             if serializer.is_valid():
#                 toPred=serializer.validated_data['toPred']
#                 predicte=TrainAPIView.model.predict(toPred)
#                 return Response({'predicted':predicte }, status=status.HTTP_200_OK)
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)




# def train_model(request):
#     if request.method == 'POST':
#         csv_file = TextIOWrapper(request.FILES['csv_file'].file, encoding='utf-8')
#         df = pd.read_csv(csv_file)
#         train, test = np.split(df.sample(frac=1), [int(0.7*len(df))])
#         train, x_train, y_train = scale_dataset(train, oversample=True)
#         test, x_test, y_test = scale_dataset(test, oversample=False)
#         svm_model = SVC(kernel='linear')
#         svm_model = svm_model.fit(x_train, y_train)
#         y_pred = svm_model.predict(x_test)
#         accuracy=classification_report(y_test, y_pred)
#         return render(request, 'result.html', {'accuracy': accuracy})
        




