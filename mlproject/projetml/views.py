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
from .serializers import csvSerializer


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



class TrainAPIView(APIView):
    
    def post(self, request, format=None):
       serializer = csvSerializer(data=request.data)
       if serializer.is_valid():
            csv_file = serializer.validated_data['csv_file']

           # csv_file = TextIOWrapper(request.FILES['csv_file'].file, encoding='utf-8')
            df = pd.read_csv(csv_file)
            train, test = np.split(df.sample(frac=1), [int(0.7*len(df))])
            train, x_train, y_train = scale_dataset(train, oversample=True)
            test, x_test, y_test = scale_dataset(test, oversample=False)
            svm_model = SVC(kernel='linear')
            svm_model = svm_model.fit(x_train, y_train)
            y_pred = svm_model.predict(x_test)
            accuracy=classification_report(y_test, y_pred)
            return Response({'accuracy': accuracy}, status=status.HTTP_200_OK)
        
       return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
       

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
        




