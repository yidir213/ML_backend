import pandas as pd
import numpy as np
# from io import TextIOWrapper


from sklearn.preprocessing import StandardScaler #pour la normalisation des données
from imblearn.over_sampling import RandomOverSampler #pour creer des échantillons en plus pour la classe avec le moins de échantillons

#accuracy calculator
from sklearn.metrics import classification_report,accuracy_score

# for spliting the dataset
from sklearn.model_selection import train_test_split

#django
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from django.core.cache import cache
from .serializers import csvSerializer,predSerializer


#models:

def svm_model():
    from sklearn.svm import SVC
    model=SVC(kernel='linear')
    return model

def decision_tree():
  from sklearn.tree import DecisionTreeClassifier
  model=DecisionTreeClassifier()
  return model

def naive_bayes():
  from sklearn.naive_bayes import GaussianNB
  model=GaussianNB()
  return model

def logisticReg():
  from sklearn.linear_model import LogisticRegression
  model=LogisticRegression()
  return model

# Multi-Layer Perceptron (MLP) neural network
# accuracy=0.96 for iris
def mlp():
  from sklearn.neural_network import MLPClassifier
  model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
  return model

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    model=KNeighborsClassifier(n_neighbors=2)
    return model


def ada():
  from sklearn.ensemble import AdaBoostClassifier
  model=AdaBoostClassifier(n_estimators=100, random_state=42)
  return model




def split_scale(dfs):
    df = dfs
    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values
    x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test, y_train, y_test

# def scale_dataset(dataframe, oversample=False):
#   X = dataframe[dataframe.columns[:-1]].values
#   y = dataframe[dataframe.columns[-1]].values

#   #   ##normalisation
#   # scaler = StandardScaler()
#   # X = scaler.fit_transform(X)

#   # if oversample:
#   #   ros = RandomOverSampler()
#   #   X, y = ros.fit_resample(X, y)

#   data = np.hstack((X, np.reshape(y, (-1, 1))))

#   return data, X, y



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
            selected_model = serializer.validated_data['model']
            
            #read the dataset
            df = pd.read_csv(csv_file)

            # train, test = np.split(df.sample(frac=1), [int(0.7*len(df))])

            #split the dataset
            x_train,x_test, y_train,y_test = split_scale(df)
            
            models = {
                'svm': svm_model,
                'knn': knn,
                'mlp':mlp,
                "logistic_reg":logisticReg,
                'naive_bayes':naive_bayes,
                "decision_tree":decision_tree,
                "ada":ada
            }

            #chooosing a model accoreding to user choise
            model = models.get(selected_model)()

            #train the model with the dataset provided by the user
            model = model.fit(x_train, y_train)

            cache.set('model', model)

            #pridct y(target) values using the model
            y_pred = model.predict(x_test)

            # accuracy1=classification_report(y_test, y_pred)
            
            #comparing the predicted values with the actual values and calculating the accuracy
            accuracy=accuracy_score(y_test, y_pred)

            return Response({'accuracy':f"{accuracy:.2f}" }, status=status.HTTP_200_OK)
       
       print(serializer.errors) 
       return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def post_pred(self,request):
        if cache.get('model') is None:
            return Response({'error': "Le Model n'est pas encore entrainé"}, status=status.HTTP_400_BAD_REQUEST)
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
        




