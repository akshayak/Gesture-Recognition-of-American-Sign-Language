# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd 
from os import listdir
from sklearn.utils import shuffle
import sklearn.utils
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
import pickle



def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


data2 = pd.DataFrame()    
#book
filenames = find_csv_filenames(r"./CSV_Thursday/CSV/book")
for name in filenames:
    data = pd.read_csv(r"./CSV_Thursday/CSV/book/"+name) 
    col=[]
    total_rows=len(data)
    for i in range(total_rows):
        col.append('book')
    data['result'] = col
    data.to_csv(r"./CSV_Thursday/CSV/book/"+name)

filenames = find_csv_filenames(r"./CSV_Thursday/CSV/book")
for name in filenames:
    data1 = pd.read_csv(r"./CSV_Thursday/CSV/book/"+name) 
    data2=data2.append(data1)

#car
filenames = find_csv_filenames(r"./CSV_Thursday/CSV/car")
for name in filenames:
    data = pd.read_csv(r"./CSV_Thursday/CSV/car/"+name) 
    col=[]
    total_rows=len(data)
    for i in range(total_rows):
        col.append('car')
    data['result'] = col
    data.to_csv(r"./CSV_Thursday/CSV/car/"+name) 
filenames = find_csv_filenames(r"./CSV_Thursday/CSV/car")
for name in filenames:
    data1 = pd.read_csv(r"./CSV_Thursday/CSV/car/"+name) 
    data2=data2.append(data1)

#movie
filenames = find_csv_filenames(r"./CSV_Thursday/CSV/movie")
for name in filenames:
    data = pd.read_csv(r"./CSV_Thursday/CSV/movie/"+name) 
    col=[]
    total_rows=len(data)
    for i in range(total_rows):
        col.append('movie')
    data['result'] = col
    data.to_csv(r"./CSV_Thursday/CSV/movie/"+name) 
filenames = find_csv_filenames(r"./CSV_Thursday/CSV/movie")
for name in filenames:
    data1 = pd.read_csv(r"./CSV_Thursday/CSV/movie/"+name) 
    data2=data2.append(data1)

#total
filenames = find_csv_filenames(r"./CSV_Thursday/CSV/total")
for name in filenames:
    data = pd.read_csv(r"./CSV_Thursday/CSV/total/"+name) 
    col=[]
    total_rows=len(data)
    for i in range(total_rows):
        col.append('total')
    data['result'] = col
    data.to_csv(r"./CSV_Thursday/CSV/total/"+name) 
filenames = find_csv_filenames(r"./CSV_Thursday/CSV/total")
for name in filenames:
    data1 = pd.read_csv(r"./CSV_Thursday/CSV/total/"+name) 
    data2=data2.append(data1)

#gift
filenames = find_csv_filenames(r"./CSV_Thursday/CSV/gift")
for name in filenames:
    data = pd.read_csv(r"./CSV_Thursday/CSV/gift/"+name) 
    col=[]
    total_rows=len(data)
    for i in range(total_rows):
        col.append('gift')
    data['result'] = col
    data.to_csv(r"./CSV_Thursday/CSV/gift/"+name) 
filenames = find_csv_filenames(r"./CSV_Thursday/CSV/gift")
for name in filenames:
    data1 = pd.read_csv(r"./CSV_Thursday/CSV/gift/"+name) 
    data2=data2.append(data1)

#sell
filenames = find_csv_filenames(r"./CSV_Thursday/CSV/sell")
for name in filenames:
    data = pd.read_csv(r"./CSV_Thursday/CSV/sell/"+name) 
    col=[]
    total_rows=len(data)
    for i in range(total_rows):
        col.append('sell')
    data['result'] = col
    data.to_csv(r"./CSV_Thursday/CSV/sell/"+name) 
filenames = find_csv_filenames(r"./CSV_Thursday/CSV/sell")
for name in filenames:
    data1 = pd.read_csv(r"./CSV_Thursday/CSV/sell/"+name) 
    data2=data2.append(data1)
  

#print(data2.shape)



#shuffle data
data2 = shuffle(data2)
data2 = data2.sample(frac=1).reset_index(drop=True)
data2 = sklearn.utils.shuffle(data2)
data2 = data2.reset_index(drop=True)

X = data2
print (X)
y = pd.DataFrame(data=data2, columns=['result'])
X.drop(X.iloc[:, 0:2], inplace=True, axis=1)
X.drop(X.iloc[:, 1:16], inplace=True, axis=1) #removing nose_score to rightEar_y
X.drop(X.iloc[:, 25:37], inplace=True, axis=1) #removing leftKnee_score to rightAnkle_y
del X['result']

sign = {'sell': 0,'book': 1,'total': 2,'gift': 3,'movie': 4 ,'car': 5} 
y.result = [sign[item] for item in y.result] 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42) 
#print(OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test))
model=KNeighborsClassifier(n_neighbors = 6)
model.fit(X,y)
pred=model.predict(X_test)
print("Acc:",model.score(X_test,y_test))
accuracy = accuracy_score(y_test, pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# # creating a confusion matrix 
cm = confusion_matrix(y_test, pred) 

print("Confusion matrix: ",cm)

data3 = pd.read_csv('C:/Users/aksha/Documents/Fall2019/Mobile computing (CSE535)/Assignment 2/CSV_Thursday1/CSV/movie/MOVIE_PRACTICE_3_PATEL(2).csv') 

data3 = shuffle(data3)
data3 = data3.sample(frac=1).reset_index(drop=True)
data3 = sklearn.utils.shuffle(data3)
data3 = data3.reset_index(drop=True)

X3=data3

#y_testing = pd.DataFrame(data=data3, columns=['result'])
#print("y_test=",y_testing)
#y_testing.result = [sign[item] for item in y_testing.result] 
X3.drop(X.iloc[:, 0:1], inplace=True, axis=1)
X3.drop(X3.iloc[:, 1:16], inplace=True, axis=1) #removing nose_score to rightEar_y
X3.drop(X3.iloc[:, 25:37], inplace=True, axis=1) #removing leftKnee_score to rightAnkle_y
pred1 = model.predict(X3) 
print("prediction:",pred1)
a=[]
for i in range(len(pred1)):
    a.append(4)
predictions=[round(value) for value in pred1]

accuracy=accuracy_score(a,predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

car=0
book=0
movie=0
gift=0
total=0
sell=0
for i in range(len(pred1)):    
    if(pred1[i]==0):
        sell=sell+1
    elif(pred1[i]==1):
        book=book+1
    elif(pred1[i]==2):
        total=total+1
    elif(pred1[i]==3):
        gift=gift+1
    elif(pred1[i]==4):
        movie=movie+1
    elif(pred1[i]==5):
        car=car+1
car1=(car/len(pred1))*100
book1=(book/len(pred1))*100
movie1=(movie/len(pred1))*100
gift1=(gift/len(pred1))*100
total1=(total/len(pred1))*100
sell1=(sell/len(pred1))*100
max1=[]
max1.append(car1)
max1.append(book1)
max1.append(movie1)
max1.append(gift1)
max1.append(total1)
max1.append(sell1)
s=max1.index(max(max1))
print("Classification Report: ")
print(classification_report(a, pred1))
if(s==0):
    print("Predicted Label : car")
elif(s==1):
    print("Predicted Label : book")
elif(s==2):
    print("Predicted Label : movie")
elif(s==3):
    print("Predicted Label : gift")
elif(s==4):
    print("Predicted Label : total")
elif(s==5):
    print("Predicted Label : sell")

pickle.dump( model, open( "knn.p", "wb" ) )




  
