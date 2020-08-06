import pandas as pd 
from os import listdir

sum=0

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


filenames = find_csv_filenames(r"./CSV/car")
for name in filenames:
    data = pd.read_csv(r"./CSV/car/"+name) 
    col=[]
    total_rows=len(data)
    sum=sum+total_rows
    for i in range(total_rows):
        col.append('car')
    data['result'] = col
    data.to_csv(r"./CSV/car/"+name)
    
data2 = pd.DataFrame()

filenames = find_csv_filenames(r"./CSV/car")
for name in filenames:
    data1 = pd.read_csv(r"./CSV/car/"+name) 
    data2=data2.append(data1,sort=False)

filenames = find_csv_filenames(r"./CSV/book")
for name in filenames:
    data = pd.read_csv(r"./CSV/book/"+name) 
    col=[]
    sum=sum+total_rows
    total_rows=len(data)
    for i in range(total_rows):
        col.append('book')
    data['result'] = col
    data.to_csv(r"./CSV/book/"+name)
    
#data3 = pd.DataFrame()

filenames = find_csv_filenames(r"./CSV/book")
for name in filenames:
    data1 = pd.read_csv(r"./CSV/book/"+name) 
    data2=data2.append(data1,sort=False)

filenames = find_csv_filenames(r"./CSV/movie")
for name in filenames:
    data = pd.read_csv(r"./CSV/movie/"+name) 
    col=[]
    sum=sum+total_rows
    total_rows=len(data)
    for i in range(total_rows):
        col.append('movie')
    data['result'] = col
    data.to_csv(r"./CSV/movie/"+name)
    
#data3 = pd.DataFrame()

filenames = find_csv_filenames(r"./CSV/movie")
for name in filenames:
    data1 = pd.read_csv(r"./CSV/movie/"+name) 
    data2=data2.append(data1,sort=False)

filenames = find_csv_filenames(r"./CSV/gift")
for name in filenames:
    data = pd.read_csv(r"./CSV/gift/"+name) 
    col=[]
    sum=sum+total_rows
    total_rows=len(data)
    for i in range(total_rows):
        col.append('gift')
    data['result'] = col
    data.to_csv(r"./CSV/gift/"+name)
    
#data3 = pd.DataFrame()

filenames = find_csv_filenames(r"./CSV/gift")
for name in filenames:
    data1 = pd.read_csv(r"./CSV/gift/"+name) 
    data2=data2.append(data1,sort=False)

filenames = find_csv_filenames(r"./CSV/total")
for name in filenames:
    data = pd.read_csv(r"./CSV/total/"+name) 
    col=[]
    sum=sum+total_rows
    total_rows=len(data)
    for i in range(total_rows):
        col.append('total')
    data['result'] = col
    data.to_csv(r"./CSV/total/"+name)
    
#data3 = pd.DataFrame()

filenames = find_csv_filenames(r"./CSV/total")
for name in filenames:
    data1 = pd.read_csv(r"./CSV/total/"+name) 
    data2=data2.append(data1,sort=False)

filenames = find_csv_filenames(r"./CSV/sell")
for name in filenames:
    data = pd.read_csv(r"./CSV/sell/"+name) 
    col=[]
    sum=sum+total_rows
    total_rows=len(data)
    for i in range(total_rows):
        col.append('sell')
    data['result'] = col
    data.to_csv(r"./CSV/sell/"+name)
    
#data3 = pd.DataFrame()

filenames = find_csv_filenames(r"./CSV/sell")
for name in filenames:
    data1 = pd.read_csv(r"./CSV/sell/"+name) 
    data2=data2.append(data1,sort=False)

from sklearn.utils import shuffle
data2 = shuffle(data2)
data2 = data2.sample(frac=1).reset_index(drop=True)
import sklearn.utils
data2 = sklearn.utils.shuffle(data2)
data2 = data2.reset_index(drop=True)

X = data2
y = pd.DataFrame(data=data2, columns=['result'])

X.drop(X.iloc[:, 0:2], inplace=True, axis=1)
X.drop(X.iloc[:, 1:16], inplace=True, axis=1) #removing nose_score to rightEar_y
X.drop(X.iloc[:, 25:37], inplace=True, axis=1) #removing leftKnee_score to rightAnkle_y

del X['result']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

import xgboost as xgb

dtrain = xgb.DMatrix(data=X, label=y)
dtest = xgb.DMatrix(data=X_test)

params = {
    'max_depth': 6,
    'objective': 'multi:softmax',  # error evaluation for multiclass training
    'num_class': 6,
    'n_gpus': 0
}

bst = xgb.train(params, dtrain)

pred = bst.predict(dtest)

from sklearn.metrics import classification_report

print(classification_report(y_test, pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)

from sklearn.metrics import accuracy_score

predictions = [round(value) for value in pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

data3 = pd.DataFrame()
data2 = pd.DataFrame()

data3 = pd.read_csv(r"./MOVIE_PRACTICE_3_Zhou.csv") 

from sklearn.utils import shuffle
data3 = shuffle(data3)
data3 = data3.sample(frac=1).reset_index(drop=True)
import sklearn.utils
data3 = sklearn.utils.shuffle(data3)
data3 = data3.reset_index(drop=True)

X3=data3
y1 = pd.DataFrame(data=data3, columns=['result'])

X3.drop(X3.iloc[:, 0:1], inplace=True, axis=1)
X3.drop(X3.iloc[:, 1:16], inplace=True, axis=1) #removing nose_score to rightEar_y
X3.drop(X3.iloc[:, 25:37], inplace=True, axis=1) #removing leftKnee_score to rightAnkle_y

dtest = xgb.DMatrix(data=X3)
pred = bst.predict(dtest)

a=[]
for i in range(len(pred)):
    a.append(4)

from sklearn.metrics import accuracy_score

predictions = [round(value) for value in pred]
# evaluate predictions
accuracy = accuracy_score(a, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

car=0
book=0
movie=0
gift=0
total=0
sell=0
for i in range(len(pred)):    
    if(pred[i]==0):
        sell=sell+1
    elif(pred[i]==1):
        book=book+1
    elif(pred[i]==2):
        total=total+1
    elif(pred[i]==3):
        gift=gift+1
    elif(pred[i]==4):
        movie=movie+1
    elif(pred[i]==5):
        car=car+1
car1=(car/len(pred))*100
book1=(book/len(pred))*100
movie1=(movie/len(pred))*100
gift1=(gift/len(pred))*100
total1=(total/len(pred))*100
sell1=(sell/len(pred))*100
print(car1)
print(book1)
print(movie1)
print(gift1)
print(total1)
print(sell1)
max1=[]
max1.append(car1)
max1.append(book1)
max1.append(movie1)
max1.append(gift1)
max1.append(total1)
max1.append(sell1)
s=max1.index(max(max1))

if(s==0):
    print("car")
elif(s==1):
    print("book")
elif(s==2):
    print("movie")
elif(s==3):
    print("gift")
elif(s==4):
    print("total")
elif(s==5):
    print("sell")

import pickle
filename = 'xg_model.p'
pickle.dump(bst, open(filename, 'wb'))

  
