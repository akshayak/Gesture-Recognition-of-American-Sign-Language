import flask
import werkzeug
import pdb
from flask import request,render_template

from tensorflow import keras
from sklearn.utils import shuffle
import sklearn.utils        
import json
import pandas as pd
import xgboost as xgb
import pickle


app = flask.Flask(__name__)

@app.route("/sample_service",methods = ["POST","GET"])
def index():
    if request.method == "POST":
        def flatten_json(nested_json, exclude=['']):
            """Flatten json object with nested keys into a single level.
                Args:
                    nested_json: A nested json object.
                    exclude: Keys to exclude from output.
                Returns:
                    The flattened json object if successful, None otherwise.
            """
            out = {}
            def flatten(x, name='',exclude=exclude):
                if type(x) is dict:
                    for a in x:
                        if a not in exclude: flatten(x[a], name + a + '_')
                elif type(x) is list:
                    i = 0
                    for a in x:
                        flatten(a, name + str(i) + '_')
                        i += 1
                else:
                    out[name[:-1]] = x
            flatten(nested_json)
            return out

        def getLabels(model,pred):
            print("Model Name: ",model)
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
            print("Car accuracy: ",car1)
            print("Book accuracy: ",book1)
            print("Movie accuracy:",movie1)
            print("Gift accuracy:",gift1)
            print("Total accuracy:",total1)
            print("Sell accuracy:",sell1)

            max1=[]
            max1.append(car1)
            max1.append(book1)
            max1.append(movie1)
            max1.append(gift1)
            max1.append(total1)
            max1.append(sell1)
            s=max1.index(max(max1))
            result=""
            if(s==0):
                result="car"
            elif(s==1):
                result="book"
            elif(s==2):
                result="movie"
            elif(s==3):
                result="gift"
            elif(s==4):
                result="total"
            elif(s==5):
                result="sell"
            return result

       
        randomForest=pickle.load( open("randomForest.p", "rb" ) )
        decisionTree=pickle.load( open("decisionTree.p", "rb" ) )
        knn=pickle.load( open("knn.p", "rb" ) )
        xgBoost=pickle.load( open("xg_model.p", "rb" ) )


        df1 = request.json
        with open('data1.json', 'w') as outfile:
           json.dump(request.json, outfile)

        df1=pd.read_json('C:/Users/aksha/Documents/Fall2019/Mobile computing (CSE535)/Assignment 2/My Code/data1.json')
        dft2=pd.DataFrame(df1)
        dft2.shape


        dft2=pd.DataFrame([flatten_json(x) for x in df1['keypoints']])

        df_score=pd.DataFrame(data=df1,columns=['score'])
        for i in range(0,17):
            col=str(i)+"_part"
            del dft2[col]
        
        
        dft2.rename(columns = {'0_position_x':'nose_x','0_position_y':'nose_y','0_score':'nose_score',
            '1_position_x':'leftEye_x','1_position_y':'leftEye_y','1_score':'leftEye_score',
            '2_position_x':'rightEye_x','2_position_y':'rightEye_y','2_score':'rightEye_score',
            '3_position_x':'leftEar_x','3_position_y':'leftEar_y','3_score':'leftEar_score',
            '4_position_x':'rightEar_x','4_position_y':'rightEar_y','4_score':'rightEar_score',
            '5_position_x':'leftShoulder_x','5_position_y':'leftShoulder_y','5_score':'leftShoulder_score',
            '6_position_x':'rightShoulder_x','6_position_y':'rightShoulder_y','6_score':'rightShoulder_score',
            '7_position_x':'leftElbow_x','7_position_y':'leftElbow_y','7_score':'leftElbow_score',
            '8_position_x':'rightElbow_x','8_position_y':'rightElbow_y','8_score':'rightElbow_score',
            '9_position_x':'leftWrist_x','9_position_y':'leftWrist_y','9_score':'leftWrist_score',
            '10_position_x':'rightWrist_x','10_position_y':'rightWrist_y','10_score':'rightWrist_score',
            '11_position_x':'leftHip_x','11_position_y':'leftHip_y','11_score':'leftHip_score',
            '12_position_x':'rightHip_x','12_position_y':'rightHip_y','12_score':'rightHip_score',
            '13_position_x':'leftKnee_x','13_position_y':'leftKnee_y','13_score':'leftKnee_score',
            '14_position_x':'rightKnee_x','14_position_y':'rightKnee_y','14_score':'rightKnee_score',
            '15_position_x':'leftAnkle_x','15_position_y':'leftAnkle_y','15_score':'leftAnkle_score',
            '16_position_x':'rightAnkle_x','16_position_y':'rightAnkle_y','16_score':'rightAnkle_score'}, inplace = True) 
        
        dft2=dft2[['nose_score','nose_x','nose_y','leftEye_score','leftEye_x','leftEye_y','rightEye_score','rightEye_x',
        'rightEye_y','leftEar_score','leftEar_x','leftEar_y','rightEar_score','rightEar_x','rightEar_y','leftShoulder_score',
        'leftShoulder_x','leftShoulder_y','rightShoulder_score','rightShoulder_x','rightShoulder_y','leftElbow_score','leftElbow_x',
        'leftElbow_y','rightElbow_score','rightElbow_x','rightElbow_y','leftWrist_score','leftWrist_x','leftWrist_y','rightWrist_score',
        'rightWrist_x','rightWrist_y','leftHip_score','leftHip_x','leftHip_y','rightHip_score','rightHip_x','rightHip_y','leftKnee_score',
        'leftKnee_x','leftKnee_y','rightKnee_score','rightKnee_x','rightKnee_y','leftAnkle_score','leftAnkle_x','leftAnkle_y','rightAnkle_score',
        'rightAnkle_x','rightAnkle_y']]

        dft2.insert (0, "score_overall", df_score)

        dft2.drop(dft2.iloc[:, 1:16], inplace=True, axis=1) #removing nose_score to rightEar_y
        dft2.drop(dft2.iloc[:, 25:37], inplace=True, axis=1) #removing leftKnee_score to rightAnkle_y

        dft2.head()

        dft2 = shuffle(dft2)
        dft2 = dft2.sample(frac=1).reset_index(drop=True)
        dft2 = sklearn.utils.shuffle(dft2)
        dft2 = dft2.reset_index(drop=True)

        X=dft2

        random_prediction=randomForest.predict(X)
        dtree_prediction=decisionTree.predict(X)
        knn_prediction=knn.predict(X)
        dtest = xgb.DMatrix(data=X)
        xgBoost_prediction=xgBoost.predict(dtest)

        random_label=getLabels("Random Forest",random_prediction)
        dtree_label=getLabels("Decision Tree",dtree_prediction)
        knn_label=getLabels("KNN",knn_prediction)
        xgBoost_label=getLabels("XgBoost",xgBoost_prediction)

        reqjson = '{"1": "'+ random_label+'","2": "'+ dtree_label+'","3": "'+ knn_label+'","4": "'+ xgBoost_label+'"}'

        reqjson = json.loads(reqjson)
        return reqjson


    else:
        return "Please send post request"

if(__name__ == "__main__"):
    #app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(host = "localhost", port=4505, debug=True)
