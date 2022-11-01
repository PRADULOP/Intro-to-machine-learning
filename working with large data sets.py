#1.SCALLING OR NORMALIZATION-ONLY DONE FOR INPUTS
 #BEFORE NORMALIZATION OR SCALING
 import numpy as np
 import matplotlib.pyplot as plt
 np.random.seed(0)     #random.seed() keeps the generated values constant 
 #using numpy ,access the random package and access the randint library,in the range of 1-50,generate 30 integers
 x1=np.random.randint(1,50,30)
 x1=np.sort(x1)
 x2=np.random.randint(10000,70000,30)
 plt.plot(x1,x2)
  
  #2.AFTER NORMALIZATION
x1min=min(x1) 
x1max=max(x1)
x2min=min(x2)
x2max=max(x2)
x1norm=(x1-x1min)/(x1max-x1min)
x2norm=(x2-x2min)/(x2max-x2min)
plt.plot(x1norm,x2norm)

#HERE THESE 2 GRAPHS ARE NO WHERE RELATED TO SCALING OR NORMALIZATION
#SO JUST WANT TO KNOW HOW NORALIZATION WORKS
  
#*LOGISTIC REGRESSION -CLASSIFICATION TECHNIQUE(SUPERWISED LEARNING)*

#DATASET-https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/Social_Network_Ads.csv

#1.  GATHER THE DATA AND CREATE THE DATAFRAME
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/Social_Network_Ads.csv')
df

#PURCHASED COLUMN-0 AND 1
#0-NOT PURCHASED
#1-PURCHASED
#4.
#INPUT- AGE AND ESTIMATED SALARY
#OUTPUT- PURCHASED
# taking INPUT AND OUTPUT
#INPUT IS ALWAYS 2D
x=df.iloc[:,2:4].values
x

#OUTPUT IS ALWAYS ONE DIMENSIONAL
y=df.iloc[:,4].values
y

#5.TRAIN AND TEST VARIABLES
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
#random_state is very similar to random.seed(0)
#by default -75% of the training data -training variables,rest 25% is used as testing data
# the moment we run the train_split cell 
# the data of the x variable is divided into x_train(75%) and x_test(25%)
# the data of the y variable is divided into y_train(75%) and y_test(25%)
print(x.shape) 
print(x_train.shape)
print(x_test.shape)

print(y.shape) 
print(y_train.shape)
print(y_test.shape)

#6.NORMALIZATION OR SCALING
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#7.APPLY CLASSIFIER/REGRESSOR/CLUSTERER - CLASSIFIER
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#8.fitting the model
model.fit(x_train,y_train)   # we are using the x_train and y_train values to train/fit the model

#9.Predictor variable
y_pred = model.predict(x_test)
y_pred                        # predicted output values

y_test                        #actual output values

#10.to find out the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test) * 100

#INDIVIDUAL PREDICTION
model.predict([[20,25000]])    #I want to know , if a person aged 20 and salary 25000 has purchased the product or not
