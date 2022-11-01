#linear regression - superwised learning technique(data is well labelled).
#data set - area vs prices.
#data set sources - kaggle,github.
#dataset - https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/areavsprices.csv.
#dataset = raw data , dataframe = tabled data.

#CREATE A MACHINE LEARNING MODEL WHICH COULD PREDICT THE PRICES USING THE DATASET GIVEN.

#1. TAKE THE DATA AND CREATE DATA FRAME.

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/areavsprices.csv')
df 

# AS OUR DATASET IS TOO SMALL WE CAN SKIP THE 2ND STEP.

#3. DATA VISUALISATIN AND CREATION OF GRAPHS.

plt.plot(df['Area'],df['Prices'])

#4. DIVIDE DATA INTO INPUT AND OUTPUT

#INPUT = AREA -INPUT IS ALWAYS TWO DIMENSIONAL ARRAY
#OUTPUT = PRICE -OUTPUT IS ALWAYS ONE DIMENSIONAL ARRAY 
#if there is a colon(:) in the columns space inside iloc[] then it is a 2d array,to make it 1d array just avoid the colon as df.iloc[0:6,1]

x=df.iloc[0:6,0:1].values                                   #.value converts data into array.an alternative to x is x=df.iloc[:,0:1].values 
print(x)
y=df.iloc[0:6,1].values
print(y)

#AGAIN AS OUR DATASET IS TOO SMALL WE CAN SKIP STEP 5 AND 6

#7.RUN A CLASSIFIER REGRESSOR OR CLUSTERER

from sklearn.linear_model import LinearRegression
model=LinearRegression()

#8. FIT THE MODEL

model.fit(x,y)            # fit mappes all the inputs and outputs(plotting all tha values of inputs and outputs).
                          # in the imaginary graph of linear regression we are just plotting all the values of inputs(x) and outputs(y).
  
  #9. PREDICT THE OUTPUT

y_pred=model.predict(x)   #using the input values,we predict the output
print(y_pred)             #got the predicted outputs

#now we compare the predicted output with our actual output

y                         #y contains the actual output values

#when we compare the y_pred and y values we reach to a conclusion that there is a huge difference between the actual and predicted values.
#it does not mean that the model is inaccurate,it only means that our model is non linear.
#JUST TO CHECK IF OUR MODEL HAS PREDICTED PROPERLY OR NOT , WE HAVE A CROSS VERIFICATION TECHNIQUE
#y = mx + C - EQUATION OF A STRAIGHT LINE
#m - slope
#C - Y-intercept
#x - independant variable
#y - dependent variable

m = model.coef_ # slope(m)
m

C = model.intercept_  # y-intercept(C)
C

#checking

m*1400+c                     # we are just doing individual predictions for 1400 sqft

#INDIVIDUAL PREDICTION(actual)
model.predict([[1400]])     #since our input is in two dimensional we've to put the value inside[[]]

#since both values are same our model is correct
#To check wheather the model is linear or not
#VISUALISATION-BEST FIT LINE
plt.scatter(x,y)
plt.plot(x,y_pred,color='pink')
