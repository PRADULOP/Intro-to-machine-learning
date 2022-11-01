#EXPLORATORY DATA ANALYSIS(EDA) - PRE MACHINE LEARNING
#Before performing EDA or ML or AI , we always need some data,mainly in form of a dataset
#dataset - a collection of data
#dataframe - using pandas library ,we create tabled data,which is called as Dataframe
#gather data and create dataframe
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/areavsprices.csv')
df

df.info()  # It tells us the information abou our dataframe

#suppose you have a very large dataset,say 500000 rows ,finding out the unique fruit names would have been a challenge
#So here we've python into the rescue
fname = df['fruit_name'].unique()
fname

#I want to know the exact count of each and every fruit
fsize = df.groupby('fruit_name',sort = False).size()
fsize

#VISUALIZATION - GRAPH - matplotlib
import matplotlib.pyplot as plt
#plt.bar(x-axis,y-axis)
plt.bar(fname,fsize,color = ['red','pink','orange','lime'])
#matplotlib does not follow alphabetical order, so add ,sort = False in df.groupby

plt.bar(df['Area'],df['Prices'],color='pink')         #plot a barchart

df.info()                                             #gives tne informations about the data
print(df)                                             
df.head()                                             #returns first 5 row indexes
print(df)
df.head(10)                                           #returns rows upto 10
print(df)
df.tail()                                             #returns the last 5 rows
print(df)
df.tail(10)                                           #returns the last 10 rows
print(df)
#slicing
df[:25]                                               #first 25 rows
print(df)
df[25:]                                               #rows from 25
print(df)
df[25:40]                                             #row 25 to 39
print(df)
df.iloc[25:41,0:2]                                    #df.iloc[rowsize,columnsize]
print(df)
df['Area'].unique()                                   #LIST the unique values of that particluar row
print(df)
df.groupby('Prices',sort='FALSE').size()              #displays count of each unique data
print(df)

df.size                                               #returns total size of the data set

df.shape                                              #returns the number of rows and columns
