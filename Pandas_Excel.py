import pandas as pd
import numpy as np



# read excel file and assign a dataframe to it
data=pd.read_excel('IrmaMudThicknessComparisons.xlsx')
df = pd.DataFrame(data)
print(df.head())

#drop a column in the dataframe
newdata = df.drop(['Notes'], axis=1)
print(newdata)

#find null values and drop them (don't need to find them, the dropna command works independently
null = newdata.isnull()
newdata1 =newdata.dropna()
print(newdata1.isnull())
print(len(newdata1['Region']))
print(newdata1)

#remove all rows with 0
df2 = pd.DataFrame(newdata1)
newdata2 = df2[df2['Thick1(cm)'] != 0]
newdata2 = df2[df2['Thick2(cm)'] != 0]
print(newdata2)

#put column Notes back in