import pandas as pd
print("Hello")
data=pd.read_excel('IrmaMudThicknessComparisons.xlsx')
print(data.head())
print(data['Thick1(cm)'])

