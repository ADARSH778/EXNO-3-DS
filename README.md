# EXNO-3-DS

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:
### STEP 1:
Read the given Data.
### STEP 2:
Clean the Data Set using Data Cleaning Process.
### STEP 3:
Apply Feature Encoding for the feature in the data set.
### STEP 4:
Apply Feature Transformation for the feature in the data set.
### STEP 5:
Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  ### 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  ### 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/704cc4d4-40ec-4e31-a24a-fcd59840aabd)
### Ordinal Encoder
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/3e4006cd-b2f4-4673-94d2-b6335beb2dc2)

```
df
```
![image](https://github.com/user-attachments/assets/43653a20-8e52-4ee4-b71e-d46f9868d931)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/1ec3056f-c6d5-49f8-bf03-dd27a021ae8a)
### Label Encoder 
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/0e9fd189-a180-4031-bb09-feba1295a25e)
### One Hot Encoder
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/201fda7e-2988-48ae-bfd0-1a07915f56b2)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/b8ad26a6-ff35-4c31-ba01-3e0e7cc84a77)
### Binary Encoder
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/1df990f4-fc06-48f7-a901-1b6ec2045cf7)

```
from category_encoders import BinaryEncoder
df=pd.read_csv('data.csv')
df
```
![image](https://github.com/user-attachments/assets/7922c132-61b6-4acd-8a3b-7a4275c87f0a)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/e7a7bcf9-55a4-4256-b514-f59518cd9097)
### Target Encoder
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc['City'],y=cc['Target'])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/06d9748d-62ab-4182-bde0-6953a50dd807)

```
from scipy import stats
import numpy as np
df=pd.read_csv('Data_to_Transform.csv')
df
```
![image](https://github.com/user-attachments/assets/4659ae73-0385-45a1-adcc-1d63be93a7cd)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/dc20057a-3bca-4281-86ad-5bb81de4b20f)

```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/2d5ab3eb-b333-4e62-997d-2ac3ac0cb815)

```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/c434c3f4-4d69-481f-bf5e-dca7d22db751)

```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/696b9dd3-25f3-4713-a27f-4434629c503a)

```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/cfb1c93c-a14b-407d-84bb-d1429247b370)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/7bacd9f9-3d83-479d-bf31-1cd12268bd59)

```
df.skew()
```
![image](https://github.com/user-attachments/assets/0e03083d-9f6e-4493-b5b7-2b96260a0395)

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/3b0788f0-040e-4350-b562-5359ef96d47e)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/32eb19a5-a76b-4a83-8493-553137eeb509)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ce0c63bb-2cb2-4efb-b6f3-1d8005feecc5)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/04474b24-e0c2-4d45-8558-561d7f36b736)

```
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9665d7de-42f8-4c68-99a7-5cc027214a96)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/de4f52b6-133f-45f9-9f21-1b183fd3f385)

```
dt=pd.read_csv("titanic_dataset.csv")
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/7f99d887-01dc-42cb-842c-53efef6e399c)

```
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/892a873e-1b41-4646-ad3f-385832872ea4)

## RESULT:
Thus with the given datasets,Tasks of Feature Encoding,Feature Transformation process and saving the data to a file was performed successfully.

       
