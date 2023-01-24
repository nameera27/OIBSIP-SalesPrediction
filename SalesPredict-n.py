import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
df=pd.read_csv("Advert.csv")
df=df.drop(columns='Unnamed: 0',axis=1)
print(df)
print(df.head)
print(df.shape)
print(df.describe())

sns.heatmap(df.corr(),cmap="GnBu",annot=True)
plt.show()


X=df.iloc[:,:-1]
print(X)
Y=df.iloc[:,-1]
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7,test_size=0.3,random_state=100)
reg=LinearRegression()
reg.fit(X_train,Y_train)
y_pred = reg.predict(X_test)
print(reg.intercept_)
print(reg.coef_)
coeff_df = pd.DataFrame(reg.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)
predictions =reg.predict(X_test)
print(predictions)
print(r2_score(Y_test,predictions))