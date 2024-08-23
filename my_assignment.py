import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import warnings
warnings.filterwarnings("ignore")
##################################
# matplotlib inline
data_file = "Industrial_data_for regression_analysis.txt"
data = pd.read_csv(data_file,sep='\t',encoding='utf-8')
data.info()
######################################################
data.describe()
######################################################
fig_cols = 6
fig_rows = len(data.columns)
plt.figure(figsize=(4*fig_cols,4*fig_rows))

i=0
for col in data.columns:
    i+=1
    ax=plt.subplot(fig_rows,fig_cols,i)
    sns.distplot(data[col],fit=stats.norm)
    
    i+=1
    ax=plt.subplot(fig_rows,fig_cols,i)
    res = stats.probplot(data[col],plot=plt)
plt.tight_layout()
plt.show()
####################################################
from sklearn.model_selection import train_test_split
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
######################################################
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
clf = LinearRegression()
clf.fit(X_train,y_train)
score = mean_squared_error(y_test,clf.predict(X_test))
print("LinearRegression: ", score)
#####################################################

