# %%
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

# %%
#Importing the dataset
data = pd.read_csv("Advertising_data.csv")
data.head()

# %% [markdown]
# # Simple Linear Regression

# %% [markdown]
# Estimate sales with respect to the ads on TV. 

# %%
#Initializing the variables
X = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)

# %%
#Ploting a graph to see the points
plt.figure(figsize=(16, 8))
plt.scatter(X, y, c='red')
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
# plt.show()

# %%
#Splitting our dataset to Training and Testing dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
#Fitting Linear Regression to the training set

reg = LinearRegression()
reg.fit(X_train, y_train)

# %%
#predicting the Test set result
y_pred = reg.predict(X_test)
plt.figure(figsize=(16, 8))
plt.scatter(X, y, c='red')
plt.plot(
    X_test,
    y_pred,
    c='green',
    linewidth=2
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
# plt.show()

# %%
#Calculating the Coefficients
reg.coef_

# %%
#Calculating the Intercept
reg.intercept_

# %%
#Calculating the R squared value

r2_score(y_test, y_pred)

# %%
output = reg.predict([[230.1]])
print(output)

# %% [markdown]
# # Multiple Linear Regression

# %%
#Initializing the variables
X = data.drop(['sales'], axis=1)
y = data['sales'].values.reshape(-1,1)

# %%
#Splitting our dataset to Training and Testing dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
#Fitting Linear Regression to the training set

multiple_reg = LinearRegression()
multiple_reg.fit(X_train, y_train)

# %%
#predicting the Test set result
y_pred = multiple_reg.predict(X_test)

# %%
#Calculating the Coefficients
multiple_reg.coef_

# %%
#Calculating the Intercept
multiple_reg.intercept_

# %%
#Calculating the R squared value
r2_score(y_test, y_pred)

# %%
#Taking the input from the user
print("Enter the ammount you will invest on:")
tv = float(input("TV : "))
radio = float(input("Radio : "))
newspaper = float(input("Newspaper : "))

#predicting the sales with respect to the inputs
output = multiple_reg.predict([[tv,radio,newspaper]])
print("you will get Rs{:.2f} sales by advertising Rs{} on TV, Rs{} on Radio and Rs{} on newspaper."\
      .format(output[0][0] if output else "not predictable",tv,radio,newspaper))

# %%
#Saving the model
if not os.path.exists('models'):
    os.makedirs('models')
    
MODEL_PATH = "models/multiple_reg.sav"
pickle.dump(multiple_reg, open(MODEL_PATH, 'wb'))

# %%



