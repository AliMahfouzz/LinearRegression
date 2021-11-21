#######################################################################################
#########################using least square method#####################################
#######################################################################################
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#set the width and the height of the figure "graph"
plt.rcParams['figure.figsize'] = (20.0, 10.0)

#reading data from csv
data = pd.read_csv('headbrain.csv')

#get x and y axis
X = data["Head Size(cm^3)"].values
Y = data["Brain Weight(grams)"].values

#calculate mean of x and y data using numpy
#mean means  if we have [[10,1],[20,10]] then mean of x is 15 and the mean of y is 11/2 = 5.5
#mean of x = sum(x)/count(x)
mean_x = np.mean(X)
mean_y = np.mean(Y)

#get the length of x
m = len(X)

#using the formula to calculate b1 and b0
numer = 0
denom = 0

#calculate the (x-mean_x) * (y-mean_y) for all values in csv and (x-mean_x)^2
#then we can conclude b1 and b0 those are the coefficient for the axe fitting the data

for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

#print coefficients
print(b1, b0)

#plotting values and regression line

max_x = np.max(X) + 100
min_x = np.min(X) - 100

#calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x

#plotting line
plt.plot(x, y, color='#58b970', label='Regression Line')
#plotting scatter points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

#set the axis x and y then plot the graphs
plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()

#calculate r square values based on the prediction and mean_y for all points
ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2 #(y-mean_y)^2
    ss_r += (Y[i] - y_pred) ** 2 #(y-predicted_y)^2
r2 = 1 - (ss_r/ss_t)

print(r2)


##############################################################################################
#################################implementation using scikitlearn#############################
###########################################################################################3

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# cannot use Rank 1 matrix in scikit learn
X = X.reshape((m , 1))
#creating model
reg = LinearRegression()
#fitting training data
reg = reg.fit(X, Y)
#Y prediction
Y_pred = reg.predict(X)
#calculating r2 score
r2_score = reg.score(X, Y)

print(r2_score)