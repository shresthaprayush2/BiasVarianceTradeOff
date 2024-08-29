# #Bias/ Variance provides an appropriate to estimate the performance and also a way to select the model
# #A model that undefits the data and is less flexible is high bias model
# #A model that overfits the data and is more flexible high variance model
# # A high variance model will perform good on training data but fail on testing data
# # A high bias model will perform similarly on testing and training data, not ncessarily better

#A simple case you've build a model that is not performing well. You have to now make a decision on what to do to make your model perform well.
#Here are few ways you can tweak ( currently ignoring other hyperparameters)
#Using a more complex model
#Using a more simple model
#Using larger dataset
#Using smaller dataset

#Lets look at an example

#Importing necessary packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import numpy as np

#Creating a simple function to make the data
def makeData(N):
    rng = np.random.RandomState(seed=123)
    #Creating two dimensional data because sikcit take nfeatures,nsample for X which is 2 dimensional
    X = rng.rand(N,1)**2
    #Creaing y that is related to X by using random arithematic
    #X.ravel() change two dimnesion to single dimens
    y = 10-1.0/(X.ravel()+0.1)
    return X,y

#Calling a function makedata that returns X and Y and setting them in a variable
Xtrain,yTrain = makeData(100)

#Creating a simple linearspaced data and giving it a new axis
Xtest = np.linspace(-0.1,1.1,500)[:,np.newaxis]


#Visualizing
#Visualizing the data we see that the data cannont be simply represented by a straight line or linear equation
#To identify which will be best i.e which degree will be best we will vary the degree
plt.scatter(Xtrain.ravel(),yTrain,color='black')
plt.savefig('ActialDaata.png')
axis = plt.axis()


#Importing necessasry modules
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#To demonstarte the complex and simpler model i am using polynomial regression and applying linear regression
# A simple model is of lower degree in this case 2
# A complex model is of higher degree 3
def myPolynomialRegression(degree=1):
    #Using pipeline to cobine multiple steps.
    #First change the degree of data and then apply linear regression
    model = make_pipeline(PolynomialFeatures(degree),LinearRegression(fit_intercept=True))
    #Returning pip which is a model
    return model

#To find the validation score and training score we use validation curve
from sklearn.model_selection import validation_curve
#Creating a new set of degrees to give a range and identify which degree is better
newDegree = range(0,25)
#Using validation curve to get trainScore and valscore
#Validatocurve take estimator which is ML model of interest
# X data, ydata , param_name : this changes from algorithm to algorithm ,
# param_range = range to switch the value of range of parameter to choose from
trainScore,valScore = validation_curve(estimator=myPolynomialRegression(),X=Xtrain,y=yTrain,
                                       param_name='polynomialfeatures__degree',param_range=newDegree,cv=7)
#Plt.plot is used to plot in graph takes
#X axis data,y axis trainscore
plt.plot(newDegree,np.median(trainScore,1),color='blue',label='training_score')
plt.plot(newDegree,np.median(valScore,1),color='red',label='validation_score')
plt.legend(loc='best')
plt.ylim(0,1.1)
plt.xlim(0,15)##Tweak this to change the graph clearly
plt.xlabel('Degree')
plt.ylabel('Score')
plt.savefig('BiasVar1.png')
plt.show()
# From the resulting graph we can see that after degree 5 there is no change in the performance of the algorithm i.e val score
# Hence we can conclude that optimal is 5

# Lets validate that by plotting over the data
#using 4 degrees as test degrees
degree=[1,3,5,100]

#Plotting original data first
plt.scatter(Xtrain.ravel(),yTrain,color='black')

#Using loop to get single degree
for x in degree:
    #Passing through the model
    model = myPolynomialRegression(x)
    model.fit(Xtrain,yTrain)
    ypred = model.predict(Xtest)
    plt.plot(Xtest.ravel(),ypred,label=f'degree {x}')


plt.xlim(-0.1,1.0)
plt.ylim(-2,12)
plt.legend(loc='best')
plt.savefig('PlottingData.png')
plt.show()

#From here we can see that this result support our findings, the optimal sweet spot is 5


#Before you conclude one more step, will the number of data point affect the result
#For this lets create for 1000 data set. The function makes in easier for use to get data
##### TRraining data size
X2,y2 = makeData(1000)
plt.scatter(X2.ravel(),y2)
plt.savefig('DataView.png')
plt.show()

newDegree = np.arange(0,25)
train_score2, val_score2 = validation_curve(estimator=myPolynomialRegression(),X=X2,y=y2,
                                            param_name='polynomialfeatures__degree',
                                            param_range=newDegree,cv=7)
plt.plot(newDegree,np.median(train_score2,1),color='blue',label='training score 1000')
plt.plot(newDegree,np.median(val_score2,1),color='red',alpha=0.4,label='validation score 1000')
plt.plot(newDegree,np.median(trainScore,1),color='blue',label='training score',linestyle='dashed')
plt.plot(newDegree,np.median(valScore,1),color='red',label='validation score',linestyle='dashed')
plt.legend(loc='lower center')
plt.ylim(0,1.1)
plt.xlim(0,6)
plt.xlabel('degree')
plt.ylabel('score')
plt.savefig('LargerData.png')

#Nope having large amount of data shows no significant improvement.






