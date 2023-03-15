# National-Footballer-Value-predicting
This was a project created as part of a CCAI-312 (Pattern Recognition)

# Problem Definition
By predicting player value, an investor company can select the most potential players who are most likely to succeed.

analyzing footballer acceleration, balance, and other best relative factorsthat lead us to a true prediction of player value in euros. 

# Data Description
players_20.csv is attached with the submission code; it has been scraped from the publicly available website sofifa.com.


# Method
For Modeling we use:

-Simple Linear Regression.

-linear Regression with Data transformation.

-Grid Search & Random Forest regression technique.

-other libraries

numpy, Pandas, matplotlib.pyplot, seaborn

# Experemental Results

![image](https://user-images.githubusercontent.com/97242283/225423706-67529f90-88fa-465a-800e-694d92e1a0a7.png)

# Discution
Firstly, We apply simple Linear Regression on selected features to find out their effect on the target, which is players’ current market values. but we gain disapointed result upon that we decided to 
first, we remove goalkeepers cause they have statics vary from other players which effect the model accuracy.

secondarily, we abstract 42 relative features from 106 instead of 7 as before.

thirdly, we will go through some data transformation functions that add a lot to improve our model.

after that we managed to decreas the previous rmse from 4.8 and to become 4.0 that is a good thing but not good enough.
Third phase for improvment we apply Grid Search with Random Forest Regressor to reach the best results, grid search finds the optimal combination of one or more hyperparameters that gives the most optimal model complying with a bias-variance tradeoff, in other words finding the optimal set of parameters by evaluation of all possible parameter combinations. inaddition to  Random Forest witch has multiple decision trees as a base for learning models. we approach the Random Forest regression technique like any other machine learning technique.
