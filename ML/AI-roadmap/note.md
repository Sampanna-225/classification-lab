# AI / ML 
<!-- Ctrl + Shift + V  for preview mode-->
## All that i learned new 
**BUILDING LIST ON THE FLY :** Using [argument] of a pre existing list we mask or manipulate the elements of the list to build a new one.

    Eg: suppose x is a list and val is the elements we extract in the argument so:

    new_x = [val for val in x if 0.0<= val <= 100.0]

    Here we extract each element ( val for val in x) and take it into the list only if it is in the intervl[0.0  ,100.0]

**BOOLEAN MASKING :**  A feature in Numpy that helps mask an argument directly into the array removing the pain of conditional (if-else) with super fast C - code that apples to arrays. 

    !!!This also flattens the array.
    - You can also boolean mask a dataframe in pandas.

    - Requires a variable assignement cause masking returns a value and doesnt create a new variable.
    - Standard and , not doesnt work in it cause they are the shortcuts used by python else we use & , | , ~ where they are and , or , not respectively
    - Eg : let x be an array then: 
        x = x[(x>20) | (x<10)]
**RESHAPING :** Reshaping helps rearrange the dimensions of the arrays with just a simple rule:
    
    The total number of elements must remains the same so a 12 element array can be changed from (2,6) to (3, 4) or visevera on dimensions but the rows and columns must be multiplied to the total elements initially.
    x.reshape(3, 4)


## Scikit-Learn:

**IMPORT :** We use from sklearn to import models and operations

    Such as :
    from sklearn.linear_model import LinearRegression
    from sklearn.matrics import mean_squared_error , r2_score

    We can also split imported datas between train and test where Scikit learn auto shuffles the data and splits it into training ans testing sets for convinence.

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

Here 0.25 is 25% of data taken as Testing data
42 is the seed of the random so we can reproduce the results if we want

**MODEL :** Models in Scikit learn follow a simple procedure which can be resused to make many models.

    model = LinearRegression()
    model.fit(X , Y)
    model.predict()

Thats it!

**MEAN SQUARED ERROR :** Taking Mean of Square of Difference between the actual output and predicted output
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**$R^2$ Score**: The Coefficient of Determination ($R^2$) measures how well a model's predictions approximate real-world data. It represents the proportion of variance in the dependent variable explained by the model.

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Where: 


$y_i$ (Actual / Ground Truth): The true observed value for observation $i$.

$\hat{y}_i$ (Predicted Value): The value estimated or predicted by your model for observation $i$.

$\bar{y}$ (Mean of Actual Values): The baseline average of all true target values ($\bar{y} = \frac{1}{n}\sum y_i$).

$SS_{\text{res}}$ (Sum of Squared Residuals): The sum of squared errors between the true values and your predictions. It measures variance unexplained by the model.

$SS_{\text{tot}}$ (Total Sum of Squares): The total sum of squared differences between actual values and their mean. It measures the total inherent variance in the data (the error if you simply predicted the mean $\bar{y}$ every time).

**MODELS LEARNED:** These are the models ive learn:

    Linear Regression : predicting the upcomming value using simple regression and weather it will rise or decrease based on previous values.

        from sklearn.linear_modal import LinearRegression

    Logistic Regression : Classifing models in 0s and 1s .

        from sklearn.linear_model import LogisticRegression

**Wrappers :** These work as a wrap around to improve how it is used:

    MultiOuputClassifier : In Logistic Regression it strictly requires a 1D output on .fit so it requires a wrapper to predict multi dimentional array .



    [array([[9.99996698e-01, 3.30233715e-06]]),
    array([[0.71940796, 0.28059204]]),
    array([[0.35132954, 0.64867046]])]

**Prediction Method :**

    .predict() : Rounds up 


    .predict() : Proba gives the exact percentage
    [[Probability of 0, Probability of 1]] is the returned array which shows probability from 0 and 1 to be :


**Confusion Matrixes :** 

 Think of it like a reality test which checks weather your actual test output matched the prediction. Like if the machine was expected to give 1 and it predicted 0 we got a False Negative.

    -It gives us the exact thing the model got wrong instead of some accuracy cause even if a model made 99 right and 1 wrong it is a huge liability on Big data.

    it follows this gride:
    (True/False -> Matched or Not)(Positve/Negative -> What it guessed 1 or 0)

$$\begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}$$

    So if we see [
        [1,0]
        [0,1]
    ]

    We know the data is TN so it rightly guessed 0.

If were given datas: 

y_true = np.array([[0, 1, 0]])

y_pred = np.array([[1, 1, 0]])

    The output may come:
    [[[0 1]
    [0 0]]

    [[0 0]
    [0 1]]

    [[1 0]
    [0 0]]]

*Types of Confusion Matrixes :*

-> *Confusion Matrix (Standard)*:

    import numpy as np
    from sklearn.metrics import confusion_matrix

    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

Here the output is Funny cause it gives :

    [[3 1]
     [1 3]]

Which is just howmany in a general unlike in higher dimentional arrays or multilabels. So,

    3 Truely gussed 0 , 1 Falsely guessed 0
    3 Truely gussed 1 , 1 Falsely guessed 1

-> *Multilabel Confusion Matrix :*

    import numpy as np
    from sklearn.metrics import multilabel_confusion_matrix

    y_true = np.array([[0, 1, 0]])
    y_pred = np.array([[1, 1, 0]])

    print(multilabel_confusion_matrix(y_true, y_pred))

## Prcision Calculating :

"When the model shouts '1', how often is it right?"
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

Adding in values: 

$$\text{Precision} = \frac{3}{3 + 1} = \frac{3}{4} = 0.75 \text{ (or 75\%)}$$
What this means: Out of every time your model guessed 1, it was right 75% of the time (and threw 1 false alarm).

-> For multi label we calculate of different flags and average it out.

## Recall :

"Out of all the real '1's out there, how many did we catch?"

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

Adding in values:

$$\text{Recall} = \frac{3}{3 + 1} = \frac{3}{4} = 0.75 \text{ (or 75\%)}$$

What this means: Out of all the actual positive cases that existed in reality, your model successfully caught 75% of them (and missed 1).


##  F1 Score:

Precision and Recall collide with each other and its hard to grade a model solely based on them cause a model may have high recall (it guessed most of the 1s correct) but in doing soo solely spammed 1s on many possibles 0 hense precision may be low and vice versa. 

So we need a single metric to grade our model balancing both false alrams and missed threats.

The F1-Score is calculated using Precision ($P$) and Recall ($R$):

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### !!! IMPORTANT !!!

Here we see the F1 score using harmonic mean. Why harmonic mean you may ask, why not arithmetic :

Here is the hard truth , the below example shows how each mean acts up:

**Arithmetic Mean**

Its just a simple formulae how bad could it be taking the mean of precision and recall to establish a relation between them so even if one falls and other increases we still get the mean values, right?

$$\text{Average} = \frac{1.0 + 0.0}{2} = \mathbf{0.5 \text{ (50\%)}}$$

Here precision 1.0 is taken which says the model predicted 1 its always right but recall is set 0 meaning the model hit 0 targets. The average is 5.0.

Now there can not be a model that guesses 1 100% accurately but has never guessed 1 correctly this is a model that can mathematically not exist. But we see its average 5.0 which makes no sense.

**Harmonic Mean**

Here we dont only find mean we find it accurately balancing each term with proper relation. So if either of the precision and recall tend to hit 0 the accuracy crashes striaght forward.

$$F_1 = 2 \times \frac{1.0 \times 0.0}{1.0 + 0.0} = \mathbf{0.0 \text{ (0\%)}}$$

But there is a little takeaway. The harmonic mean punishes extreme imbalance. It forces your model to prove that it can handle both false alarms and missed threats simultaneously. If a model sacrifices one completely to look good on the other, the F1-score exposes it instantly.

It reflects hard on whats the actual prediction accuracy of the model.

