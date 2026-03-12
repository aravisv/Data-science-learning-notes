Classification - To judge an observation and put it under some category

In regression, we predict the continuous value. But here we classify into particular group the observation belogs based on its properties

If there are only two groups, then the classification becomes binary classification
Else it is multiclass classification.
Multilabel classification is where an observation could belong to multiple classes.
Imbalanced classification is when the number of observations in each class is very unequal.

Applications of binary classification - email spam or not, credit card transaction fraud or not..
Applications of multiclass classification - sentiment analysis, letters and digits recognition from handwriting…
Applications of multilabel classification - objects in an image, possible diseases from blood test report, news article tagging…
Applications of imbalanced classification - people who default the loan, machine failure prediction

Algorithms are logistic regression, decision trees, KNN neighbours, Random forest, support vector machine, Naive bayes

Logistic regression

linear regression predicts continuous value. but here, we are interested in knowing the probability of an observation belonging to a particular group. hence the output shall be bound between 0 and 1. for that purpose we shall use sigmoid transformation, an S shaped curve. not a straight line.

Odds = P(x) / 1 - P(x)
odd is in the range of [0, inf)
to convert it into real line, we do the log odds, which is log( P(x) / 1 - P(x) ). this is in the range of ( -inf, +inf )

if we equate it to simple linear equation and solve for P(x), we get

p(x) = e^(β0 + β1·x1) / (1 + e^(β0 + β1·x1))

Usually a threshold is set, the 2 sides of threshold belong to 2 different groups

Let’s say the odds of the event are:

odds(x1)=e^(β0+β1·x1)

Now if the x1 is increased by 1, then

odds(x1+1) = e^(β0+β1·(x1+1)) = e^(β0+β1x1) * e^β1

implies that the odds increase by e^β1

if β1 > 0, then the odds increase. else decreases
if β1 = 0 , then the input has no effect on the odds

The coefficients are found out by using gradient descent on the Maximum likelihood estimate method. MLE is the probability of finding that observation in the class.

$$
L(\beta_0, \beta_1) = \prod_{i=1}^{n} p(x_i)^{y_i} \cdot (1 - p(x_i))^{1 - y_i}
$$

We multiply all the probabilities together assuming that the outcomes are independent of each other. Because this product can get extremely small, we usually work with the log-likelihood, which turns the product into a sum and is easier to compute and optimize.

Unlike MSE, here log loss is preferred. The Log Loss equation returns the logarithm of the magnitude of the change, rather than just the distance from data to prediction. 

Squared loss works well for linear regression because the relationship between input and output is linear and produces a convex optimization problem. In logistic regression, the sigmoid transformation makes the relationship non-linear, causing squared loss to produce a non-convex loss surface, so cross-entropy loss is used instead.

Another key feature of log loss is that it heavily penalizes predictions that are both wrong and confident. For example, if a model predicts an event will happen with 95% probability, but it does not happen (actual value is 0), the log loss will be very high. It's better to be somewhat wrong than emphatically wrong.

$$
\text{Log Loss} = -\frac{1}{N}\sum_{i=1}^{N} y_i\log(y_i') + (1 - y_i)\log(1 - y_i')
$$

derivating the log loss function wrt weight and bias, we get

$$
\frac{\partial L}{\partial w} = \frac{1}{N} X^T (y_{pred} - y)
$$

$$
\frac{\partial L}{\partial b} = \frac{1}{N} \sum (y_{pred} - y)
$$

Since the dataset is prepared explicitly without using the weights, there could be numerous values for these weights
Also it will be a wrong expectation to match the weights with other different classification algorithms
Because their methods of classifying are different

| Algorithm | What it learns |
| --- | --- |
| Logistic Regression | probability boundary |
| LDA | Gaussian class distributions |
| Naive Bayes | conditional probabilities |
| KNN | neighbor voting |
| Decision Tree | rule splits |
| SVM | maximum margin boundary |

But what we can compare instead are the accuracy and confusion matrix

sources :

https://www.ibm.com/think/topics/classification-machine-learning

https://developers.google.com/machine-learning/crash-course/logistic-regression/loss-regularization