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

These algorithms can be classified as
Linear - Logistic regression, LDA
Probabilistic / Generative models - LDA, QDA, Naive Bayes
Distance based model - KNN
Trees - Random tree, decision tree
Margin based models - SVM

**Logistic regression**

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

**LDA**

Linear discriminant analysis - used for both dimensionality reduction and classification

LDA works by identifying a linear combination of features that separates or characterizes two or more classes of objects or events. LDA does this by projecting data with two or more dimensions into one dimension so that it can be more easily classified.

It does so by maximizing the distance of means between classes, and minimizing the variance within classes. 

The model assumes every predictor comes from Gaussian distribution. It estimates the mean vector for each class and a shared covariance matrix, then uses these to compute the most probable class for a new observation.

Assumptions :

- The input dataset has a Gaussian distribution, where plotting the data points gives a bell-shaped curve.
- The data set is linearly separable, meaning LDA can draw a straight line or a decision boundary that separates the data points.
- Each class has the same covariance matrix.

Dimensionality reduction involves separating data points with a straight line. Mathematically, linear transformations are analyzed using eigenvectors and eigenvalues.

Preparations to use LDA -

1. preprocess the data to normalize and center it
2. choose appropriate lower number of dimensions
3. regularize the model

How LDA works - 

1. calculate between class means
2. calculate within class variance
3. project data into lower dimensional space

LDA equation is as below

$$
δ(x) = x * ( σ^2 * (μ0-μ1) - 2 * σ^2 * (μ0^2-μ1^2) + ln(P(w0) / P(w1)))
$$

where,
δ(x) represents the linear discriminant function.
x represents the input data point.
μ0 and μ1 are the means of the two classes.
σ^2 is the common within-class variance.
P(ω0) and P(ω1) are the prior probabilities of the two classes.

LDA classification is direct, in the sense that it does not need training iterations.
By directly applying the formula, we will be able to classify.

The LDA function produces score, which is a continuous value. 
Calculate LDA for each class and choose class with largest discriminant value.

ex - 
LDA for class 1 : $\delta_0(x) = x^T \Sigma^{-1} \mu_0 - \frac{1}{2} \mu_0^T \Sigma^{-1} \mu_0 + \ln(\pi_0)$

LDA for class 2 : $\delta_1(x) = x^T \Sigma^{-1} \mu_1 - \frac{1}{2} \mu_1^T \Sigma^{-1} \mu_1 + \ln(\pi_1)$

choose the class for which δ is more

decision boundary equation, when the scores are equal is as follows

$$
x^T \Sigma^{-1} (\mu_1 - \mu_0) - \frac{1}{2} (\mu_1 + \mu_0)^T \Sigma^{-1} (\mu_1 - \mu_0) + \ln \left( \frac{\pi_1}{\pi_0} \right) = 0
$$

if we check the formula, its in the linear form with x appears as a first-order term

Advantages - 
Simple and efficient computation
Works when the dimensions are high
Handle multicollinearity - high correlations bw features

Disadvantages - 
Shared mean distributions - when the features overlap, LDA will fail to create an effective linear decision boundary
Not suitable for unlabeled data

sources:

https://www.ibm.com/think/topics/linear-discriminant-analysis

**QDA**

QDA is same as LDA except that here it assumes the variances of each class to be different.

$$
\delta_k(x) = -\frac{1}{2} \log |\Sigma_k| - \frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log \pi_k
$$

since the covariance matrix is not same, in effect we have the polynomial of x as 2. hence the name quadratic discriminant analysis. here we have to estimate the variance for each classes, hence computationally its more intensive than LDA. but the decision boundary would be more flexible than LDA.

Observation will belong to the class for which the QDA score is high.

Covariance matrix ( Σ ) - Square matrix with variances between different variables (or classes in this context). The diagonal entries of the covariance matrix are the variances and the other entries are the covariances

sources : 

https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html

https://online.stat.psu.edu/stat857/node/80/

Naive Bayes

Using the conditional probability we classify the observation. Similar to LDA, QDA this is also direct approach which does not need model training.

Conditional probability, p(B|A) = p(A|B)p(B) / p(A) 
also , p(A) = Sum i ( p(A | Bi)p(Bi) . where Bi is all the classes 

Example - 

Disease prevalence: P(D) = 0.3
No disease: P(ND) = 0.7

Symptom, S:

P(S | D) = 0.4
P(S | ND) = 0.6

A patient has the symptom. Whats the prob that he has disease?

Find P(D | S) → 

= P ( S | D) P (D) /  ( P ( S | D) P (D) + P ( S | ND) P (ND) )

= 0.4×0.3 / (0.4×0.3)+(0.6×0.7) = 0.12 / 0.54 = 0.22

In spite of their apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many real-world situations, famously document classification and spam filtering. They require a small amount of training data to estimate the necessary parameters.

This algo computes the posterior probability of each class given the observed features using Bayes’ theorem. It assumes that the features are conditionally independent given the class. (It is not completely true in real life. For example, the next word depends on the prior word in a sentence. But this simplicity of the Naive Bayes method and its efficiency makes it up to the assumption held). The observation belongs to that class for which the prob is maximum using the Bayes probability formula.

$$
P(C \mid x_1, x_2, \dots, x_n) \propto P(C) \prod_{i=1}^{n} P(x_i \mid C)
$$

where,

$$
\begin{align*}P(C) & \rightarrow \text{Prior} \\P(x_i \mid C) & \rightarrow \text{Likelihood} \\P(C \mid x) & \rightarrow \text{Posterior}\end{align*}
$$

Different types of Naive Bayes algorithms - 
Gaussian Naive Bayes - when the data is normally distributed. model is fitter by finding mean and variance
Bernoulli - when the output is binary / True - False
Multinomial - multinomial distribution. This variant is useful when using discrete data, such as frequency counts, and it is typically applied within natural language processing use cases, like spam classification.

Advantages -
Scales well, Less complex
Can handle high dimensional data

Disadvantages - 
Zero frequency - if the feature is not in training set, the prior prob becomes zero. (This can be handled via Laplace smoothing)
Assumption of conditional independence of predictors doesnt hold true always

Applications - 
Spam filtering, document classification, sentiment analysis, mental state prediction

Just a peek on posterior formulas of other Naive Bayes - Gaussian, Multinomial, Bernoulli respectively (quadratic, linear, linear decision boundary)

$$
\log P(C=k \mid x) \propto \log P(k) - \frac{1}{2} \sum_{i=1}^{n} \left( \log(2\pi\sigma_{ki}^2) + \frac{(x_i - \mu_{ki})^2}{\sigma_{ki}^2} \right)

\log P(C=k \mid x) \propto \log P(k) + \sum_{i=1}^{n} x_i \log \theta_{ki}

\log P(C=k \mid x) \propto \log P(k) + \sum_{i=1}^{n} \left[ x_i \log \theta_{ki} + (1 - x_i) \log(1 - \theta_{ki}) \right]
$$

[ Naive Bayes becomes linear when log likelihood is linear in features, which happens in discrete count models (text) ]

Sources :

https://scikit-learn.org/stable/modules/naive_bayes.html

https://www.ibm.com/think/topics/naive-bayes