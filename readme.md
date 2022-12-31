# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Python and ML

## 1. Data structures

### 1.1 Strings

#### string interpolation

  name = 'Chris'
  print(f'Hello {name}')
  print('Hey %s %s' % (name, name))
  print(
  "My name is {}".format((name))
  )

### 1.2 Lists

#### copy and deepcopy()

copy - ref on  object,  all elements ref on copy

deepcopy - new object, and every element ref on copy, but when change, new element object

#### list and array

- array same data type
- array +, - linear operations, different methods append
- arrays less memory

### 1.6 Operators

#### difference == and is

Identity (is) and equality (==), can check by id(var), same or note memory cell

## 2.Design patterns

### 1. Decorators

Function of function and return function (logger, get time of execution)

def logging(func):
 def log_function_called():
   print(f'{func} called.')
   func()
 return log_function_called

## 3.OOP

#### exemple (self, ), class(cls, ) and static () methods

 **Методы экземпляра** : принимают параметр self и относятся к определенному экземпляру класса. selfin call

 **Статические методы** : используют декоратор @staticmethod, не связаны с конкретным экземпляром и являются автономными (атрибуты класса или экземпляра не изменяются). - dont change class or example

**Методы класса** : принимают параметр cls, можно изменить сам класс. call (cls, )

#### func and func()

func - object, you can use it as attribute, as var, func()  - call of func

#### ORM

object-relational mapping - To connect database with model data

SQLAlchemy in Flask, ORM in Django

## 4.Modules

package is catalog with modules, all packages are moduls

## 5. Exceptions

### example

try:
    # попробовать сделать это
except:
    # если блок try не сработал, попробовать это
finally:
    # всегда делать это

## 6. Statistic and probability

### Map, MLE and MOP

[https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation]()

It is closely related to the method of [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood "Maximum likelihood") (ML) estimation, but employs an augmented [optimization objective](https://en.wikipedia.org/wiki/Optimization_(mathematics)) "Optimization (mathematics)") which incorporates a [prior distribution](https://en.wikipedia.org/wiki/Prior_distribution "Prior distribution") (that quantifies the additional information available through prior knowledge of a related event) over the quantity one wants to estimate. MAP estimation can therefore be seen as a [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) "Regularization (mathematics)") of maximum likelihood estimation.

The case of {\displaystyle \sigma _{m}\to \infty }![\sigma _{m}\to \infty ](https://wikimedia.org/api/rest_v1/media/math/render/svg/d2e857e35d457c260d420a9e55ec398d96b7e199) is called a non-informative prior and leads to an ill-defined a priori probability distribution; in this case {\displaystyle {\hat {\mu }}_{\mathrm {MAP} }\to {\hat {\mu }}_{\mathrm {MLE} }.}![{\displaystyle {\hat {\mu }}{\mathrm {MAP} }\to {\hat {\mu }}{\mathrm {MLE} }.}](https://wikimedia.org/api/rest_v1/media/math/render/svg/fee6f2f7bd3533fb7ad0cc5880ef649e6af84223)

MLE - maximum likelihood

MOP = method of momentums

### What is maximum likelihood estimation? Could there be any case where it doesn’t exist?
A method for parameter optimization (fitting a model). We choose parameters so as to maximize the likelihood function (how likely the outcome would happen given the current data and our model).
maximum likelihood estimation (MLE) is a method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the likelihood of making the observations given the parameters. MLE can be seen as a special case of the maximum a posteriori estimation (MAP) that assumes a uniform prior distribution of the parameters, or as a variant of the MAP that ignores the prior and which therefore is unregularized.
for gaussian mixtures, non parametric models, it doesn’t exist
### A/B testing
#### In an A/B test, how can you check if assignment to the various buckets was truly random?

* Plot the distributions of multiple features for both A and B and make sure that they have the same shape. More rigorously, we can conduct a permutation test to see if the distributions are the same.
* MANOVA to compare different means

#### What might be the benefits of running an A/A test, where you have two buckets who are exposed to the exact same product?

  * Verify the sampling algorithm is random.

#### What would be the hazards of letting users sneak a peek at the other bucket in an A/B test?
The user might not act the same suppose had they not seen the other bucket. You are essentially adding additional variables of whether the user peeked the other bucket, which are not random across groups.
#### How would you run an A/B test for many variants, say 20 or more?
one control, 20 treatment, if the sample size for each group is big enough.
Ways to attempt to correct for this include changing your confidence level (e.g. Bonferroni Correction) or doing family-wide tests before you dive in to the individual metrics (e.g. Fisher's Protected LSD).

####  How would you run an A/B test if the observations are extremely right-skewed?
lower the variability by modifying the KPI
cap values
percentile metrics
log transform

#### How would you design an experiment to determine the impact of latency on user engagement?
The best way I know to quantify the impact of performance is to isolate just that factor using a slowdown experiment, i.e., add a delay in an A/B test.

### Hypotesis testing

#### What is a p-value? What is the difference between type-1 and type-2 error?
A p-value is defined such that under the null hypothesis less than the fraction p of events have parameter values more extreme than the observed parameter. It is not the probability that the null hypothesis is wrong.
type-1 error: rejecting Ho when Ho is true
type-2 error: not rejecting Ho when Ha is true

## 7. ML
### What is unbiasedness as a property of an estimator? Is this always a desirable property when performing inference? What about in data analysis or predictive modeling?
Unbiasedness means that the expectation of the estimator is equal to the population value we are estimating. This is desirable in inference because the goal is to explain the dataset as accurately as possible. However, this is not always desirable for data analysis or predictive modeling as there is the bias variance tradeoff. We sometimes want to prioritize the generalizability and avoid overfitting by reducing variance and thus increasing bias.



