# knowledge wiki

# Python

## 1. Data structures

### type annotation, type conversion (implicit and explicit)

Например  `def indent_right(s: str, width: int) -> str:.`

Например `price: int = 5`

- Implicit

Неявное преобразование типов автоматически выполняется интерпретатором Python.

Python позволяет избежать потери данных в неявном преобразовании типов.

```
integer_number = 123
float_number = 1.23
new_number = integer_number + float_number
```

- Explicit

Явное преобразование типов также называется приведением типов, типы данных объекта преобразуются с использованием предопределенной функции.

При приведении типов может произойти потеря данных, поскольку мы приводим объект к определенному типу данных.

int(‘3’), str(), float()

### 1.1 Strings

### string interpolation

name = ‘Chris’ print(f’Hello {name}‘) print(’Hey %s %s’ % (name, name)) print( “My name is {}”.format((name)) )

### 1.2 Lists

### copy and deepcopy()

copy - ref on object, all elements ref on copy

deepcopy - new object, and every element ref on copy, but when change, new element object

### list and array

- array same data type
- array +, - linear operations, different methods append
- arrays less memory

### 1.6 Operators

### difference == and is

Identity (is) and equality (==), can check by id(var), same or note memory cell

## 2.Design patterns

### Decorators

Function of function and return function (logger, get time of execution)

Декоратор – это функция, которая получает функцию в качестве входных данных и возвращает её в качестве выходных данных, но расширяя её функционал без изменения основной модели.

У декоратора есть одна приятная особенность: вы можете написать код для одной функции, а далее использовать его по-необходимости с другими.

Типичный случай использования декораторов – это логирование.

def logging(func): def log_function_called(): print(f’{func} called.’) func() return log_function_called

### Assigning arguments of function

В Python аргументы передаются путем присваивания.(official guide
Фактические параметры (аргументы) вызова функции вводятся в локальную таблицу символов вызываемой функции при ее вызове; таким образом, аргументы передаются с использованием вызова по значению, «где значение всегда является ссылкой на объект, а не значением объекта», там же сноска добавляет:
На самом деле, «вызов по ссылке на объект» был бы лучшим описанием, поскольку, если передается изменяемый (mutable) объект, вызывающая сторона увидит любые изменения, внесенные в нее вызываемой стороной (элементы, вставленные в 

## 3.OOP

### exemple (self, ), class(cls, ) and static () methods

**Методы экземпляра** : принимают параметр self и относятся к определенному экземпляру класса. selfin call

**Статические методы** : используют декоратор @staticmethod, не связаны с конкретным экземпляром и являются автономными (атрибуты класса или экземпляра не изменяются). - dont change class or example

**Методы класса** : принимают параметр cls, можно изменить сам класс. call (cls, )

### func and func()

func - object, you can use it as attribute, as var, func() - call of func

### Как изменить способ вывода объектов?

Используйте методы **str** и __repr_ *str* - readable **repr** - unumbigous str(3) == str(“3”)

### перегрузка операторов

> под операторами в данном контексте понимаются не только знаки +, -, *, /, обеспечивающие операции сложения, вычитания и др., но также специфика синтаксиса языка, обеспечивающая операции создания объекта, вызова объекта как функции, обращение к элементу объекта по индексу, вывод объекта
> 
> 
> **call**, **getitem**
> 

### ORM

object-relational mapping - To connect database with model data

SQLAlchemy in Flask, ORM in Django

## 4.Modules

### Модуль, пакет, библиотека

– это просто файл Python, который предназначен для импорта в скрипты или в другие модули. Он содержит функции, классы и глобальные переменные.

Пакет – это набор модулей, которые сгруппированы вместе внутри папки для обеспечения согласованной функциональности. Пакеты могут быть импортированы точно так же, как модули. Обычно в них есть **init**.pyfile, который указывает интерпретатору Python обрабатывать их.

Библиотека – это набор пакетов.

package is catalog with modules, all packages are moduls, library is catalog with packages

### Как бы вы использовали * args и **kwargs?

- args и **kwargs – это конструкции, которые делают функции Python более гибкими, принимая изменчивое количество аргументов.
- args передаёт изменчивое количеств
- во аргументов без ключевых слов в список
- *kwargs передаёт изменчивое количество аргументов с ключевыми словами в словарь

## 5. Exceptions

### example

try: # попробовать сделать это except: # если блок try не сработал, попробовать это finally: # всегда делать это

## 6. Multiprocessing and multithreading

### В чём заключается проблема с многопоточностью в python?

Глобальная блокировка интерпретатора (или GIL) не позволяет интерпретатору Python выполнять более одного потока одновременно. Проще говоря, GIL требует, чтобы в Python всегда выполнялся только один поток.

### В чём разница между многопроцессорностью и многопоточностью?

Многопроцессорность и многопоточность – это парадигмы программирования, направленные на ускорение вашего кода.

**Многопроцессорность** – это вариант реализации вычислений, когда для решения некоторой прикладной задачи используется несколько независимых процессоров. Процессоры независимы и не взаимодействуют друг с другом: они не используют одну и ту же область памяти и имеют строгую изоляцию между собой. Что касается приложений, то многопроцессорная обработка подходит для рабочих нагрузок с интенсивным использованием ЦП. Однако он имеет большой объем памяти, который пропорционален количеству процессоров.

С другой стороны, в **многопоточных** приложениях потоки находятся внутри одного процессора. Следовательно, они используют одну и ту же область памяти: они могут изменять одни и те же переменные и могут мешать друг другу. В то время как процессы строго выполняются параллельно, в Python в данный момент времени выполняется только один поток, и это связано с глобальной блокировкой интерпретатора (GIL). Многопоточность подходит для приложений, связанных с вводом-выводом, таких как очистка веб-страниц или извлечение данных из базы данных.

Если вы хотите узнать больше о многопоточности и многопроцессорности, я рекомендую вам ознакомиться с [этой статьей](https://medium.com/contentsquare-engineering-blog/multithreading-vs-multiprocessing-in-python-ece023ad55a).

## 7. Input output

### Как прочитать файл объемом 8 ГБ на Python с помощью компьютера с 2 ГБ ОЗУ

with open(“./large_dataset.txt”) as input_file: for line in input_file: process_line(line)

# Statistic and probability

### Map, MLE and MOP

https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation

It is closely related to the method of [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood) (ML) estimation, but employs an augmented [optimization objective](https://en.wikipedia.org/wiki/Optimization_(mathematics)) “Optimization (mathematics)”) which incorporates a [prior distribution](https://en.wikipedia.org/wiki/Prior_distribution) (that quantifies the additional information available through prior knowledge of a related event) over the quantity one wants to estimate. MAP estimation can therefore be seen as a [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)) “Regularization (mathematics)”) of maximum likelihood estimation.

The case of {

*{m}} is called a non-informative prior and leads to an ill-defined a priori probability distribution; in this case {*

[https://wikimedia.org/api/rest_v1/media/math/render/svg/d2e857e35d457c260d420a9e55ec398d96b7e199](https://wikimedia.org/api/rest_v1/media/math/render/svg/d2e857e35d457c260d420a9e55ec398d96b7e199)

{ }_{ }.}

[https://wikimedia.org/api/rest_v1/media/math/render/svg/fee6f2f7bd3533fb7ad0cc5880ef649e6af84223](https://wikimedia.org/api/rest_v1/media/math/render/svg/fee6f2f7bd3533fb7ad0cc5880ef649e6af84223)

MLE - maximum likelihood

MOP = method of momentums

### What is maximum likelihood estimation? Could there be any case where it doesn’t exist?

A method for parameter optimization (fitting a model). We choose parameters so as to maximize the likelihood function (how likely the outcome would happen given the current data and our model). maximum likelihood estimation (MLE) is a method of estimating the parameters of a statistical model given observations, by finding the parameter values that maximize the likelihood of making the observations given the parameters. MLE can be seen as a special case of the maximum a posteriori estimation (MAP) that assumes a uniform prior distribution of the parameters, or as a variant of the MAP that ignores the prior and which therefore is unregularized. for gaussian mixtures, non parametric models, it doesn’t exist

### A/B testing

### In an A/B test, how can you check if assignment to the various buckets was truly random?

- Plot the distributions of multiple features for both A and B and make sure that they have the same shape. More rigorously, we can conduct a permutation test to see if the distributions are the same.
- MANOVA to compare different means

### What might be the benefits of running an A/A test, where you have two buckets who are exposed to the exact same product?

- Verify the sampling algorithm is random.

### What would be the hazards of letting users sneak a peek at the other bucket in an A/B test?

The user might not act the same suppose had they not seen the other bucket. You are essentially adding additional variables of whether the user peeked the other bucket, which are not random across groups.

### How would you run an A/B test for many variants, say 20 or more?

one control, 20 treatment, if the sample size for each group is big enough. Ways to attempt to correct for this include changing your confidence level (e.g. Bonferroni Correction) or doing family-wide tests before you dive in to the individual metrics (e.g. Fisher’s Protected LSD).

### How would you run an A/B test if the observations are extremely right-skewed?

lower the variability by modifying the KPI cap values percentile metrics log transform

### How would you design an experiment to determine the impact of latency on user engagement?

The best way I know to quantify the impact of performance is to isolate just that factor using a slowdown experiment, i.e., add a delay in an A/B test.

### Hypotesis testing

### What is a p-value? What is the difference between type-1 and type-2 error?

A p-value is defined such that under the null hypothesis less than the fraction p of events have parameter values more extreme than the observed parameter. It is not the probability that the null hypothesis is wrong. type-1 error: rejecting Ho when Ho is true type-2 error: not rejecting Ho when Ha is true

## Bootstrapping

is any test or metric that uses [random sampling with replacement](https://en.wikipedia.org/wiki/Sampling_(statistics)#Replacement_of_selected_units) (e.g. mimicking the sampling process), and falls under the broader class of [resampling](https://en.wikipedia.org/wiki/Resampling_(statistics)) “Resampling (statistics)”) methods. Bootstrapping assigns measures of accuracy (bias, variance, [confidence intervals](https://en.wikipedia.org/wiki/Confidence_interval), prediction error, etc.) to sample estimates. [[1]](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#cite_note-:0-1) [[2]](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#cite_note-2) This technique allows estimation of the sampling distribution of almost any statistic using random sampling methods

It may also be used for constructing [hypothesis tests](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing). It is often used as an alternative to [statistical inference](https://en.wikipedia.org/wiki/Statistical_inference) based on the assumption of a parametric model when that assumption is in doubt, or where parametric inference is impossible or requires complicated formulas for the calculation of [standard errors](https://en.wikipedia.org/wiki/Standard_error).

## Statistical inference

is the process of using data analysis to infer properties of an underlying distribution of probability.[1] Inferential statistical analysis infers properties of a population, for example by **testing hypotheses and deriving estimates** It is assumed that the observed data set is sampled from a larger population.

Inferential statistics can be contrasted with **descriptive** statistics. Descriptive statistics is solely concerned with properties of the observed data, and it does not rest on the assumption that the data come from a larger population. In machine learning, the term inference is sometimes used instead to mean “make a prediction, by evaluating an already trained model”;[2] in this context inferring properties of the model is referred to as training or learning (rather than inference), and using a model for prediction is referred to as inference (instead of prediction); see also predictive inference ( gone to obsolete because new approach - data for analysis has **error**)

## Sample and Population estimator, consistent and unbiased

Sample - статистическая оценка, population - оцениваямая величина

Смещенная и состоятельная оценка

![https://www.notion.soconsistent.png](https://www.notion.soconsistent.png)

Alt text

In [statistics](https://en.wikipedia.org/wiki/Statistics), the **bias of an estimator** (or **bias function** ) is the difference between this [estimator](https://en.wikipedia.org/wiki/Estimator)’s [expected value](https://en.wikipedia.org/wiki/Expected_value) and the [true value](https://en.wikipedia.org/wiki/True_value) of the parameter being estimated. An estimator or decision rule with zero bias is called ***unbiased*** . In statistics, “bias” is an *objective* property of an estimator. Bias is a distinct concept from [consistency](https://en.wikipedia.org/wiki/Consistent_estimator): consistent estimators converge in probability to the true value of the parameter, but may be biased or unbiased; see [bias versus consistency](https://en.wikipedia.org/wiki/Consistent_estimator#Bias_versus_consistency) for more.ML

![https://www.notion.sobiasedness.png](https://www.notion.sobiasedness.png)

Alt text

![https://www.notion.so../../../Downloads/biased2.png](https://www.notion.so../../../Downloads/biased2.png)

An estimator is said to be

**unbiased**

if its bias is equal to zero for all values of parameter

*θ*

, or equivalently, if the expected value of the estimator matches that of the parameter

## What is unbiasedness as a property of an estimator? Is this always a desirable property when performing inference? What about in data analysis or predictive modeling?

Unbiasedness means that the expectation of the estimator is equal to the population value we are estimating. This is desirable in inference because the goal is to explain the dataset as accurately as possible. However, this is not always desirable for data analysis or predictive modeling as there is the bias variance tradeoff. We sometimes want to prioritize the generalizability and avoid overfitting by reducing variance and thus increasing bias.

# ML

## Metrics

## Multicollinearity, correlation, covariance

regressor - dependent variable

*Correlation is a statistical measure that indicates the extent to which two or more variables move together*

*Covariance is another measure that describes the degree to which two variables tend to deviate from their means in similar ways*

*Collinearity is a linear association between two predictors - when correlation > 0.7*

collinearity can be detected :

- Prominent changes in the estimated regression coefficients by adding or deleting a predictor
- correlation matrix
- VIF
- PCA, PCR

**principal component regression** ( **PCR** ) is a [regression analysis](https://en.wikipedia.org/wiki/Regression_analysis) technique that is based on [principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA). More specifically, PCR is used for [estimating](https://en.wikipedia.org/wiki/Estimation) the unknown [regression coefficients](https://en.wikipedia.org/wiki/Linear_regression) in a [standard linear regression model](https://en.wikipedia.org/wiki/Linear_regression).

In PCR, instead of regressing the dependent variable on the explanatory variables directly, the [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) of the explanatory variables are used as [regressors](https://en.wikipedia.org/wiki/Dependent_and_independent_variables). One typically uses only a subset of all the principal components for regression, making PCR a kind of [regularized](https://en.wikipedia.org/wiki/Regularization_(mathematics)) “Regularization (mathematics)”) procedure and also a type of [shrinkage estimator](https://en.wikipedia.org/wiki/Shrinkage_estimator).

You have several variables that are positively correlated with your response, and you think combining all of the variables could give you a good prediction of your response. However, you see that in the multiple linear regression, one of the weights on the predictors is negative. What could be the issue?

- Multicollinearity refers to a situation in which two or more explanatory variables in a [multiple regression](https://en.wikipedia.org/wiki/Multiple_regression) model are highly linearly related.
- Leave the model as is, despite multicollinearity. The presence of multicollinearity doesn’t affect the efficiency of extrapolating the fitted model to new data provided that the predictor variables follow the same pattern of multicollinearity in the new data as in the data on which the regression model is based.
- principal component regression

## Calibration of ML models

Calibrationclassifiercv in sklearn

[calibration of lightgbm (log regression for leaves)](https://www.notion.so/calibration-of-lightgbm-log-regression-for-leaves-bf8f65b6114d4493be7c0855be318ef3) 

## Linear regression

### our linear regression didn’t run and communicates that there are an infinite number of best estimates for the regression coefficients. What could be wrong?

- p > n.
- If some of the explanatory variables are perfectly correlated (positively or negatively) then the coefficients would not be unique.

## regularization

is a process that changes the result answer to be “simpler”. It is often used to obtain results for ill-posed problems or to prevent overfitting.[2]

Although regularization procedures can be divided in many ways, following delineation is particularly helpful:

Explicit regularization is regularization whenever one explicitly adds a term to the optimization problem. These terms could be priors, penalties, or constraints. Explicit regularization is commonly employed with ill-posed optimization problems. The regularization term, or penalty, imposes a cost on the optimization function to make the optimal solution unique. Implicit regularization is all other forms of regularization. This includes, for example, early stopping, using a robust loss function, and discarding outliers. Implicit regularization is essentially ubiquitous in modern machine learning approaches, including stochastic gradient descent for training deep neural networks, and ensemble methods (such as random forests and gradient boosted trees).

## What is the curse of dimensionality?

- High dimensionality makes clustering hard, because having lots of dimensions means that everything is “far away” from each other.
- For example, to cover a fraction of the volume of the data we need to capture a very wide range for each variable as the number of variables increases
- All samples are close to the edge of the sample. And this is a bad news because prediction is much more difficult near the edges of the training sample.
- The sampling density decreases exponentially as p increases and hence the data becomes much more sparse without significantly more data.
- We should conduct PCA to reduce dimensionality

# SQL

## Data modelling

 ED Modeler is a tool used for designing and analyzing data structures with standardized designs, it supports deployment of diagrammatic data, regardless of its location and structure, it offers automated features to generate schema and hybrid architecture.
Key features:
Visual data modeling: Create data models using a visual interface, making it easy to see the relationships between different entities and data elements.

Reverse engineering: Reverse engineer existing databases to create a visual representation of the data structures.

Forward engineering: Generate SQL scripts or DDL statements based on the data model, to create or modify databases directly from the data model.

Collaboration: Multiple users can work on a data model simultaneously, making it easy for teams to collaborate on data modeling projects.

Documentation: Generate detailed documentation of data models, including a list of entities, attributes, and relationships.

Data governance: Allows organizations to enforce data governance policies and ensure compliance with industry standards and regulations.

Integration: Easy integration with database management systems, data warehousing solutions, and business intelligence tools.

Data modeling: Create visual data models, including entity-relationship diagrams (ERDs) and data flow diagrams (DFDs)

# System Design