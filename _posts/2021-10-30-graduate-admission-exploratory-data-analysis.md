---
title: Exploratory Data Analysis of Graduate Admission data
layout: single
classes: wide
toc: true
custom_css: plotly
---

# Introduction

In this post, I'll apply different EDA (Exploratory Data Analysis) techniques on the [Graduate Admission 2 data](https://www.kaggle.com/mohansacharya/graduate-admissions).

The goal in this data is to predict the _student's chance of admission_ to a postgraduate education, given several _predictor_ variables for the student.

# Import libraries

```python
import pandas as pd
import plotly.express as px
from scipy import stats
```

# Load data

There are two data files:

- `Admission_Predict.csv`
- `Admission_Predict_Ver1.1.csv`

Will use the second one, since it contains more data points.

```python
df = pd.read_csv("data/Admission_Predict_Ver1.1.csv")
```

According to the dataset author on Kaggle, the columns in this data represents:

- `GRE Score`: The Graduate Record Examinations is a standardized test that is an admissions requirement for many graduate schools in the United States and Canada.
- `TOEFL Score`: Score in TOEFL exam.
- `University Rating`: Student undergraduate university ranking.
- `SOP`: Statement of Purpose strength.
- `LOR`: Letter of Recommendation strength.
- `CGPA`: Undergraduate GPA.
- `Research`: Whether student has research experience or not.
- `Chance of Admit`: Admission chance.

# Getting to know the data

In this section, we'll take a quick look at the data, to see how many row are there, and whther there are any missing values or not, to decie what kind of preprocessing will be needed.

```python
df.head()
```

<div>
  <style scoped>
    .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
    }

    .dataframe tbody tr th {
      vertical-align: top;
    }

    .dataframe thead th {
      text-align: right;
    }

  </style>
  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: right">
        <th></th>
        <th>Serial No.</th>
        <th>GRE Score</th>
        <th>TOEFL Score</th>
        <th>University Rating</th>
        <th>SOP</th>
        <th>LOR</th>
        <th>CGPA</th>
        <th>Research</th>
        <th>Chance of Admit</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>1</td>
        <td>337</td>
        <td>118</td>
        <td>4</td>
        <td>4.5</td>
        <td>4.5</td>
        <td>9.65</td>
        <td>1</td>
        <td>0.92</td>
      </tr>
      <tr>
        <th>1</th>
        <td>2</td>
        <td>324</td>
        <td>107</td>
        <td>4</td>
        <td>4.0</td>
        <td>4.5</td>
        <td>8.87</td>
        <td>1</td>
        <td>0.76</td>
      </tr>
      <tr>
        <th>2</th>
        <td>3</td>
        <td>316</td>
        <td>104</td>
        <td>3</td>
        <td>3.0</td>
        <td>3.5</td>
        <td>8.00</td>
        <td>1</td>
        <td>0.72</td>
      </tr>
      <tr>
        <th>3</th>
        <td>4</td>
        <td>322</td>
        <td>110</td>
        <td>3</td>
        <td>3.5</td>
        <td>2.5</td>
        <td>8.67</td>
        <td>1</td>
        <td>0.80</td>
      </tr>
      <tr>
        <th>4</th>
        <td>5</td>
        <td>314</td>
        <td>103</td>
        <td>2</td>
        <td>2.0</td>
        <td>3.0</td>
        <td>8.21</td>
        <td>0</td>
        <td>0.65</td>
      </tr>
    </tbody>
  </table>
</div>

```python
df.columns
```

    Index(['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
           'LOR ', 'CGPA', 'Research', 'Chance of Admit '],
          dtype='object')

```python
df.shape
```

    (500, 9)

```python
df.isnull().sum()
```

    Serial No.           0
    GRE Score            0
    TOEFL Score          0
    University Rating    0
    SOP                  0
    LOR                  0
    CGPA                 0
    Research             0
    Chance of Admit      0
    dtype: int64

```python
df.dtypes
```

    Serial No.             int64
    GRE Score              int64
    TOEFL Score            int64
    University Rating      int64
    SOP                  float64
    LOR                  float64
    CGPA                 float64
    Research               int64
    Chance of Admit      float64
    dtype: object

The dataset consists of 500 samples and 9 columns: 8 _predictors_ and one _target_ variable.

There are no missing values (which is a very good thing!), but some column names need to be cleaned, and the `Serial No.` must be removed, as it has nothing to do with the student's overall admission chance.

Lookin at the `dtypes` it seems that all columns are in the correct data type, discrete columns are in `int64` and continuous in `float64`.

# Data cleaning and Preprocessing

As stated in the previous section, only few _cleaning_ will be performed, mainly:

- remove extra whitespace from column names.
- drop `Serial No.` column
- convert `Research` column to bool.

```python
df.columns
```

    Index(['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
           'LOR ', 'CGPA', 'Research', 'Chance of Admit '],
          dtype='object')

Pandas has a great feature which allows us to apply multiple functions on the `DataFrame` in a sequential order: the [pipe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html) method.

Here, I'll define two separate functions for applying each processing step, and then call them using the `pipe` function.

```python
def read_data():
    temp_df = pd.read_csv("data/Admission_Predict_Ver1.1.csv")
    return temp_df
```

```python
def normalize_column_names(temp_df):
    return temp_df.rename(
        columns={"LOR ": "LOR", "Chance of Admit ": "Chance of Admit"}
    )
```

```python
def drop_noisy_columns(temp_df):
    return temp_df.drop(columns=["Serial No."])
```

```python
def normalize_dtypes(temp_df):
    return temp_df.astype({"Research": bool, "University Rating": str})
```

```python
def sort_uni_ranking(temp_df):
    return temp_df.sort_values(by="University Rating")
```

Now, we plug them together:

```python
df = (
    read_data()
    .pipe(normalize_column_names)
    .pipe(drop_noisy_columns)
    .pipe(normalize_dtypes)
    .pipe(sort_uni_ranking)
)
```

```python
df.columns
```

    Index(['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA',
           'Research', 'Chance of Admit'],
          dtype='object')

```python
df.shape
```

    (500, 8)

```python
df.dtypes
```

    GRE Score              int64
    TOEFL Score            int64
    University Rating     object
    SOP                  float64
    LOR                  float64
    CGPA                 float64
    Research                bool
    Chance of Admit      float64
    dtype: object

We _cleaned_ the data with a _clean_ code!

# Exploratory Data Analysis (EDA)

In this section, we'll explore the data _visually_ and summarize it using _descriptive statistic_ methods.

To keep things simpler, we'll divide this section into three subsections:

1. Univariate analysis: in this section we'll focus only at one variable at a time, and study the variable descriptive statistics with some charts like: Bar chart, Line chart, Histogram, Boxplot, etc ..., and how the variable is distributed, and if there is any _skewness_ in the distribution.
2. Bivariate analysis: in this section we'll study the relation between _two_ variables, and present different statistics such as Correlation, Covariance, and will use some other charts like: scatterplot, and will make use of the `hue` parameter of the previous charts.
3. Multivariate analysis: in this section we'll study the relation between three or more variables, and will use additional type of charts, such as parplot.

## Univariate Analysis

Here in this section, will perform analysis on each variable individually, but according to the variable type different methods and visualization will be used, main types of variables:

- Numerical: numerical variables are variables which measures things like: counts, grades, etc ..., and they don't have a _finite_ set of values, and they can be divided to:
  - Continuous: continuous variables are continous measurements such as weight, height.
  - Discrete: discrete variables represent counts such as number of children in a family, number of rooms in a house.
- Categorical: a categorical variable is a variable which takes one of a limited values, and it can be further divided to:
  - Nominal: nominal variable has a finite set of possible values, which don't have any ordereing relation among them, like countries, for example we can't say that `France` is higher than `Germany`: `France` > `Germany`, therfore, there's no sense of ordering between the values in a noinal variable.
  - Ordinal: in contrast to `Nominal` variable, ordinal varible defines an ordering relation between the values, such as the student performance in an exam, which can be: `Bad`, `Good`, `Very Good`, and `Excellent` (there's an ordering relation among theses values, and we can say that `Bad` is lower than `Good`: `Bad` < `Good`)
  - Binary: binary variables are a special case of nominal variables, but they only have _two_ possible values, like admission status which can either be `Accepted` or `Not Accepted`.

resources:

- [Variable types and examples](https://www.statsandr.com/blog/variable-types-and-examples/)
- [What is the difference between ordinal, interval and ratio variables? Why should I care?](https://www.graphpad.com/support/faq/what-is-the-difference-between-ordinal-interval-and-ratio-variables-why-should-i-care/)

Let's see what are the types of variables in our dataset:

```python
df.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Chance of Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.00000</td>
      <td>500.000000</td>
      <td>500.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>316.472000</td>
      <td>107.192000</td>
      <td>3.374000</td>
      <td>3.48400</td>
      <td>8.576440</td>
      <td>0.72174</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.295148</td>
      <td>6.081868</td>
      <td>0.991004</td>
      <td>0.92545</td>
      <td>0.604813</td>
      <td>0.14114</td>
    </tr>
    <tr>
      <th>min</th>
      <td>290.000000</td>
      <td>92.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>6.800000</td>
      <td>0.34000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>308.000000</td>
      <td>103.000000</td>
      <td>2.500000</td>
      <td>3.00000</td>
      <td>8.127500</td>
      <td>0.63000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>317.000000</td>
      <td>107.000000</td>
      <td>3.500000</td>
      <td>3.50000</td>
      <td>8.560000</td>
      <td>0.72000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>325.000000</td>
      <td>112.000000</td>
      <td>4.000000</td>
      <td>4.00000</td>
      <td>9.040000</td>
      <td>0.82000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>340.000000</td>
      <td>120.000000</td>
      <td>5.000000</td>
      <td>5.00000</td>
      <td>9.920000</td>
      <td>0.97000</td>
    </tr>
  </tbody>
</table>
</div>

- Discrete: `GRE Score` and `TOEFL Score` are discrete variables.
- Continuous: `CGPA` and `Chance of Admit` are continuous variables.
- Ordinal: `University Rating`, `SOP` and `LOR` are ordinal variables.
- Binary: `Research` is a binary variable.

### `GRE Score`

The `GRE Score` is a discrete variable.

```python
df["GRE Score"].describe()
```

    count    500.000000
    mean     316.472000
    std       11.295148
    min      290.000000
    25%      308.000000
    50%      317.000000
    75%      325.000000
    max      340.000000
    Name: GRE Score, dtype: float64

```python
print(df["GRE Score"].mode())
```

    0    312
    dtype: int64

```python
print(stats.skew(df["GRE Score"]))
```

    -0.03972223277299966

```python
fig = px.histogram(
    df,
    x="GRE Score",
    nbins=20,
    marginal="box",
    title="Distribution of the <b>GRE Score</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/1_gre_score_distribution.html %}

We can conclude from the previous chart the following:

- The GRE scores are _very close_ to a normal distribution, with a small negative skewnewss (left skewed).
- The most common scores are between `310` and `325`.
- The average score is `316` with a standard deviation of `11.2`.
- There are no outliers.

This variable doesn't need any further processing.

### `TOEFL Score`

The `TOEFL Score` is a discrete variable.

```python
df["TOEFL Score"].describe()
```

    count    500.000000
    mean     107.192000
    std        6.081868
    min       92.000000
    25%      103.000000
    50%      107.000000
    75%      112.000000
    max      120.000000
    Name: TOEFL Score, dtype: float64

```python
print(df["TOEFL Score"].mode())
```

    0    110
    dtype: int64

```python
print(stats.skew(df["TOEFL Score"]))
```

    0.09531393010261811

```python
fig = px.histogram(
    df,
    x="TOEFL Score",
    marginal="box",
    nbins=15,
    title="Distribution of <b>TOEFL Score</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/2_toefl_score_distribution.html %}

```python
df["TOEFL Score"].value_counts()[:4]
```

    110    44
    105    37
    104    29
    106    28
    Name: TOEFL Score, dtype: int64

From the previous chart, we can conclude:

- TOEFL scores are also normally distributed, with a small positive (right skewness).
- The average TOEFL score is `107` with a standard deviation `6`.
- The most common scores are: `110`, `105`, `104` and `112`.
- There are no outliers.

The variable doesn't need any further processing.

### `University Rating`

The `University Rating` is an ordinal variable, it represents the student's undergraduate university ranking on a scale 1-5.

```python
df["University Rating"].value_counts()
```

    3    162
    2    126
    4    105
    5     73
    1     34
    Name: University Rating, dtype: int64

```python
temp_df = df.groupby(by="University Rating", as_index=False).agg(
    counts=pd.NamedAgg(column="University Rating", aggfunc="count")
)
```

```python
temp_df["University Rating"] = temp_df["University Rating"].astype(str)
```

```python
fig = px.bar(
    data_frame=temp_df,
    x="University Rating",
    y="counts",
    color="University Rating",
    color_discrete_sequence=px.colors.qualitative.D3,
    title="Distribution of <b>University rating</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/3_uni_rating_distribution.html %}

We can see that the most common rating is in the middle: `3`. The chart shows that the ratings are distributed in a similar fashion to the normal distrbution.

### `SOP`

`SOP` stands for the strength of _Statement of Purpose_ which is a necessary document for graduate applications. The values were (mostly) entered by the students, and it's on scale 1-5, so this is an ordinal variable.

```python
df["SOP"].value_counts()
```

    4.0    89
    3.5    88
    3.0    80
    2.5    64
    4.5    63
    2.0    43
    5.0    42
    1.5    25
    1.0     6
    Name: SOP, dtype: int64

```python
temp_df = df.groupby(by="SOP", as_index=False).agg(
    counts=pd.NamedAgg(column="SOP", aggfunc="count")
)
```

```python
temp_df["SOP"] = temp_df["SOP"].astype(str)
```

```python
fig = px.bar(
    data_frame=temp_df,
    x="SOP",
    y="counts",
    color="SOP",
    color_discrete_sequence=px.colors.qualitative.Prism,
    title="Distribution of <b>SOP</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/4_sop_distribution.html %}

Most students estimated the strength of their _Statement of Purpose_ between `3` and `4`.

### `LOR`

`LOR` stands for the strength of _Letter of Recommendation_. The values were (mostly) entered by the students, and it's on scale 1-5, so this is an ordinal variable.

```python
df["LOR"].value_counts()
```

    3.0    99
    4.0    94
    3.5    86
    4.5    63
    2.5    50
    5.0    50
    2.0    46
    1.5    11
    1.0     1
    Name: LOR, dtype: int64

```python
temp_df = df.groupby(by="LOR", as_index=False).agg(
    counts=pd.NamedAgg(column="LOR", aggfunc="count")
)
```

```python
temp_df["LOR"] = temp_df["LOR"].astype(str)
```

```python
fig = px.bar(
    data_frame=temp_df,
    x="LOR",
    y="counts",
    color="LOR",
    color_discrete_sequence=px.colors.qualitative.Prism,
    title="Distribution of <b>LOR</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/5_lor_distribution.html %}

Most of the students rated the strength of ther _Letter of Recommendation_ between `3` and `4`.

### `CGPA`

The `CGPA` stands for the student's _cumulative grade point average_, which represents the average of grade points obtained in all the subjects by the student.

It's a continuous variable, on a scale 0-10.

```python
df["CGPA"].describe()
```

    count    500.000000
    mean       8.576440
    std        0.604813
    min        6.800000
    25%        8.127500
    50%        8.560000
    75%        9.040000
    max        9.920000
    Name: CGPA, dtype: float64

```python
print(stats.skew(df["CGPA"]))
```

    -0.026532613141817388

```python
fig = fig = px.histogram(
    data_frame=df,
    x="CGPA",
    marginal="box",
    nbins=12,
    title="Distribution of <b>CGPA</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/5_lor_distribution.html %}

As we can see, this variable is _very_ close to a normal distribution, with a small negative (left) skewness, and there are no outliers.

### `Research`

The `Research` variable indicates whether the student has any research experience or not, so it's a `Binary` variable.

Although, it would be better to have a variable like `Research duration` which expresses for how long was the student involved in a research activity.

```python
df["Research"].value_counts()
```

    True     280
    False    220
    Name: Research, dtype: int64

```python
temp_df = df.groupby(by="Research", as_index=False).agg(
    counts=pd.NamedAgg(column="Research", aggfunc="count")
)
```

```python
fig = px.bar(
    data_frame=temp_df,
    x="Research",
    y="counts",
    color="Research",
    color_continuous_scale=px.colors.qualitative.D3,
    title="Distribution of <b>Research</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/7_research_distribution.html %}

From this plot, we can see that the number of students who have a research experience is _almost_ equal to the number of students who don't.

Later, we'll study the relation of this variable with other variables.

### `Chance of Admit`

Quoting the dataste author from this [thread](https://www.kaggle.com/mohansacharya/graduate-admissions/discussion/79063#464899):

> chance of admit is a parameter that was asked to individuals (some values manually entered) before the results of the application

So thie column is not an actual _probability of admission_ estimated by the universities or something, rather, it's an estimation by the student themselves of how likely they'll be admitted to the university.

```python
df["Chance of Admit"].describe()
```

    count    500.00000
    mean       0.72174
    std        0.14114
    min        0.34000
    25%        0.63000
    50%        0.72000
    75%        0.82000
    max        0.97000
    Name: Chance of Admit, dtype: float64

```python
print(stats.skew(df["Chance of Admit"]))
```

    -0.2890955854789938

```python
fig = px.histogram(
    data_frame=df,
    x="Chance of Admit",
    marginal="box",
    title="Distribution of <b>Chance of Admit</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/8_chance_of_admit_distribution.html %}

```python
df[df["Chance of Admit"] < 0.36]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Chance of Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>376</th>
      <td>297</td>
      <td>96</td>
      <td>2</td>
      <td>2.5</td>
      <td>2.0</td>
      <td>7.43</td>
      <td>False</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>92</th>
      <td>298</td>
      <td>98</td>
      <td>2</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>8.03</td>
      <td>False</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>
</div>

The plot shows that most student estimated their chance of admission between `0.7` and `0.75`.

The distribution is _moderately_ skewed to the left with a negative skew value `-0.29`.

There are also two outlier values `0.34`.

## Bivariate Analysis

In this section, we'll focus on studying the relationship between two different variables, to answer different question, like

- What is the relation between variable `x` and variable `y`? is it linear or non-linear?
- In case of a linear relation, is positive linear relation or negative linear relation? and how _strong_ is the relation?
- How the distribution for two variables changes?

### Correlation matrix

We'll start off by computing the correlation matrix using `.corr` method of pandas, which computes the pairwise correlation of columns.

The method used for calculating the correlation between two variables `x` and `y` is the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).

The pearson coefficient is a measure of the linear correlation between two variables `x` and `y`, and it takes values between `-1` and `+1`.

A negative value indicates a negative correlation (i.e. when one variable _increases_ the other _decreases_), and a positive value is the opposite (the two variables _increases_/_decreases_ at the same time)

<!-- ![Correlation coefficient](https://upload.wikimedia.org/wikipedia/commons/3/34/Correlation_coefficient.png) -->
<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/34/Correlation_coefficient.png">
</p>

Here, we'll compute the correlations only for `GRE Score`, `TOEFL Score` and `CGPA` variables, because they are _numeric_ variables, and they weren't estimated by the students themselves.

```python
numeric_cols = ["GRE Score", "TOEFL Score", "CGPA"]
```

```python
corr = df[numeric_cols].corr()
```

```python
fig = px.imshow(
    corr,
    color_continuous_scale="PuBu",
    color_continuous_midpoint=0.6,
    title="Correlation matrix",
)

fig.show()
```

{% include graduate-admission-figures/9_numerical_variables_correlation_matrix.html %}

We can see from the correlation matrix that the three variables have a strong positive correlation. We'll look closer at the relations between variables using scatter plots.

### Scatter plot

Scatter plots are a good way to show the spread of points for two variables `x` and `y`, and view the relation between the two variables (e.g. linear, non-linear, ...) and the trend of the linear relation (positive, negative)

An easy way to show multiple scatter plots on the same figure is either using `scatter_matrix` or `pairplot`.

```python
fig = px.scatter_matrix(
    df,
    dimensions=numeric_cols,
    title="Scatter matrix of student's <b>TOEFL Score</b>, <b>GRE Score</b>, and <b>CGPA</b>",
)

fig.update_traces(diagonal_visible=False)

fig.show()
```

{% include graduate-admission-figures/10_toefl_gre_cgpa_scatter_matrix.html %}

It's evident from these plots that the relation between the variables is positive linear relation, with some outlier points, and they all have an upward trend.

Let's show the scatter for each two variables at a time:

#### `TOEFL Score` vs. `GRE Score`

```python
corr_value = df["TOEFL Score"].corr(df["GRE Score"])

fig = px.scatter(
    data_frame=df,
    x="TOEFL Score",
    y="GRE Score",
    marginal_x="histogram",
    marginal_y="histogram",
    trendline="ols",
    trendline_color_override="red",
    title=f"Correlation between <b>TOEFL Score</b> and <b>GRE Score</b> is: {corr_value:.2f}",
)

fig.show()
```

{% include graduate-admission-figures/11_toefl_gre_scatter.html %}

#### `TOEFL Score` vs. `CGPA`

```python
corr_value = df["TOEFL Score"].corr(df["CGPA"])

fig = px.scatter(
    data_frame=df,
    x="TOEFL Score",
    y="CGPA",
    marginal_x="histogram",
    marginal_y="histogram",
    trendline="ols",
    trendline_color_override="red",
    title=f"Correlation between <b>TOEFL Score</b> and <b>CGPA</b> is: {corr_value:.2f}",
)

fig.show()
```

{% include graduate-admission-figures/12_toefl_cgpa_scatter.html %}

#### `GRE Score` vs. `CGPA`

```python
corr_value = df["GRE Score"].corr(df["CGPA"])

fig = px.scatter(
    data_frame=df,
    x="GRE Score",
    y="CGPA",
    marginal_x="histogram",
    marginal_y="histogram",
    trendline="ols",
    trendline_color_override="red",
    title=f"Correlation between <b>GRE Score</b> and <b>CGPA</b> is: {corr_value:.2f}",
)

fig.show()
```

{% include graduate-admission-figures/13_gre_cgpa_scatter.html %}

From the previous three charts we can say that: students who perform well in their `TOEFL` exams tend to also perform well in `GRE` exams, and they _mostly_ have high `GPA` (higher than 9).

### Bivariate distributions

Another way to study the relation between two variables is with 2D Histograms (distribution).

Just like the distributions we used in the **Univariate Analysis** section, we can show the distribution for two variables `x` and `y`, which would give us better insights on how much the values from the two variables overlap, and show cluster regions in the 2D space.

Compared to scatter plots, 2D histograms are better at handling large amounts of data, as they use rectangular bins, and count the number of points withing each bin.

#### `TOEFL Score` vs. `GRE Score`

```python
fig = px.density_heatmap(
    data_frame=df,
    x="TOEFL Score",
    y="GRE Score",
    color_continuous_scale="PuBu",
    title="Joint distribution of <b>TOEFL Score</b> and <b>GRE Score</b> variables",
)

fig.show()
```

{% include graduate-admission-figures/14_toefl_gre_joint_distribution.html %}

We can see from this chart some _clusters_ (regions).

For example there are two clusters of students who scored between `110` and `115` in the `TOEFL` exam and between `320` and `330` in the `GRE` exam. These two clusters account for about 100 students (which is 20% of the total dataset).

#### `TOEFL Score` vs. `CGPA`

```python
fig = px.density_heatmap(
    data_frame=df,
    x="TOEFL Score",
    y="CGPA",
    color_continuous_scale="PuBu",
    title="Joint distribution of <b>TOEFL Score</b> and <b>CGPA</b> variables",
)

fig.show()
```

{% include graduate-admission-figures/15_toefl_cgpa_joint_distribution.html %}

This chart shows that about `170` students has `TOEFL` score in range `[105-115]` and their `CGPA` is in range `[8-9]`

#### `GRE Score` vs. `CGPA`

```python
fig = px.density_heatmap(
    data_frame=df,
    x="GRE Score",
    y="CGPA",
    color_continuous_scale="PuBu",
    title="Bivariate distribution of <b>GRE Score</b> and <b>CGPA</b> variables",
)

fig.show()
```

{% include graduate-admission-figures/16_gre_cgpa_joint_distribution.html %}

This chart shows that almost `120` students has `GRE` scores in the range `[310-320]` and their `CGPA` is in thae range `[8-9]`.

### `Research`

One interesting question would be: How does research experience affects other variables? Do students who have research experience have beeter _CGPA_? Do they have better scores in _TOEFL_ or _GRE_ (or both)?

We can display the same distributions we used in the _Univariate analysis_ section, with conditioning on `Research` variable:

```python
fig = px.histogram(
    data_frame=df,
    x="TOEFL Score",
    color="Research",
    barmode="group",
    title="Conditional distribution of <b>TOEFL Score</b> variable on the <b>Research</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/17_toefl_conditional_distribution_on_research.html %}

```python
fig = px.histogram(
    data_frame=df,
    x="GRE Score",
    color="Research",
    barmode="group",
    title="Conditional distribution of <b>GRE Score</b> variable on the <b>Research</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/18_gre_conditional_distribution_on_research.html %}

```python
fig = px.histogram(
    data_frame=df,
    x="CGPA",
    color="Research",
    barmode="group",
    title="Conditional distribution of <b>CGPA</b> variable on the <b>Research</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/19_cgpa_conditional_distribution_on_research.html %}

We can see that students who engage in research activities and have research experience, tend to perform better in both _TOEFL_ and _GRE_ exams, and they have higher _GPA_, in comparison to students who have no research experience.

### `University Rating`

The `University Rating` variable represents the ranking of the university from which the student graduated.

We might ask: How does the relation between student different scores changes for different university ranking?

Let's see how the university ranking affects student's scores and GPA:

```python
fig = px.histogram(
    data_frame=df,
    x="TOEFL Score",
    color="University Rating",
    barmode="group",
    color_discrete_sequence=px.colors.sequential.Blugrn,
    title="Conditional distribution of <b>TOEFL Score</b> variable on the <b>University Rating</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/20_toefl_conditional_distribution_on_uni_rating.html %}

```python
fig = px.histogram(
    data_frame=df,
    x="GRE Score",
    color="University Rating",
    color_discrete_sequence=px.colors.sequential.Blugrn,
    barmode="group",
    title="Conditional distribution of <b>GRE Score</b> variable on the <b>University Rating</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/21_gre_conditional_distribution_on_uni_rating.html %}

```python
fig = px.histogram(
    data_frame=df,
    x="CGPA",
    color="University Rating",
    color_discrete_sequence=px.colors.sequential.Blugrn,
    barmode="group",
    title="Conditional distribution of <b>CGPA</b> variable on the <b>University Rating</b> variable",
)

fig.show()
```

{% include graduate-admission-figures/22_cgpa_conditional_distribution_on_uni_rating.html %}

The university ranking plays an important role in student's scores and GPA, we can observe that students who go to higher ranking universities, have higher scores in the _TOEFL_ and _GRE_ exams, and they have higher _GPA_.

## Multivariate Analysis

So far, all the plots we used before were used either to explore one variable, or to show the relation between a pair of variables.

However, we are often interested in answering the question: **How does the relation between two variables changes as a function of a _third_ variable?**

In this section, we'll focus on answering these kinds of questions, where we'll use similar plots to the ones we used before, with conditioning on other variable.

### Scatter matrix with `Research`

We start by plotting the scatter matrix for variables `TOEFL Score`, `GRE Score` and `CGPA` (just as we did in bivariate analysis section), with showing the relation to `Research` variable with color coding.

This will draw an overall picture of how these variables are related to each other, and how this relation changes when whether student has a research experience or not.

```python
fig = px.scatter_matrix(
    data_frame=df,
    dimensions=numeric_cols,
    color="Research",
    title="Scatter matrix for <b>TOEFL Score</b>, <b>GRE Score</b>, and <b>CGPA</b> conditioning on <b>Research</b> variable",
)

fig.update_traces(diagonal_visible=False)

fig.show()
```

{% include graduate-admission-figures/23_toefl_gre_cgp_scatter_conditional_on_research.html %}

We can how students who have research experience tend to have higher scores and better GPA.

### Bivariate distribution with `University Rating`

As we saw earlier, the ranking of student's university plays an important role in other variables, and higher university ranking is linked with better performance in TOEFL and GRE exams, and higher GPA.

Here, we'll show the bivariate distribution of each pair of variables, for different university ranking values.

This way, we will be able to study the relation between two variables `x` and `y` as a function of the university ranking value.

#### `TOEFL Score` vs. `GRE Score`

```python
fig = px.density_heatmap(
    data_frame=df,
    x="TOEFL Score",
    y="GRE Score",
    color_continuous_scale="PuBu",
    facet_col="University Rating",
    title="<b>TOEFL Score</b> vs. <b>GRE Score</b> for different values of <b>University Rating</b>",
)

fig.show()
```

{% include graduate-admission-figures/24_toefl_gre_joint_distribution_conditional_on_uni_rating.html %}

#### `TOEFL Score` vs. `CGPA`

```python
fig = px.density_heatmap(
    data_frame=df,
    x="TOEFL Score",
    y="CGPA",
    facet_col="University Rating",
    color_continuous_scale="PuBu",
    title="<b>TOEFL Score</b> vs. <b>CGPA</b> for different values of <b>University Rating</b>",
)

fig.show()
```

{% include graduate-admission-figures/25_toefl_cgpa_joint_distribution_conditional_on_uni_rating.html %}

#### `GRE Score` vs. `CGPA`

```python
fig = px.density_heatmap(
    data_frame=df,
    x="GRE Score",
    y="CGPA",
    facet_col="University Rating",
    color_continuous_scale="PuBu",
    title="<b>GRE Score</b> vs. <b>CGPA</b> for different values of <b>University Rating</b>",
)

fig.show()
```

{% include graduate-admission-figures/26_gre_cgpa_joint_distribution_conditional_on_uni_rating.html %}

All these charts support our previous hypothesis: **University ranking influences positively student's performance.**

### `Research` and `University Rating`

In all previous plots we used either the `Research` variable or `University Rating` variable, however, it would very helpful to see how these two variables interact with each other, and how other vriables interact with them.

Here, we'll use scatterplot to show relation between two variables `x` and `y`, and show how this relation changes for different values of `Research` and `University Rating`.

This would be useful for answering questions:

- Do students who go to higher-ranking universitries, are more involved in research activiteis?
- How research experience and university ranking influence student's performance in exams and GPA?
- Are there any outlier points?

#### `TOEFL Score` vs. `GRE Score`

```python
fig = px.scatter(
    data_frame=df,
    x="TOEFL Score",
    y="GRE Score",
    color="Research",
    facet_col="University Rating",
    trendline="ols",
    symbol="Research",
    title="Scatter matrix for <b>TOEFL Score</b> vs. <b>GRE Score</b> conditioning on <b>University Rating</b> and <b>Research</b> variables",
)

fig.show()
```

{% include graduate-admission-figures/27_toefl_gre_scatter_conditional_on_uni_rating_and_research.html %}

#### `TOEFL Score` vs. `CGPA`

```python
fig = px.scatter(
    data_frame=df,
    x="TOEFL Score",
    y="CGPA",
    color="Research",
    facet_col="University Rating",
    trendline="ols",
    symbol="Research",
    title="Scatter matrix for <b>TOEFL Score</b> vs. <b>CGPA</b> conditioning on <b>University Rating</b> and <b>Research</b> variables",
)

fig.show()
```

{% include graduate-admission-figures/28_toefl_cgpa_scatter_conditional_on_uni_rating_and_research.html %}

#### `GRE Score` vs. `CGPA`

```python
fig = px.scatter(
    data_frame=df,
    x="GRE Score",
    y="CGPA",
    color="Research",
    facet_col="University Rating",
    trendline="ols",
    symbol="Research",
    title="Scatter matrix for <b>GRE Score</b> vs. <b>CGPA</b> conditioning on <b>University Rating</b> and <b>Research</b> variables",
)

fig.show()
```

{% include graduate-admission-figures/29_gre_cgpa_scatter_conditional_on_uni_rating_and_research.html %}

As we can see, higher university ranking is linked with research experience, and they both affects student's scores and GPA.

We can also see that there are some _outlier_ points, where student goes to a low-ranking university and has no research experience, but have _good_ scores.

This way of studying how the relation between two variables changes as a function of other variables is very useful when we have a very large amount of data, and we want to study the relation between two _quanitative variables_, and how this relation changes with respect to other _categorical variables_, as we will have fewer points to investigate, rather than a single scatter plot with too many points.
