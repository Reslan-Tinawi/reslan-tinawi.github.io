---
title: Text Classification with Python (and some AI Explainability!)
layout: single
classes: wide
tags: [NLP, sklearn, NLTK]
---

This post will demonstarte the use of machine learning algorithms for the problem of *Text Classification* using [scikit-learn](https://scikit-learn.org/stable/) and [NLTK](https://www.nltk.org/) libraries. I will use the [20 Newsgroups data set](http://qwone.com/~jason/20Newsgroups/) as an example, and talk about the main stpes of developing a machine learning model, from loading the data in its raw form to evaluating the model's predictions, and finally I'll shed some light on the concept of *Explainable AI* and use [Lime](https://github.com/marcotcr/lime) library for explaining the model's predictions.

This post is not intended to be a step-by-step tutorial, rather, I'll address the main steps of developing classification models, and provide resources for digging deeper. The source code can be found [here](https://github.com/Reslan-Tinawi/20-newsgroups-Text-Classification)

# What is Text Classification?

In short, *Text Classification* is the taks of assigning a set of predifined tags or categories to text according to its content. There are two types of classification tasks:

- Binary Classification: in this type, there are **only** two classes to predict, like spam email classification.
- Multuclass Classification: in this type, the set of classes consists of `n` class (where `n` > 2), and the classifier try to predict one of these `n` classes, like News Articles Classification, where news articles are assigned classes like *politics*, *sport*, *tech*, etc ...

In this post, I'll focus on multiclass classification for classifying news articles, the following figure outlines the working of news articles classification:

<figure>
    <a href="/assets/images/text-classification-post-assets/news-articles-classification.jpg">
        <img src="/assets/images/text-classification-post-assets/news-articles-classification.jpg">
    </a>
    <figcaption><a href="https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/">Source</a></figcaption>
</figure>

# The dataset:

I will use the [20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/), quoting the official dataset website:

> The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.
The data is organized into 20 different newsgroups, each corresponding to a different topic. Some of the newsgroups are very closely related to each other (e.g. `comp.sys.ibm.pc.hardware` / `comp.sys.mac.hardware`), while others are highly unrelated (e.g `misc.forsale` / `soc.religion.christian`). Here is a list of the 20 newsgroups, partitioned (more or less) according to subject matter: 

<table style='font-family:"Courier New", Courier, monospace; font-size:80%'>
    <tr>
        <td>comp.graphics<br>comp.os.ms-windows.misc<br>comp.sys.ibm.pc.hardware<br>comp.sys.mac.hardware<br>comp.windows.x
        </td>
        <td>rec.autos<br>rec.motorcycles<br>rec.sport.baseball<br>rec.sport.hockey</td>
        <td>sci.crypt<br>sci.electronics<br>sci.med<br>sci.space</td>
    </tr>
    <tr>
        <td>misc.forsale</td>
        <td>talk.politics.misc<br>talk.politics.guns<br>talk.politics.mideast</td>
        <td>talk.religion.misc<br>alt.atheism<br>soc.religion.christian</td>
    </tr>
</table>

# A glance over the data:

The dataset consists of 18,846 samples divided into 20 classes.

Before jumping right away to the machine learning part (training and validating the model), it's always better perform some [Exploratory Data Analysis (EDA)](https://en.wikipedia.org/wiki/Exploratory_data_analysis), Wikipedia's definition of EDA:
> In statistics, exploratory data analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.

I won't go into detail about the EDA, but a good read about it is this nice article: [What is Exploratory Data Analysis?](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15)

Simply put, we can say that EDA is the set of methods that helps us to reveal characteristics of the data we're dealing with, in this post I'll only perform a few *visualization* on the data, and see if it needs any further cleaning.

*Note*: for creating the visualization I'll use two data visualization libraries:
- [seaborn](https://seaborn.pydata.org/): Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
- [Plotly](https://plotly.com/python/): Plotly's Python graphing library makes *interactive*, publication-quality graphs.

<!-- Check the appropriate way to quote -->

<!-- TODO: make sure the following charts are centered -->

## Categories Percentages:

In balanced data each class (label) has an (almost) equal number of instances, as opposed to imbalanced data in which the distribution across the classes is not equal, and a few classes have high percentage of the samples, while others have only a low percentage.

The following chart shows that our dataset is *balanced* because classes have a nearly equal number of instances.

{% include text-classification-post-charts/categories-percentages-pie-chart.html %}

## Average Article Length:

The following chart shows how the average article length varies by category, one might see that the ploitics articles are very lengthy (especially the middle east ones), compared to tech-related articles.

{% include text-classification-post-charts/average-article-length-bar-chart.html %}

We know that the classification will be based on the article content, and classifiers generally look for words that distinguishably describe the categories, and as observed in the previous chart, some categories (`mac_hardware`, `pc_hardware`, ...) are short on average which means they have only a *handful* set of words, these facts might later explain why the classifier get confused between categories.

## Wordclouds:

<!-- TODO: enhance the styling of the wordclouds -->

The previous two charts gave us only statistics about the data, not the actual content of the data, which is *words*.

Wordsclouds are useful for quickly perceiving the dominant words in data, they depict words in different sizes, the higher the word frequency the bigger its size in the visualization.

The following are 4 wordclouds for `grapichs`, `medicine`, `sport-hocky`, and `politics-middle-east` categories, generate using this library: [WordCloud for Python](https://github.com/amueller/word_cloud)

<figure>
    <a href="/assets/images/text-classification-post-assets/graphics-word-cloud.png">
        <img src="/assets/images/text-classification-post-assets/graphics-word-cloud.png">
    </a>
    <a href="/assets/images/text-classification-post-assets/medicine-word-cloud.png">
        <img src="/assets/images/text-classification-post-assets/medicine-word-cloud.png">
    </a>
    <a href="/assets/images/text-classification-post-assets/sport-hockey-word-cloud.png">
        <img src="/assets/images/text-classification-post-assets/sport-hockey-word-cloud.png">
    </a>
    <a href="/assets/images/text-classification-post-assets/politics-middle-east-word-cloud.png">
        <img src="/assets/images/text-classification-post-assets/politics-middle-east-word-cloud.png">
    </a>
</figure>

We can see that the dominant words in each category are considered descriptive words for the category, with some exceptions, like the word *one* which has a high frequency in these four categories, although it's not a much of a descriptive word, the words wich have a high frequency in the language (like: *the*, *this*, *we*, ... ) are know as the stopwords, and they aremoved from the data before training the model.

# Data splitting:

Split the data into 75% training and 25% testing, with stratified sampling, to make sure that the class labels percentages in both training and testing data are (nearly) equal.

```python
X = df['text']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
```

# Text vectorization:

*Note*: in this section and in the following one, I'll draw some ideas from this book (which I really recommend): [Applied Text Analysis with Python](http://shop.oreilly.com/product/0636920052555.do)

The fourth chapter discusses in detail the different vectorization techniques, with sample implementation.

Machine learning algorithms operate *only* on numerical input, expecting a two-dimensional array of size `n_samples`, `n_features` (where rows are samples, and columns are features)

Our current input is a list of *varied-length* documents, and in order to perform machine learning algorithms on textual data, we need to transform our documents into vector representation.

This process is known as *Text Vectorization* where documents are mapped into a numerical vector representation of the same size (the resulting vectors must be all of the same size, which is `n_feature`)

There are different methods of calculating the vector representation, mainly:
- Frequency Vectors.
- One-Hot Encoding.
- Term Frequency–Inverse Document Frequency.
- Distributed Representation.

Discussing the working of each method is beyond the purpose of this article, I'll use the TF-IDF vectorization method.

TF-IDF is a weighting technique, in which every term is assigned a value relative to its rareness, the more common the word is the less weight it'll be assigned, and rare terms will be assigned higher weights.

The general idea behind this techniques is that meaning of a document is encoded in the rare terms it has.

For example, in our corpus (a corpus is the set of documents), terms like `game`, `team`, `hockey`, and `player` will be *rare* across the corpus, but common in sport articles (articles tagged as `hockey`), while other terms like `one`, `think`, `get`, and `would` which occur frequently in our corpus, but they are less significant for classifing sport articles, and therfore should be assigned lower weights.

For more details and intuition behind the TF-IDF method, I recomment this article: [What is TF-IDF?](https://monkeylearn.com/blog/what-is-tf-idf/)