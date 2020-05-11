---
title: Text Classification with Python (and some AI Explainability!)
layout: single
classes: wide
tags:
  - NLP
  - sklearn
  - NLTK
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

> The data is organized into 20 different newsgroups, each corresponding to a different topic. Some of the newsgroups are very closely related to each other (e.g. `comp.sys.ibm.pc.hardware` / `comp.sys.mac.hardware`), while others are highly unrelated (e.g `misc.forsale` / `soc.religion.christian`). Here is a list of the 20 newsgroups, partitioned (more or less) according to subject matter: 

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

