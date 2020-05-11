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

In this post, I'll focus on multiclass classification for classifying news articles.

# The dataset:

