---
title: "**densitystackr**: Estimating weighted density ensembles"
author: |
   | Evan L Ray^1,2^ and Nicholas G Reich^1^
   |
   | 1. University of Massachusetts-Amherst
   | 2. Mount Holyoke College
output: html_document
references:
- id: ray2017
  title: Prediction of Infectious Disease Epidemics via Weighted Density Ensembles
  author:
  - family: Ray
    given: Evan L
  - family: Reich
    given: Nicholas G
  container-title: GitHub (preprint)
  URL: 'https://github.com/reichlab/adaptively-weighted-ensemble/blob/master/inst/manuscript/awes-manuscript.pdf'
  type: article-journal
  issued:
    year: 2017
nocite: | 
  @ray2017
---

**Keywords**: Ensemble methods, predictive analytics, biostatistics, epidemiology

**Webpages**: https://github.com/reichlab/densitystackr, http://reichlab.io/flusight/

Ensemble methods are commonly used in predictive analytics to combine predictions from multiple distinct models. A large portion of the literature on ensemble methods focuses on classification and regression problems, with less work devoted to creating ensembles from full predictive densities. One common approach to creating ensembles is to estimate the component models separately, obtain measures of their performance on held-out data, and then estimate a higher-level model that combines predictions from the component models using their performance on the held out data. This is sometimes referred to as "stacking". In the context of density estimation, one fairly straight-forward way to combine the component models is to create a weighted average of them, where the weights are restricted to be non-negative and sum to 1. Using gradient boosting, we have developed and implemented, in the *R* package **densitystackr**, a method to estimate these model weights, potentially as a function of features such as observed data or the predictive densities themselves. Additionally, the method provides flexible options for regularization of the weight functions to prevent overfitting. We illustrate the application of this method with a weighted density ensemble for three component models for predicting measures of influenza season timing and severity in the United States, both at the national and regional levels. In an out-of-sample test phase of prediction, the ensemble methods showed overall performance that was similar to the best of the component models, but offered more consistent performance across seasons. The **densitystackr** package is currently available on GitHub, with additional features under active development, including the incorporation of additional loss functions.

# References
