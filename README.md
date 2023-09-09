# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
    To look at bank lending data and determine the credit worthiness of borrowers

* Explain what financial information the data was on, and what you needed to predict.
    The data contained information of borrowers and loans. The size of the loan, the income of the borrower, number of loans the borrower has, total debt, debt to income ration, etc. 
    The goal of these predictions is to assist the financial institution determined the high-risk and healthy loans and if the loans should be approved or denied. 
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
    The variable that i was trying to predict was regarding "loan status". In this case high-risk or healthy-loan. 
* Describe the stages of the machine learning process you went through as part of this analysis.
    1. Data loading and exploration
      Importing libraries
      Reading in the dataset
      Exploring the dataset
    2. Preparing the data
      Seperated the data into labels
      check the balance of the target variable, in this case "loans status" 
    3. Train-test Split
      Split the data into training and testing sets using 'train_test_split()'
    4. Model Building(Before resampling)
      Instantiate a Logistic Regresion model and fit it to the training data
      Make predictions on the test data using the trained model
      Calculate and print the balanced accuracy score model for evaluation
      Print a classification report with precision, recall and F1-score for each class
    5. Class imbalance handling(Random Oversampling)
      Import "RandomOverSampler"
      instatntiate the random oversampler with a random state of 1
      Apply random oversampling to the training data to balance class distribution
    6. Model Rebuilding
      Instantiate a Logistic Regression model and fit it to the resampled training data
    7.  Model Evaluation(After resampling)
      Make predictions on the test data using the model trained on resampled data
      Calculate and print the balanced accuracy score for the resampled model
      Generate a confusion matrix to evaluate model performance
      Print classification report for the resampled model. 
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
    1. Logistic Regression
      is a classification algorithm used to model the probability of a binary outcome. In this analysis is was used before and after handling the class inbalance to predict healthy and high-risk loans
    2. Train-test Split
      The dataset was split into training and testing sets using "train_test_split()". This method is essential for evaluating the model's performance on unseen data
    3. Random Oversampling
      The RandomOversampler was used to address the class imbalance of the dataset. It randomly oversampled the minority dataset (high-risk loans) to balance class distributuion. This technique helps prevent the model from being biased toward the mayority class and potetially improves the model's predictive performance. 
## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
      Healthy Loans: Precision: 1.00 Recall: 0.99 F1-Score: 1.00. 
      This all indicates that the model performs really well identifying healthy loans. Precission is 1.00 which means it's almost always correct when predicting a loan as healthy. and the recall indicates that the model is capturing the majority of actual healthy loans

      High-risk Loans:  Precision: 0.85 Recall: 0.91 F1-Score: 0.88
      The model's perfomance on high-risk loans is also good but   not as good as with predicting the healthy loans. the precision at 0.85 indicates that when the model predicts a high-risk loan it is accurate 85% of the time. The recall of 0.91 indicates that the model identifies 91% of high-risk loans


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
      Healthy Loans: Precision: 1.00 Recall: 0.99 F1-Score: 1.00
        This all indicates that the model performs really well identifying healthy loans. Precission is 1.00 which means it's almost always correct when predicting a loan as healthy. and the recall indicates that the model is capturing the majority of actual healthy loans
      
      High-Risk Loans: Precision= 0.84 Recall= 0.99 F1-Score=0.91
        Precision is indicating that the data prediction is correct 84% of the time. is just 0.1% lower than before using the RandomOverSampler. Recall is at 99% which is much higher compared to the 91% from before using the RandomOverSampler. this also makes the F1-Score much high in comparison 
## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
    The model used after using the RandomOverSampler seems to work best. Although Precision is only 0.1% lower than when not using RandomOverSampler, the Recall is much higher which makes the F1-Score 91% for high-risk loans. Essentially the F1-Score for Healthy loans on both models is 100%. on the first model is 88% for the high-risk loans and it goes up to 91% for the second model.  
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
    In this case is more important to predict the `1` (high-risk loans) as these are the borrowers that are most likely to default on their loans. The second model which uses "RandomOverSampler" brings up the F1-Score for the high-risk loans from 88% to 91%. Although this isn't a significant increase, it is still an increase in the model's performance
If you do not recommend any of the models, please justify your reasoning.
