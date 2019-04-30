---
title: "Coursera Practical Machine Learning Project"
author: "Ezeonyebuchi E. C."
date: "30/04/2019"
output:
  html_document:
    keep_md: true
    toc: true
---


Introduction:
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount 
of data about personal activity relatively inexpensively. These type of devices are part of the quantified 
self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health,
to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, I will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participant 
who were asked to perform dumbbell lifts correctly and incorrectly in 5 different ways.
The data consists of a Training data and a Test data (to be used to validate the selected model).
The goal of your project is to predict the manner in which they did the exercise. 
This is the Ã¢ÂÂclasseÃ¢ÂÂ variable in the training set. I will useany of the other variables to predict with.

EXECUTIVE SUMMARY OF THE DATA PROCESSING

1. Set up the working directory.

2. Download the datasets

3.Load the training and testing data, and perform elementary data analysis.

4.Preprocess the data, and extract the useful feature. For this purpose, 
select the columns which have most of the entry are NA and blank, and filter out 
the training and test data set and build the validate data set that use to train the model.

5.For implementing Machine Learning algorithm load the “caret” package.

6.Partition the tidy training data using “createDataPartition()” function and perform basic operations.

7.Now find out the less useful or useless predictor from the tidy training data set, and update the training data set.

8.Fit a model on the training data set i.e apply “train()” function where method is random forest algorithm (mehtod = “rf”). In order to speed up the execution trControl parameter of the “train” function is used.

9. Predict the classe of each instance of the reshaped test data set by using “prediction” function of the caret package.

10.Estimate out of sample error appropriately with cross-validation

1. Set up the working directory.

Load required libraries for the data processing

```r
library(knitr)
```

```
## Warning: package 'knitr' was built under R version 3.5.3
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.5.3
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.2.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.5.3
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```r
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.5.3
```

```
## Loading required package: rpart
```

```
## Warning: package 'rpart' was built under R version 3.5.3
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.5.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.5.3
```

```
## 
## Attaching package: 'ggplot2'
```

```
## The following object is masked from 'package:randomForest':
## 
##     margin
```

```r
library(rpart)
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 3.5.3
```

```
## Loaded gbm 2.1.5
```

```r
library(RColorBrewer)
library(corrplot)
```

```
## Warning: package 'corrplot' was built under R version 3.5.3
```

```
## corrplot 0.84 loaded
```

```r
library(ggplot2)
library(colorspace)
```

```
## Warning: package 'colorspace' was built under R version 3.5.3
```
Download the datasets
a]For the traing data

```r
TrainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
##download.file(TrainURL, destfile = "./pml-training", method="auto")
```
2] For test data

```r
TestUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
##download.file(TestUrl, destfile = "./pml-testing.csv", method = "auto")
```
Load the training and testing data, and perform elementary data analysis.

```r
trainData <- read.csv("./pml-training", header = TRUE)
dim(trainData)
```

```
## [1] 19622   160
```

```r
#str(trainData)

testData <- read.csv("./pml-testing.csv", header = TRUE)
dim(testData)
```

```
## [1]  20 160
```

```r
#str(testData)
```
Partition the training data using “createDataPartition()” function and perform basic operations
Partioning Training data set into two data sets.I am using 70% for Training and 30% for Testing:

```r
inTrain <- createDataPartition(y = trainData$classe, p=0.7, list=FALSE)
Trainset <- trainData[inTrain, ]
Testset <- trainData[-inTrain, ]
dim(Trainset)
```

```
## [1] 13737   160
```

```r
dim(Testset)
```

```
## [1] 5885  160
```
Preprocess the data, and extract the useful feature. For this purpose, 
select the columns which have most of the entry are NA and blank, and filter out 
the training and test data set and build the validate data set that use to train the model.

```r
NZV <- nearZeroVar(Trainset)
Trainset <- Trainset[, -NZV]
Testset  <- Testset[, -NZV]
dim(Trainset)
```

```
## [1] 13737   104
```

```r
dim(Testset)
```

```
## [1] 5885  104
```
Remove variables that are mostly NA (mNA)

```r
mNA <- sapply(Trainset, function(x) mean(is.na(x))) > 0.95
Trainset <- Trainset[, mNA==FALSE]
Testset  <- Testset[, mNA==FALSE]
dim(Trainset)
```

```
## [1] 13737    59
```

```r
dim(Testset)
```

```
## [1] 5885   59
```
Remove identification only variables (columns 1 to 5)

```r
Trainset <- Trainset[, -(1:5)]
Testset  <- Testset[, -(1:5)]
dim(Trainset)
```

```
## [1] 13737    54
```

```r
dim(Testset)
```

```
## [1] 5885   54
```
The number of variables for analysis has been reduced to 54 with the above cleaning exercise 
Correlation Analysis

```r
corMatrix <- cor(Trainset[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

![](Coursera_Practical_Machine_Learning_Project_files/figure-html/unnamed-chunk-13-1.png)<!-- -->
Prediction Models Building
Three methods will be applied to the regression models. 
A Confusion Matrix is plotted at the end of each for better visualization of the  the accuracy of the models.

(a)Random Forest
model fit

```r
controlRF <- trainControl(method="cv", number=3, verbose=FALSE)
modFitRandForest <- train(classe ~ ., data=Trainset, method="rf",
                          trControl=controlRF)
modFitRandForest$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.15%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3906    0    0    0    0 0.000000000
## B    4 2652    2    0    0 0.002257336
## C    0    4 2392    0    0 0.001669449
## D    0    0    7 2245    0 0.003108348
## E    0    0    0    4 2521 0.001584158
```
Prediction on Test Dataset

```r
predictRandForest <- predict(modFitRandForest, newdata=Testset)
confMatRandForest <- confusionMatrix(predictRandForest, Testset$classe)
confMatRandForest
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    3    0    0    0
##          B    1 1134    1    0    0
##          C    0    1 1025    1    0
##          D    0    1    0  962    2
##          E    1    0    0    1 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.998           
##                  95% CI : (0.9964, 0.9989)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9974          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9956   0.9990   0.9979   0.9982
## Specificity            0.9993   0.9996   0.9996   0.9994   0.9996
## Pos Pred Value         0.9982   0.9982   0.9981   0.9969   0.9982
## Neg Pred Value         0.9995   0.9989   0.9998   0.9996   0.9996
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1927   0.1742   0.1635   0.1835
## Detection Prevalence   0.2846   0.1930   0.1745   0.1640   0.1839
## Balanced Accuracy      0.9990   0.9976   0.9993   0.9987   0.9989
```

```r
plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(confMatRandForest$overall['Accuracy'], 4)))
```

![](Coursera_Practical_Machine_Learning_Project_files/figure-html/unnamed-chunk-15-1.png)<!-- -->
(b) Method: Decision Trees
model fit

```r
set.seed(4000)
modFitDT <- rpart(classe ~ ., data=Trainset, method="class")
fancyRpartPlot(modFitDT)
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![](Coursera_Practical_Machine_Learning_Project_files/figure-html/unnamed-chunk-16-1.png)<!-- -->
Prediction on Test dataset

```r
predictDT <- predict(modFitDT, newdata=Testset, type="class")
confMatDT <- confusionMatrix(predictDT, Testset$classe)
confMatDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1459   92    2   16    7
##          B  117  843   68   78   39
##          C    0   63  853   41    3
##          D   85   70   94  744   71
##          E   13   71    9   85  962
## 
## Overall Statistics
##                                           
##                Accuracy : 0.826           
##                  95% CI : (0.8161, 0.8356)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7804          
##                                           
##  Mcnemar's Test P-Value : 8.747e-15       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8716   0.7401   0.8314   0.7718   0.8891
## Specificity            0.9722   0.9364   0.9780   0.9350   0.9629
## Pos Pred Value         0.9258   0.7362   0.8885   0.6992   0.8439
## Neg Pred Value         0.9501   0.9376   0.9649   0.9544   0.9747
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2479   0.1432   0.1449   0.1264   0.1635
## Detection Prevalence   0.2678   0.1946   0.1631   0.1808   0.1937
## Balanced Accuracy      0.9219   0.8382   0.9047   0.8534   0.9260
```

```r
plot(confMatDT$table, col = confMatDT$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDT$overall['Accuracy'], 4)))
```

![](Coursera_Practical_Machine_Learning_Project_files/figure-html/unnamed-chunk-17-1.png)<!-- -->

(c)Generalized Boosted Model
model fit

```r
set.seed(4000)
control_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFit_GBM  <- train(classe ~ ., data=Trainset, method = "gbm",
                     trControl = control_GBM, verbose = FALSE)
modFit_GBM$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 52 had non-zero influence.
```
Prediction on Test dataset

```r
predict_GBM <- predict(modFit_GBM, newdata=Testset)
confMat_GBM <- confusionMatrix(predict_GBM, Testset$classe)
confMat_GBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1667   14    0    0    0
##          B    7 1117   11    0    3
##          C    0    7 1010   13    3
##          D    0    1    3  950    8
##          E    0    0    2    1 1068
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9876          
##                  95% CI : (0.9844, 0.9903)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9843          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9958   0.9807   0.9844   0.9855   0.9871
## Specificity            0.9967   0.9956   0.9953   0.9976   0.9994
## Pos Pred Value         0.9917   0.9815   0.9777   0.9875   0.9972
## Neg Pred Value         0.9983   0.9954   0.9967   0.9972   0.9971
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2833   0.1898   0.1716   0.1614   0.1815
## Detection Prevalence   0.2856   0.1934   0.1755   0.1635   0.1820
## Balanced Accuracy      0.9962   0.9881   0.9898   0.9915   0.9932
```

```r
plot(confMat_GBM$table, col = confMat_GBM$byClass, 
     main = paste("GBM - Accuracy =", round(confMat_GBM$overall['Accuracy'], 4)))
```

![](Coursera_Practical_Machine_Learning_Project_files/figure-html/unnamed-chunk-19-1.png)<!-- -->
The accuracy of the 3 regression modeling methods above are:

```r
RandomForest <- print(paste("Random Forest - Accuracy =",
                            round(confMatRandForest$overall['Accuracy'], 4)))
```

```
## [1] "Random Forest - Accuracy = 0.998"
```

```r
DecisionTree <- print(paste("Decision Tree - Accuracy =",
                            round(confMatDT$overall['Accuracy'], 4)))
```

```
## [1] "Decision Tree - Accuracy = 0.826"
```

```r
GBM <- print(paste("GBM - Accuracy =", round(confMat_GBM$overall['Accuracy'], 4)))
```

```
## [1] "GBM - Accuracy = 0.9876"
```
From the above observation, RandomForest model gives a better result and will be applied to predict the 20 quiz results 
(testing (testData) dataset) as shown below.

```r
predictTEST <- predict(modFitRandForest, newdata=testData)
predictTEST
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
summary(predictTEST)
```

```
## A B C D E 
## 7 8 1 1 3
```

create a character vector of the predictions and check the length of the vector

```r
predictTEST <- c(as.character(predictTEST))

#Length of the predicted vector
length(predictTEST)
```

```
## [1] 20
```
Estimate out of sample error appropriately with cross-validation
Out of sample error

```r
dim(Testset)
```

```
## [1] 5885   54
```
True accuracy of the predicted model

```r
outOfSampleError.accuracy <- sum(predictTEST == Testset$classe)/length(predictTEST)
```

```
## Warning in `==.default`(predictTEST, Testset$classe): longer object length
## is not a multiple of shorter object length
```

```
## Warning in is.na(e1) | is.na(e2): longer object length is not a multiple of
## shorter object length
```

```r
outOfSampleError.accuracy
```

```
## [1] 65.2
```
Out of sample error and percentage of out of sample error

```r
outOfSampleError <- 1 - outOfSampleError.accuracy
outOfSampleError
```

```
## [1] -64.2
```

```r
e <- outOfSampleError * 100
paste0("Out of sample error estimation: ", round(e, digits = 2), "%")
```

```
## [1] "Out of sample error estimation: -6420%"
```
