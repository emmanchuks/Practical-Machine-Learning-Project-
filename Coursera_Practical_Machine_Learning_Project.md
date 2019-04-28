---
title: "Coursera Practical Machine Learning Project"
author: "Ezeonyebuchi E. C."
date: "27/04/2019"
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
This is the âclasseâ variable in the training set. I will useany of the other variables to predict with.

Load needed libraries

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
Downloading the websites
1).
For the traing data

```r
TrainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
##download.file(TrainURL, destfile = "./pml-training", method="auto")
```

2).For test data

```r
TestUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
##download.file(TestUrl, destfile = "./pml-testing.csv", method = "auto")
```
Reading, Cleaning and Exploring the data

```r
trainData <- read.csv("./pml-training", header = TRUE)
#head(trainData)
dim(trainData)
```

```
## [1] 19622   160
```

```r
#str(trainData)
#summary(trainData)

testData <- read.csv("./pml-testing.csv", header = TRUE)
#head(testData)
dim(testData)
```

```
## [1]  20 160
```

```r
#str(testData)
#summary(testData)
```
Partioning the training set into two
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
Removing variables with Nearly Zero Variance

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

![](Coursera_Practical_Machine_Learning_Project_files/figure-html/unnamed-chunk-9-1.png)<!-- -->
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
##         OOB estimate of  error rate: 0.23%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    0    0    0    1 0.0002560164
## B    4 2651    3    0    0 0.0026335591
## C    0    4 2391    1    0 0.0020868114
## D    0    0   10 2241    1 0.0048845471
## E    0    1    0    6 2518 0.0027722772
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
##          A 1674    3    0    0    0
##          B    0 1134    1    0    0
##          C    0    1 1025    2    0
##          D    0    1    0  962    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9986          
##                  95% CI : (0.9973, 0.9994)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9983          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9956   0.9990   0.9979   1.0000
## Specificity            0.9993   0.9998   0.9994   0.9998   1.0000
## Pos Pred Value         0.9982   0.9991   0.9971   0.9990   1.0000
## Neg Pred Value         1.0000   0.9989   0.9998   0.9996   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1927   0.1742   0.1635   0.1839
## Detection Prevalence   0.2850   0.1929   0.1747   0.1636   0.1839
## Balanced Accuracy      0.9996   0.9977   0.9992   0.9989   1.0000
```

```r
plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(confMatRandForest$overall['Accuracy'], 4)))
```

![](Coursera_Practical_Machine_Learning_Project_files/figure-html/unnamed-chunk-11-1.png)<!-- -->
(b) Method: Decision Trees
model fit

```r
set.seed(4000)
modFitDT <- rpart(classe ~ ., data=Trainset, method="class")
fancyRpartPlot(modFitDT)
```

![](Coursera_Practical_Machine_Learning_Project_files/figure-html/unnamed-chunk-12-1.png)<!-- -->
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
##          A 1493  168   40   67   48
##          B   54  709   62   91  130
##          C   14   88  831  136   96
##          D   91  137   70  612  112
##          E   22   37   23   58  696
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7376          
##                  95% CI : (0.7262, 0.7488)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6674          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8919   0.6225   0.8099   0.6349   0.6433
## Specificity            0.9233   0.9290   0.9313   0.9167   0.9709
## Pos Pred Value         0.8221   0.6778   0.7133   0.5988   0.8325
## Neg Pred Value         0.9555   0.9111   0.9587   0.9276   0.9235
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2537   0.1205   0.1412   0.1040   0.1183
## Detection Prevalence   0.3086   0.1777   0.1980   0.1737   0.1421
## Balanced Accuracy      0.9076   0.7757   0.8706   0.7758   0.8071
```

```r
plot(confMatDT$table, col = confMatDT$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDT$overall['Accuracy'], 4)))
```

![](Coursera_Practical_Machine_Learning_Project_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

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
## There were 53 predictors of which 53 had non-zero influence.
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
##          A 1664    4    0    0    0
##          B    8 1121    1    1    5
##          C    0   12 1018    7    2
##          D    2    2    6  954   17
##          E    0    0    1    2 1058
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9881         
##                  95% CI : (0.985, 0.9907)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.985          
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9940   0.9842   0.9922   0.9896   0.9778
## Specificity            0.9991   0.9968   0.9957   0.9945   0.9994
## Pos Pred Value         0.9976   0.9868   0.9798   0.9725   0.9972
## Neg Pred Value         0.9976   0.9962   0.9983   0.9980   0.9950
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2828   0.1905   0.1730   0.1621   0.1798
## Detection Prevalence   0.2834   0.1930   0.1766   0.1667   0.1803
## Balanced Accuracy      0.9965   0.9905   0.9939   0.9921   0.9886
```

```r
plot(confMat_GBM$table, col = confMat_GBM$byClass, 
     main = paste("GBM - Accuracy =", round(confMat_GBM$overall['Accuracy'], 4)))
```

![](Coursera_Practical_Machine_Learning_Project_files/figure-html/unnamed-chunk-15-1.png)<!-- -->
The accuracy of the 3 regression modeling methods above are:

```r
RandomForest <- print(paste("Random Forest - Accuracy =",
                            round(confMatRandForest$overall['Accuracy'], 4)))
```

```
## [1] "Random Forest - Accuracy = 0.9986"
```

```r
DecisionTree <- print(paste("Decision Tree - Accuracy =",
                            round(confMatDT$overall['Accuracy'], 4)))
```

```
## [1] "Decision Tree - Accuracy = 0.7376"
```

```r
GBM <- print(paste("GBM - Accuracy =", round(confMat_GBM$overall['Accuracy'], 4)))
```

```
## [1] "GBM - Accuracy = 0.9881"
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
