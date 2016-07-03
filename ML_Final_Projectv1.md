# Practical Machine Learning Final Report
Bruce Paterson  
June 2016  



##### **Introduction**  
6 Participants were asked to lift dumbell weights in 5 different ways.
Data was recorded from accelerometers on the belt, forearm, arm, and dumbbell of the 6 participants.  The outcome variable labelled "classe" (in the training dataset) is recorded into A, B, C, D, and E categories.  The goal of this project is to predict the correctly how the 6 participants did the exercise.  This report describes how the model for the project was built.

##### **Executive Summary**  
The analysis resulted in a 9 variable model consisting of the following variables/features:  

1. num_window
2. roll_forearm
3. magnet_dumbbell_y
4. magnet_dumbbell_z
5. roll_belt
6. magnet_dumbbell_x
7. pitch_belt
8. pitch_forearm
9. accel_dumbbell_y  

The 9 variables were from a starting 159 in the original training dataset to 53 variables after basic cleaning of the training datatset.  The final model was developed using random forest approach and resulted in a 99.75% accuracy rate when applied to make predictions on the test/cross validation dataset or an out-of-sample error rate in prediction of 0.25%.  The model when applied to the "true" test dataset sample of 20 samples gave 100% accurate prediction of the classe outcome (A, B, C, D, E) of the dumbell exercise performed.  

The PCA methodology was applied to the training dataset and subsequent model fit of 12 principal components predicted on the test dataset an accuracy of 95% or 5% out-of-sample error rate.  

There were a number of runtime issues encountered when using the "train" function applied to the random forest methodology.  This is discussed in greater detail below.  

##### **Data Description**  
The training dataset comes from this link <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>.  The testing dataset comes from this link <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>.  For a more complete description of the data and the project comes from this source: <http://groupware.les.inf.puc-rio.br/har>.  

##### **Data Importing and Cleaning**  
Reading in the datasets both training and testing - the dimensions of the training and testing data is as follows:.

```r
#Read in data
setwd("~/D Driive/BPCL Client Work/Coursera/John Hopkins/Course 8/Week4/Project") #or set to actual working directory where file may have been already downloaded to
dataset = read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
dim(dataset)
```

```
## [1] 19622   160
```

```r
datatest = read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))
dim(datatest)
```

```
## [1]  20 160
```
A summary of the training data has the following characteristics - number of outcomes in each class, proportion of outcomes in each class and the proportion of outcomes in each class by participant:

```r
rbind(round(table(dataset$classe),2), round(prop.table(table(dataset$classe)),2))
```

```
##            A       B       C       D       E
## [1,] 5580.00 3797.00 3422.00 3216.00 3607.00
## [2,]    0.28    0.19    0.17    0.16    0.18
```


```r
round(prop.table(table(dataset$user_name, dataset$classe), 1),2)
```

```
##           
##               A    B    C    D    E
##   adelmo   0.30 0.20 0.19 0.13 0.18
##   carlitos 0.27 0.22 0.16 0.16 0.20
##   charles  0.25 0.21 0.15 0.18 0.20
##   eurico   0.28 0.19 0.16 0.19 0.18
##   jeremy   0.35 0.14 0.19 0.15 0.17
##   pedro    0.25 0.19 0.19 0.18 0.19
```
Nothing seems unbalanced with respect to the class/outcomes training data per se.

```r
#Clean data
library(dplyr)
library(caret)
#First remove first 6 columns not needed or necessary for any analysis - do same for test set as well
dataset = dataset[,-c(1:6)]
datatest = datatest[,-c(1:6)]
#Remove columns where no variation in data setting NA's equal to zero by using nearZeroVar function for both training
#and testing datasets
dataset[is.na(dataset)] = 0
index = 1:(dim(dataset)[2]-1)
nsv = nearZeroVar(dataset[, -154],saveMetrics=TRUE)
nsv = mutate(nsv, colnum = index)
index = nsv[nsv[,4]==FALSE,5]
dataset = dataset[, c(index, 154)]
dim(dataset)
```

```
## [1] 19622    54
```

```r
datatest = datatest[, c(index)]
dim(datatest)
```

```
## [1] 20 53
```
After removing the first 6 columns of training dataset and getting rid of "NA"" data and also variables/columns where there is almost no variation in the features/columns.  We are left with 54 columns - 1 for the outcome ("classe") and 53 variables/predictors/columns.  Doing the same to the testing data we have 53 features (we are predicting the outcomes for the testing data - objective of the project/model building exercise).  

The training dataset is split into train and cross validation datasets as follows - the training data has 60% and the cross validation ("testing") the remaing 40% of the original 19,622 rows of original training dataset (we need to test the predictions of the model fitted to any training data):

```r
#Split training data provided into training and testing/cross validation datasets
library(caret)
inTrain <- createDataPartition(y=dataset$classe, p=0.60, list=FALSE)
training <- dataset[inTrain,]
testing <- dataset[-inTrain,]
dim(training)
```

```
## [1] 11776    54
```

```r
dim(testing)
```

```
## [1] 7846   54
```
##### **Preliminary Model Analysis**  
Fitting Random Forest model to the 53 features of the training data yields the top 10 variables in order of importance (based on Accuracy):

```r
require(caret)
library(randomForest)
set.seed(3433)
#Model Fit Random Forest 
fit = randomForest(classe ~ ., data = training, importance=TRUE, ntree=100)
#Top 10 variables sorted on Accuracy (decreasing)
round(head(fit$importance[order(-fit$importance[,6]),6],10),3)
```

```
##        num_window         roll_belt magnet_dumbbell_y magnet_dumbbell_z 
##             0.146             0.115             0.111             0.109 
##      roll_forearm          yaw_belt magnet_dumbbell_x     pitch_forearm 
##             0.097             0.097             0.090             0.078 
##        pitch_belt     roll_dumbbell 
##             0.077             0.061
```
The accuracy and the summary statistics applying the fitted random forest model to make predictions on the test/cross validation data is as follows:

```r
require(caret)
require(randomForest)
testPC = predict(fit, testing) 
confusionMatrix(testing$classe,testPC)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    5 1512    1    0    0
##          C    0    4 1362    2    0
##          D    0    0    5 1279    2
##          E    0    0    0    1 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9975          
##                  95% CI : (0.9961, 0.9984)
##     No Information Rate : 0.2851          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9968          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9974   0.9956   0.9977   0.9986
## Specificity            1.0000   0.9991   0.9991   0.9989   0.9998
## Pos Pred Value         1.0000   0.9960   0.9956   0.9946   0.9993
## Neg Pred Value         0.9991   0.9994   0.9991   0.9995   0.9997
## Prevalence             0.2851   0.1932   0.1744   0.1634   0.1839
## Detection Rate         0.2845   0.1927   0.1736   0.1630   0.1837
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9989   0.9982   0.9973   0.9983   0.9992
```
The accuracy is 0.998 and the out-of-sample error rate is 16/7846 = 1- 0.998 = 0.002 or 0.20%.  This seems to be quite an accurate/good prediction when applied to the test data. 

##### **Final Model Selection**  
If we now try to fit the 10 variable model identified with the "importance" variables above and see if that is acceptable using the random forest model again:

```r
require(caret)
require(randomForest)
set.seed(3433)
fit2 = randomForest(classe ~ num_window +	roll_forearm +	magnet_dumbbell_y + magnet_dumbbell_z + yaw_belt +
                  roll_belt + magnet_dumbbell_x +	pitch_belt +	pitch_forearm +	accel_dumbbell_y,
                  data = training, importance=TRUE, ntree=100)

##The ranked variables in order of importance based on accuracy:
round(head(fit2$importance[order(-fit2$importance[,6]),6],10),3)
```

```
##        num_window         roll_belt          yaw_belt magnet_dumbbell_y 
##             0.322             0.208             0.200             0.177 
## magnet_dumbbell_z      roll_forearm magnet_dumbbell_x        pitch_belt 
##             0.174             0.164             0.161             0.160 
##     pitch_forearm  accel_dumbbell_y 
##             0.131             0.117
```

```r
#The predictions using the test data:
predictions <- predict(fit2, newdata=testing)
confusionMat <- confusionMatrix(predictions, testing$classe)
confusionMat
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    1    0    0    0
##          B    0 1517    0    0    2
##          C    0    0 1368    2    2
##          D    0    0    0 1282    3
##          E    0    0    0    2 1435
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9985          
##                  95% CI : (0.9973, 0.9992)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9981          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9993   1.0000   0.9969   0.9951
## Specificity            0.9998   0.9997   0.9994   0.9995   0.9997
## Pos Pred Value         0.9996   0.9987   0.9971   0.9977   0.9986
## Neg Pred Value         1.0000   0.9998   1.0000   0.9994   0.9989
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1933   0.1744   0.1634   0.1829
## Detection Prevalence   0.2846   0.1936   0.1749   0.1638   0.1832
## Balanced Accuracy      0.9999   0.9995   0.9997   0.9982   0.9974
```
We see the accuracy using 10 variables is an improvement over the first model using 53 variables - 0.9982% accuracy and out-of-sample error rate of 0.18%.  

Can we improve on the 10 variables and reduce this further?  
We check the correlations between the 10 variables and identify any pairs that have a correlation > than 80%:


```r
require(caret)
require(randomForest)
M = cor(training[,c("num_window",	"roll_forearm",	"magnet_dumbbell_y",	"magnet_dumbbell_z",	"yaw_belt",
                    "roll_belt",	"magnet_dumbbell_x",	"pitch_belt",	"pitch_forearm",	"accel_dumbbell_y")])
diag(M) = 0
which(abs(M)>0.80, arr.ind=TRUE)
```

```
##           row col
## roll_belt   6   5
## yaw_belt    5   6
```

```r
max(abs(M))
```

```
## [1] 0.8144956
```
We see that the "roll_belt"" and "yaw_belt"" have a correlation of 81% and also based on the order of importance that the "roll_belt" is more important - so we will drop the "yaw_belt" and fit a 3rd model using the Random Forest method. 

```r
set.seed(3433)
fit3 = randomForest(classe ~ num_window +	roll_forearm +	magnet_dumbbell_y + magnet_dumbbell_z + 
                  roll_belt + magnet_dumbbell_x +	pitch_belt +	pitch_forearm +	accel_dumbbell_y,
                  data = training, importance=TRUE, ntree=100)
#The predictions using the test data:
predictions <- predict(fit3, newdata=testing)
confusionMat <- confusionMatrix(predictions, testing$classe)
confusionMat
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    2    0    0    0
##          B    0 1515    0    0    3
##          C    0    1 1368    2    0
##          D    0    0    0 1283    5
##          E    0    0    0    1 1434
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9982        
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.2845        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9977        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9980   1.0000   0.9977   0.9945
## Specificity            0.9996   0.9995   0.9995   0.9992   0.9998
## Pos Pred Value         0.9991   0.9980   0.9978   0.9961   0.9993
## Neg Pred Value         1.0000   0.9995   1.0000   0.9995   0.9988
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1931   0.1744   0.1635   0.1828
## Detection Prevalence   0.2847   0.1935   0.1747   0.1642   0.1829
## Balanced Accuracy      0.9998   0.9988   0.9998   0.9985   0.9971
```
You see the accuracy reduces ever so slightly 99.82% to 99.75% by dropping the number of variables to 9 from 10 by excluding the "yaw_belt".   The out-of-sample error rate is now 0.25%.  

##### **Course Prediction Final Test Set Classes**  
Using the last model fitted and tested directly above (9 variable "fit3" model) - we apply this model to make predictions of the 20 sample test data yielding the following predictions:

```r
require(caret)
require(randomForest)
predictions <- predict(fit3, newdata=datatest)
datatest$classe <- predictions
test_item = 1:20
data.frame(test_item, predictions)
```

```
##    test_item predictions
## 1          1           B
## 2          2           A
## 3          3           B
## 4          4           A
## 5          5           A
## 6          6           E
## 7          7           D
## 8          8           B
## 9          9           A
## 10        10           A
## 11        11           B
## 12        12           C
## 13        13           B
## 14        14           A
## 15        15           E
## 16        16           E
## 17        17           A
## 18        18           B
## 19        19           B
## 20        20           B
```
The predictions turned out to be 100% accurate against the final 20 sample test data.

##### **PCA Models and Train Function Issues**  
The PCA method was applied to the original 54 feature training dataset and subsequently the predictions on the testing dataset were made with the resulting PCA model fitted.  The PCA identifies 12 principal components explaining 95% of the variation in the training dataset.  The predictions of the model resulted in obtaining an accuracy of about 95% on the test dataset.  So this was abandoned and random forest models as explained above were attempted.  

The "train" function when applied to the random forest models took an enormous amount of runtime - in many cases 20 to 30 minutes.  The course discussion board had some suggestions on using some "new" packages which apparently substantially improved runtimes for the "train" function.  As I was already quite far down the road in completing this analysis - I did not re-run this analysis using the "train" function and the "new" packages suggested but continued to use the actual randomForest packages and models directly without the "train" function.  Primarily because the runtimes were much more tolerable in the order of a few minutes or less.  


