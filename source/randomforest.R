library(readr)
#library(xgboost)
library(randomForest)
library(miscTools)
library(ggplot2)

# Set a random seed for reproducibility
set.seed(8888)

cat("reading the train and test data\n")
train <- read_csv("data/train.csv")
test  <- read_csv("data/test.csv")

feature.names <- names(train)[2:ncol(train)-1]

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("replacing missing values with -1\n")
train[is.na(train)] <- -1
test[is.na(test)]   <- -1


dim(train)
dim(test)

train[1:5,c(1:2, 126:128)]




#cat("training a Random Forest classifier\n")
#clf <- randomForest(train[,feature.names], factor(train$Response), ntree=1000, sampsize=5000, nodesize=2)

cat("training a Random Forest regressor\n")
clf <- randomForest(Response ~ ., data=train[,c(2:128)], ntree=200, sampsize=5000, nodesize=2)


cat("making predictions\n")
submission <- data.frame(Id=test$Id)
#submission$Response <- as.integer(round(predict(clf, data.matrix(test[,feature.names]))))

#submission$Response <- as.integer(predict(clf, data.matrix(test[,feature.names])))
#submission$Response <- predict(clf, data.matrix(test[,feature.names]))

submission$Response <-predict(clf, test)


submission[1:5,]
summary(submission)

# I pretended this was a regression problem and some predictions may be outside the range
#submission[submission$Response<1, "Response"] <- 1
#submission[submission$Response>8, "Response"] <- 8

# Zaokruživanje #1
#submission[,2] = round(submission[,2])

# Zaokruživanje #2
for (i in 1:length(submission$Response)) {
  if(floor(submission$Response[i]) == 1) submission$Response[i] = 1
  else submission$Response[i] = round(submission$Response[i])
}
hist(submission$Response)     # --> loši rezultati nakon zaokruživanja

cat("saving the submission file\n")
write_csv(submission, "data/rf_regressor_raw_v1_submission.csv")
