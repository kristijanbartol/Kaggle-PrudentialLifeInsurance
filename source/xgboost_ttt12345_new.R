library(readr)
library(xgboost)

# Set a random seed for reproducibility
set.seed(1)

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

cat("training a XGBoost classifier\n")
clf <- xgboost(data        = data.matrix(train[,feature.names]),
               label       = train$Response,
               eta         = 0.02,
               depth       = 20,
               nrounds     = 4111,
               objective   = "count:poisson",
               eval_metric = "rmse",colsample_bytree=0.66,min_child_weight=3,subsample=0.71)

cat("making predictions\n")
submission <- data.frame(Id=test$Id)
submission$Response <- as.integer(round(predict(clf, data.matrix(test[,feature.names]))))

# I pretended this was a regression problem and some predictions may be outside the range
submission[submission$Response<1, "Response"] <- 1
submission[submission$Response>8, "Response"] <- 8
submission[submission$Response==3,"Response"] <- 2

cat("saving the submission file\n")
write_csv(submission, "data/xgb_ttt12345_new_submission.csv")


