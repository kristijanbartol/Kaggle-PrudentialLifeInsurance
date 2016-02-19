library(readr)
library(caret) 
library(doMC)
library(plyr)   # Mora se ručno setirati (linija #44 (fit))
                # StackOverflow -> http://stackoverflow.com/questions/7258639/problem-loading-the-plyr-package
registerDoMC(cores = 3)

set.seed(1729) 

train = read_csv("data/train.csv")
test = read_csv("data/test.csv")
#sample_submission = read_csv("../input/sample_submission.csv")
#summary(train)
#summary(test)

feature.names <- names(train)[2:ncol(train)-1]
cat("Factor char fields\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

train$Response <- as.factor(train$Response)

feature.names1 <- c("BMI", "Medical_History_15", "Medical_History_4", "Product_Info_4", "Medical_Keyword_15", "Medical_History_23", "Medical_Keyword_3", "Wt", "Ins_Age", "Medical_History_40", "Medical_History_24", "Family_Hist_4", "InsuredInfo_6", "Medical_History_30", "Medical_History_28", "Family_Hist_2", "Family_Hist_3", "Medical_History_32", "Medical_History_5", "Medical_History_13", "InsuredInfo_5", "Employment_Info_2", "Employment_Info_3", "Family_Hist_1", "Medical_History_39", "Medical_History_18", "Medical_Keyword_38", "Medical_Keyword_23", "Medical_History_20", "InsuredInfo_7", "Insurance_History_2", "Medical_History_33", "Medical_History_1", "Product_Info_1", "Family_Hist_5", "Employment_Info_1", "InsuredInfo_2", "Medical_History_16", "Employment_Info_6")

str(train$Response) 
# Tuning syntax from http://stats.stackexchange.com/questions/171043/how-to-tune-hyperparameters-of-xgboost-trees
fitControl <- trainControl(method = "cv", number = 3) # , repeats = 3

tune_grid = expand.grid(
  nrounds = c(150),
  eta = c(0.1), #, 0.003 , 0.2
  max_depth = c(4) #, 8, 10
)

gbmGrid <- expand.grid(interaction.depth = 2, n.trees = 30, shrinkage = 0.1, n.minobsinnode=4)

date()
cat("Fitting model\n")
fit <- train(x=data.matrix(train[,feature.names1]), y=train$Response,
             method = "gbm", 
             trControl = fitControl,
             tuneGrid = gbmGrid,
             metric = "Kappa", 
             verbose = TRUE)

date()       
summary(fit)

cat("Predict model\n")
#test$Response <- round(predict(fit, newdata=data.matrix(test[,feature.names])))
test$Response <- predict(fit, newdata=data.matrix(test[,feature.names1]))

date()   
cat("saving the submission file\n")
submission <- data.frame(Id=test$Id, Response=test$Response)

#submission[submission$Response<1, "Response"] <- 1
#submission[submission$Response>8, "Response"] <- 8


write_csv(submission, "Pru_caret_cv_gbm_cls.csv")

cat("Validation predict model\n")
train$cvResponse <- predict(fit, newdata=data.matrix(train[,feature.names1]))

date()   
cat("saving the validation file\n")
val <- data.frame(Id=train$Id, Response=train$Response, cvResponse=train$cvResponse )

write_csv(val, "data/caret_gmb_submission.csv")
round((table(train$cvResponse,train$Response)/nrow(train))*100,1)

library("Metrics")
ScoreQuadraticWeightedKappa(as.numeric(train$cvResponse),as.numeric(train$Response))


