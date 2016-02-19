# H2O Starter Script
library(h2o)
library(readr)

h2o.init(nthreads=-1)

categoricalVariables = c("Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5", "Product_Info_6", "Product_Info_7", "Employment_Info_2", "Employment_Info_3", "Employment_Info_5", "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7", "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", "Insurance_History_4", "Insurance_History_7", "Insurance_History_8", "Insurance_History_9", "Family_Hist_1", "Medical_History_2", "Medical_History_3", "Medical_History_4", "Medical_History_5", "Medical_History_6", "Medical_History_7", "Medical_History_8", "Medical_History_9", "Medical_History_10", "Medical_History_11", "Medical_History_12", "Medical_History_13", "Medical_History_14", "Medical_History_16", "Medical_History_17", "Medical_History_18", "Medical_History_19", "Medical_History_20", "Medical_History_21", "Medical_History_22", "Medical_History_23", "Medical_History_25", "Medical_History_26", "Medical_History_27", "Medical_History_28", "Medical_History_29", "Medical_History_30", "Medical_History_31", "Medical_History_33", "Medical_History_34", "Medical_History_35", "Medical_History_36", "Medical_History_37", "Medical_History_38", "Medical_History_39", "Medical_History_40", "Medical_History_41")
specifiedTypes = rep("Enum", length(categoricalVariables))

cat("reading the train and test data\n")
trainlocal <- read_csv("data/train.csv")
testlocal  <- read_csv("data/test.csv")

cat("loading into h2o")
train <- as.h2o(trainlocal)
test <- as.h2o(testlocal)

cat("Converting to categorical variables")
for (f in categoricalVariables) {
  train[[f]] <- as.factor( train[[f]] )
  test[[f]]  <- as.factor( test[[f]]  )
}

independentVariables = names(train)[3:ncol(train)-1]
dependentVariable = names(train)[128]

cat("Training gbm")
h2oGbm <- h2o.gbm(x=independentVariables, y=dependentVariable, training_frame = train, 
                  learn_rate=0.025, ntrees=235, max_depth=22, min_rows=3)

cat("Creating submission frame")
prediction <- as.data.frame( predict(h2oGbm, test) )
submission <- as.data.frame(test$Id)
submission <- cbind(submission, round(prediction$predict))
names(submission) <- c("Id", "Response")

# I pretended this was a regression problem and some predictions may be outside the range
submission[submission$Response<1, "Response"] <- 1
submission[submission$Response>8, "Response"] <- 8

cat("saving the submission file\n")
write_csv(submission, "data/h2o_nontuned_submission.csv")

