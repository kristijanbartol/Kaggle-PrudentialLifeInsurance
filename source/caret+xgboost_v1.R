library(caret)
library(xgboost)
library(readr)
library(corrgram)

# Read input data files
train = read_csv("data/train.csv")
test  = read_csv("data/test.csv")

# Set a random seed for reproducability
set.seed(45)

#################
# Data analysis #
#################
#sum(complete.cases(train$Employment_Info_1)) / nrow(train) * 100 # -> PERCENTAGE_RATIO(complete.cases, ALL)

# Separate categorical, continuous and discrete variables
#cat.var.names <- c(paste("Product_Info_", c(1:3,5:7), sep=""), paste("Employment_Info_", c(2,3,5), sep=""),
#                   paste("InsuredInfo_", 1:7, sep=""), paste("Insurance_History_", c(1:4,7:9), sep=""), 
#                   "Family_Hist_1", paste("Medical_History_", c(2:14, 16:23, 25:31, 33:41), sep=""))
#cont.var.names <- c("Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4", 
#                    "Employment_Info_6", "Insurance_History_5", "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", 
#                    "Family_Hist_5")
#disc.var.names <- c("Medical_History_1", "Medical_History_15", "Medical_History_24", "Medical_History_32", 
#                    paste("Medical_Keyword_", 1:48, sep=""))

# Numerical are all variables except Product_Info_2, which is strictly categorical
#numerical <- train[sapply(train, is.numeric)]

# Calculate correlation matrix
#correlationMatrix <- cor(numerical, use = "p")
#correlationMatrix[!lower.tri(correlationMatrix)] <- 0

# This filters ones that are not more-than-cond correlated
#highlyCorrelated <- correlationMatrix[,apply(correlationMatrix, 2, function(x) any(x > 0.5))]
# Print indexes of highly correlated attributes
#head(highlyCorrelated)

# Write feature correlation to a file
#write_csv(as.data.frame(correlationMatrix > 0.75), "data/highCorrelation.csv")

# The variable with most NAs
non_na_mh32 <- subset(train$Medical_History_32, !is.na(train$Medical_History_32))

#train.cat <- train[, cat.var.names]
#test.cat <- test[, cat.var.names]

#train.cont <- train[, cont.var.names]
#test.cont <- test[, cont.var.names]

#train.disc <- train[, disc.var.names]
#test.disc <- test[, disc.var.names]

#train.cat <- as.data.frame(lapply(train.cat, factor))
#test.cat <- as.data.frame(lapply(test.cat, factor))

#summary(train.cat)
#summary(train.cont)
#summary(train.disc)
#summary(test.cat)
#summary(test.cont)
#summary(test.disc)

cat("Missing data per feature:")
apply(train, 2, function(x) { sum(is.na(x)) })

cat("Training data has ", sum(is.na(train)) / (nrow(train) * ncol(train)) * 100, "% of missing data!")

cat("Missing data per response")
round(colSums(train.na.per.response) / sum(train.na.per.response), digits=4)

##########################
# Preprocessing the data #
##########################

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

#train$Employment_Info_6 <- sapply(train$Employment_Info_6, function(x) x * 200)
#train$Employment_Info_6 <- as.integer(round(train$Employment_Info_6, digits = 0))
#train$Employment_Info_6[which(is.na(train$Employment_Info_6))] <- round(mean(train$Employment_Info_6))

normalization <- preProcess(train[2:127], 
                           thresh = 0.95,
                           na.remove = TRUE,
                           k = 5,
                           knnSummary = mean,
                           verbose = TRUE
)
trainNorm <- predict(normalization, train[2:127])

NArows = c(which(colnames(train) == "Medical_History_10"),
           which(colnames(train) == "Employment_Info_1"),
           which(colnames(train) == "Employment_Info_4"),
           which(colnames(train) == "Employment_Info_6"),
           which(colnames(train) == "Insurance_History_5"),
           which(colnames(train) == "Family_Hist_2"),
           which(colnames(train) == "Family_Hist_3"),
           which(colnames(train) == "Family_Hist_4"),
           which(colnames(train) == "Family_Hist_5"),
           which(colnames(train) == "Medical_History_1"),
           which(colnames(train) == "Medical_History_15"),
           which(colnames(train) == "Medical_History_24"),
           which(colnames(train) == "Medical_History_32")
)
imputation <- preProcess(train[NArows],
                         method = "knnImpute",
                         pcaComp = 10,
                         na.remove = TRUE,
                         k = 5,
                         knnSummary = mean,
                         outcome = NULL,
                         fudge = .2,
                         verbose = TRUE
)
# Does it make sense to impute the data? Maybe just to try with and without it...
trainImputed <- predict(imputation, train[NArows])

trainData    <- train[2:127]
trainClasses <- factor(train$Response)
levels(trainClasses) <- list("R1"=1,"R2"=2,"R3"=3,"R4"=4,"R5"=5,"R6"=6,"R7"=7,"R8"=8)

##################
# Model training #
##################
# set up the cross-validated hyper-parameter search
xgb.grid <- expand.grid (nrounds = 10,
                         max_depth = c(2, 4, 6, 8, 10, 14),
                         eta = c(0.01, 0.001, 0.0001),
                         gamma = 1,
                         colsample_bytree = c(.6, .8),
                         min_child_weight = c(1)
)

# pack the training control parameters
cv.ctrl <- trainControl(method = "repeatedcv", 
                        repeats = 2,
                        number = 5,
                        verboseIter = TRUE,
                        returnData = FALSE,
                        returnResamp = "all",
                        classProbs = TRUE,
                        seeds = NA,
                        allowParallel = TRUE
)

# eXtreme Gradient Boosting (xgboost)
xgb_tune <- train (trainData, trainClasses,
                   method = "xgbTree",
                   trControl = cv.ctrl,
                   tuneGrid = xgb.grid,
                   # na.action = na.omit
                   verbose = TRUE,
                   metric = "Kappa",
                   nthread = 3
)

#######################
# Evaluate and export #
#######################

cat("Predict model\n")
test$Response <- predict(xgb_tune, newdata=data.matrix(test[2:127]))

# Save test set result to the submission file
cat("saving the submission file\n")
submission <- data.frame(Id=test$Id, Response=as.numeric(test$Response))

write_csv(submission, "data/caret+xgboost_v1.csv")

###################
# Analyze results #
###################
# scatter plot of the AUC against max_depth and eta
ggplot(xgb_tune$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
       geom_point() + 
       theme_bw() + 
       scale_size_continuous(guide = "none")


# Plot the response of a training set
ggplot(train) + geom_histogram(aes(factor(Response))) + xlab("Response") + theme_light()

# Class Prediction
modelClasses <- predict(xgb_tune, newdata = trainData)
head(modelClasses,10)

# Class probability
modelProbs <- predict(xgb_tune, newdata = trainData, type = "prob")
head(modelProbs,10)
#hist(modelProbs$N)
#hist(modelProbs$Y)
plot(sort(modelProbs$L1),col="blue",ylab="class.probability")
points(1-sort(modelProbs$Y),col="red")

# Performance Statistics
confusionMatrix(modelClasses, trainClasses) 

plot(xgb_tune)
