library(caret)
library(readr)

set.seed(148)

# set up the cross-validated hyper-parameter search
xgb.grid <- expand.grid (nrounds = 1000,
                         lambda = c(.01, .1),
                         alpha = c(.01, .1) 
)

# pack the training control parameters
cv.ctrl <- trainControl(method = "repeatedcv",
                        repeats = 2,
                        number = 5,
                        verboseIter = TRUE,
                        returnData = FALSE,
                        returnResamp = "all",
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary,
                        allowParallel = TRUE
)

xgb_tune <- train (train, train$Response,
                   method = "xgbLinear",
                   trControl = cv.ctrl,
                   tuneGrid = xgb.grid,
                   verbose = TRUE,
                   metric = "RMSE",
                   nthread = 3
)
