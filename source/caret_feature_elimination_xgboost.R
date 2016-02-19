# http://topepo.github.io/caret/rfe.html
# arbitrary model -> xgboost

library(xgboost)
library(caret)
library(readr)

set.seed(1)

train <- read_csv("data/train.csv")
test  <- read_csv("data/test.csv")

feature.names <- names(train)[2:ncol(train)-1]

cat("Assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

example <- data.frame(RMSE = c(1.2, 1.1, 1.05, 1.01, 1.01, 1.03, 1.00),
                      v2 = c(1.2, 1.1, 1.05, 1.01, 1.01, 1.03, 1.00),
                      Variables = 1:7)

cat("Finding the row with the absolute smallest RMSE")
smallest <- pickSizeBest(example, metric = "RMSE", maximize = FALSE)
smallest

cat("Now one that is within 10% of the smallest")
within10Pct <- pickSizeTolerance(train[,feature.names], metric = "RMSE", tol = 10, maximize = FALSE)
within10Pct

minRMSE <- min(example$RMSE)
example$Tolerance <- (example$RMSE - minRMSE)/minRMSE * 100

cat("Plot the profile and the subsets selected using the two different criteria")

par(mfrow = c(2, 1), mar = c(3, 4, 1, 2))

plot(example$Variables[-c(smallest, within10Pct)],
     example$RMSE[-c(smallest, within10Pct)],
     ylim = extendrange(example$RMSE),
     ylab = "RMSE", xlab = "Variables")

points(example$Variables[smallest],
       example$RMSE[smallest], pch = 16, cex= 1.3)

points(example$Variables[within10Pct],
       example$RMSE[within10Pct], pch = 17, cex= 1.3)

with(example, plot(Variables, Tolerance))
abline(h = 10, lty = 2, col = "darkgrey")


train.noNA <- train[which(colSums(is.na(train[,feature.names])) == 0)]

cat("Recursive feature elimination function")
xgbRFE <- list(summary = defaultSummary,
               fit = function(x, y, first, last, ...) {
                 library(randomForest)
                 randomForest(train.noNA, y, 
                              importance = first, 
                              ...)
                 #xgboost(data        = data.matrix(train[,feature.names]),
                  #       label       = train$Response,
                  #       eta         = 0.025,
                  #       depth       = 10,
                  #       nrounds     = 2000,
                  #       objective   = "reg:linear",
                  #       eval_metric = "rmse")
               },
               pred = function(object, x) predict(object, x),
               rank = function(object, x, y) {
                 vimp <- varImp(object)
                 vimp <- vimp[order(vimp$Overall,decreasing = TRUE),,drop = FALSE]
                 vimp$var <- rownames(vimp)
                 vimp
               },
               selectSize = pickSizeBest,
               selectVar = pickVars
)

#ctrl <- rfeControl(functions = xgbRFE,
#                   method = "cv",
#                   verbose = TRUE)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "cv",
                   returnResamp = "all",
                   verbose = TRUE)

subsets <- c(10, 20, 35, 50, 70, 90, 100, 105, 110)

#ctrl$functions <- rfFuncs
ctrl$returnResamp <- "all"
set.seed(10)
rfeProfile <- rfe(train.noNA, train$Response, sizes = subsets, rfeControl = ctrl)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    rfeProfile
