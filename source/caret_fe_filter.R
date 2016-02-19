
library(caret)
library(readr)

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

filterCtrl <- sbfControl(functions = rfSBF, verbose = TRUE)

train.noNA <- train[which(colSums(is.na(train[,feature.names])) == 0)]

set.seed(10)

rfWithFilter <- sbf(train[9], train$Response, sbfControl = filterCtrl)
rfWithFilter
