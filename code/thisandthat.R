
source("./code/0_init.R")


generator.test = flow_images_from_directory(
  paste0(dataloc,"test"),  
  image_data_generator(rescale = 1/255),
  target_size = c(150, 150),  
  batch_size = batchsize,  
  class_mode = "binary",
  shuffle = FALSE)

yhat = predict_generator(model.2, generator.test, steps = n.test/batchsize)[,1]
#yhat = predict_proba(model.4, generator.test)
y = as.factor(ifelse(generator.test$classes[generator.test$index_array+1] == 0, "N", "Y"))

mysummary_class(data.frame(yhat = yhat, y = y))


generator.test$class_indices
batch = generator_next(generator.test)
batch[[2]]


yhat = predict(model.2, test$x)
y = as.factor(ifelse(test$y == 0, "N", "Y"))

df.train = as.data.frame(train$x)
predictors = colnames(df.train)
df.train$target = as.factor(ifelse(train$y == 0, "N", "Y"))
formula = as.formula(paste("target", "~", paste(predictors, collapse = " + ")))
fit = train(formula, df.train[c("target",predictors)],
            trControl = trainControl(method="none"),
            method = "xgbTree", 
            tuneGrid = expand.grid(nrounds = 2000, max_depth = 12, 
                                   eta = c(0.01), gamma = 0, colsample_bytree = 0.3, 
                                   min_child_weight = 5, subsample = 0.9))
df.test = as.data.frame(test$x)
yhat = predict(fit, df.test, type = "prob")[[2]]
y = as.factor(ifelse(test$y == 0, "N", "Y"))
mysummary_class(data.frame(yhat = yhat, y = y))


              