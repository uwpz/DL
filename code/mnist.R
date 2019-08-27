
#######################################################################################################################-
#|||| Initialize and ETL ||||----
#######################################################################################################################-

## Libraries
library(reticulate)
use_python("C:\\Anaconda\\envs\\r-tensorflow\\python.exe", required = TRUE)
#use_python("C:/local/Anaconda3-4.1.1-Windows-x86_64/envs/cntk-py35/python.exe", required = TRUE)
#use_condaenv(condaenv = "cntk-py35", conda = "C:/local/Anaconda3-4.1.1-Windows-x86_64/Scripts/conda.exe", required = TRUE)
#use_condaenv(condaenv = "cntk-py35", required = TRUE)
library(keras)
use_backend("tensorflow")
#use_backend("cntk")
k_backend()


## Get data
# Mnist
mnist = dataset_mnist()

# Initialize
train = list("x" = NA, "y" = NA)
test = list("x" = NA, "y" = NA)

# Dummy coding
train$y = to_categorical(mnist$train$y)
test$y = to_categorical(mnist$test$y)

# Reshape to 4D-tensor and rescale
train$x = array_reshape(mnist$train$x, c(60000, 28, 28, 1)) / 255
test$x = array_reshape(mnist$test$x, c(10000, 28, 28, 1)) / 255
plot(as.raster(train$x[1,,,], max = 1))



#######################################################################################################################-
#|||| Fit ||||----
#######################################################################################################################-

model = keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax") %>% 
  compile(
    optimizer = "rmsprop",
    #optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
model
model %>% fit(
  train$x, train$y, 
  batch_size = 64,
  epochs = 10 
)

model %>% evaluate(test$x, test$y, verbose = 0)
