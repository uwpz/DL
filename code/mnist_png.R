
skip = function() {
  from = paste0("./data/mnist/mnist_png/big/train/")
  to = paste0("./data/mnist/mnist_png/small/train/")
  l.files = list.files(from)
  n = 500
  for (i in 0:9) {
    #i=1
    subdir = paste0(to,"d",i)
    dir.create(subdir)
    set.seed(i*123)
    files = paste0(paste0(from,"d",i),"/",sample(list.files(paste0(from,"d",i)), n))
    file.copy(files,subdir)
  }
}

#######################################################################################################################-
#|||| Initialize and ETL ||||----
#######################################################################################################################-

## Libraries
library(reticulate)
library(magick) 
library(viridis)


## Backend
use_python("C:\\Anaconda\\envs\\r-tensorflow\\python.exe", required = TRUE)
#use_python("C:\\Anaconda\\envs\\r\\cntk-py35\\python.exe", required = TRUE)
#use_condaenv(condaenv = "cntk-py35", conda = "C:/local/Anaconda3-4.1.1-Windows-x86_64/Scripts/conda.exe", required = TRUE)
#use_condaenv(condaenv = "cntk-py35", required = TRUE)
library(keras)
use_backend("tensorflow")
#use_backend("cntk")
k_backend()
#cmd: "nvidia-smi -l" to monitor gpu-usage

# Functions
source("code/0_init.R")



## Parmeter
type = "small"
#load(paste0(type,"_cats_vs_dogs.RData"))

dataloc = paste0("./data/mnist/mnist_png/",type,"/")

(n.train = length(list.files(paste0(dataloc,"train"), recursive = TRUE)))
(n.test = length(list.files(paste0(dataloc,"test"), recursive = TRUE)))
n.validate = length(list.files(paste0(dataloc,"test"), recursive = TRUE)) # !!!!!!!!!!!!! Currently same as test

batchsize = 20
target_size = c(28*4, 28*4)




#######################################################################################################################-
#|||| Prepare ||||----
#######################################################################################################################-

# Get data --------------------------------------------------------------------------

## Validate and test
# Validate 
generator.validate = flow_images_from_directory(
  paste0(dataloc,"test"),   # !!!!!!!!!!!!! Currently same as test
  image_data_generator(rescale = 1/255),
  target_size = target_size,  
  batch_size = batchsize,  
  class_mode = "categorical")
# Plot
par(mfrow = c(2,2), mar = c(2,0,2,0)) 
for (i in 1:4) {
  plot(as.raster(generator_next(generator.validate)[[1]][i,,,]))
  title(paste0("Image\n",i), cex.main = 1)
}  

# Test (same as validate)
generator.test = flow_images_from_directory(
  paste0(dataloc,"test"),  
  image_data_generator(rescale = 1/255),
  target_size = target_size,  
  batch_size = batchsize,  
  class_mode = "categorical",
  shuffle = FALSE) #no shuffle !!!
# Plot
par(mfrow = c(2,2), mar = c(2,0,2,0)) 
for (i in 1:4) {
  plot(as.raster(generator_next(generator.test)[[1]][i,,,]))
  title(paste0("Image\n",i), cex.main = 1)
}  




## Train
# Data augmentation
datagen.augment = image_data_generator(
  rescale = 1/255,
  #rotation_range = 40,
  #width_shift_range = 0.2,
  #height_shift_range = 0.2,
  #shear_range = 0.2,
  #zoom_range = 0.2,
  # horizontal_flip = TRUE,
  fill_mode = "nearest"
)

# Check augmentation
img = image_load(paste0(dataloc,"train/d0/21.png"), target_size = c(150,150)) %>% #TODO
  image_to_array() %>% 
  array_reshape(c(1,150,150,3))
par(mfrow = c(2,2), mar = c(2,0,2,0)) 
plot(as.raster(img[1,,,]/255)); title("Orig")
for (i in 1:3) {
  generator_next(flow_images_from_data(img, generator = datagen.augment, batch_size = 1))[1,,,] %>% 
    as.raster() %>% 
    plot()
}   

# Generator
generator.train = flow_images_from_directory(
  paste0(dataloc,"train"),  
  datagen.augment,  
  target_size = target_size,  
  batch_size = batchsize,  
  class_mode = "categorical")




#######################################################################################################################-
#|||| Small convnet ||||----
#######################################################################################################################-

# Fit --------------------------------------------------------------------------

# Model definition
# model.1 = keras_model_sequential() %>% 
#   layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
#                 input_shape = c(target_size, 3)) %>% 
#   layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
#   layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
#   layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
#   layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
#   layer_flatten() %>% 
#   layer_dense(units = 64, activation = "relu") %>% 
#   layer_dense(units = 10, activation = "softmax") %>% 
#   compile(
#     loss = "categorical_crossentropy",
#     optimizer = optimizer_rmsprop(lr = 1e-4),
#     metrics = c("accuracy")
#   )
model.1 = keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(target_size, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax") %>% 
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("accuracy")
  )
model.1

# Fit
fit.1 = model.1 %>% fit_generator(
  generator.train,
  steps_per_epoch = n.train/batchsize, 
  #initial_epoch = 1, #must be less than epoch
  epochs = 10, 
  validation_data = generator.validate,
  validation_steps = n.validate/batchsize 
)
plot(fit.1)

# Evaluate
model.1 %>% evaluate_generator(generator.test, steps = n.validate/batchsize)
yhat = predict_generator(model.1, generator.test, steps = n.test/batchsize)
colnames(yhat) = paste0("d",0:9)
y = as.factor(paste0("d",generator.test$classes))
mysummary(data.frame(yhat, y = y))


# Save
#model.1 %>% save_model_hdf5(paste(type,"_model1.h5"))
#model.1 = load_model_hdf5(paste(type,"_model1.h5"))




# Interpret --------------------------------------------------------------------------

## Plot images with low and high residuals
# Get residuals
y_num = matrix(0, nrow(yhat), 10); colnames(y_num) = colnames(yhat);
y_num[as.matrix(data.frame(1:nrow(y_num),generator.test$classes + 1))] = 1
res = rowSums(abs(yhat - y_num))
order(res)
k = 90
i.img_low = order(res)[1:k]
res[i.img_low]
i.img_high = order(res, decreasing = TRUE)[1:k]
res[i.img_high]
round(yhat[i.img_high,], 2); y_num[i.img_high,]

# Plot
files = list.files(paste0(dataloc,"test"), recursive = TRUE)
par(mfrow = c(3,3), mar = c(1,0,1,0)) 
print("low residuals")
generator.test$classes[i.img_low]
for (i in 1:k) {
  plot(image_read(paste0(dataloc,"test/",files[i.img_low[i]])))
  title(paste0("Class = ",generator.test$classes[i.img_low[i]], 
               ",yhat = ",colnames(yhat)[which.max(yhat[i.img_low[i],])],
               ",prob = ",round(max(yhat[i.img_low[i],]), 2)), 
        cex.main = 1)
}
print("high residuals")
generator.test$classes[i.img_high]
for (i in 1:k) {
  plot(image_read(paste0(dataloc,"test/",files[i.img_high[i]])))
  title(paste0("Class = ",generator.test$classes[i.img_high[i]], 
               ",yhat = ",colnames(yhat)[which.max(yhat[i.img_high[i],])],
               ",prob = ",round(max(yhat[i.img_high[i],]),2)), 
        cex.main = 1)
}
#dev.off()



## Create CAM ("class activation map")

plot_cam = function(img_path = paste0(dataloc,"test/",files[i.img_high[i]]),
                   #target_size = target_size,
                    model = model.1,
                    layer_name = "conv2d_37",
                    titles = "blub", cex = 1,
                    reverse = FALSE) {
  # Load image
  img = image_load(img_path, target_size = target_size) %>% 
    image_to_array() %>% 
    array_reshape(c(1,target_size,3)) / 255  
  
  # The is the output feature map of the last convolutional layer
  last_conv_layer = model %>% get_layer(layer_name)
  
  # This is the gradient of the "2" class with regard to the output feature map of `conv2d_4`
  grads = k_gradients((-1)*reverse*model$output[, 6], last_conv_layer$output)[[1]]
  
  # This is a vector of shape (512,), where each entry is the mean intensity of the gradient 
  # over a specific feature map channel
  pooled_grads = k_mean(grads, axis = c(1, 2, 3))
  
  # This function allows us to access the values of the quantities we just defined:
  # `pooled_grads` and the output feature map of `conv2d_4`, given a sample image
  iterate <- k_function(list(model$input),
                        list(pooled_grads, last_conv_layer$output[1,,,])) 
  
  # These are the values of these two quantities, as arrays, given our sample image 
  c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))
  
  # We multiply each channel in the feature map array by "how important this channel is" with regard 
  # to the "dog" class
  for (z in 1:last_conv_layer$filters) {
    conv_layer_output_value[,,z] <- 
      conv_layer_output_value[,,z] * pooled_grads_value[[z]] 
  }
  
  # The channel-wise mean of the resulting feature map is our heatmap of class activation
  heatmap = apply(conv_layer_output_value, c(1,2), mean)
  heatmap = pmax(heatmap, 0) 
  heatmap = heatmap / max(heatmap)
  
  # Read the original image and it's geometry
  image <- image_read(img_path)
  plot(image)
  title(titles[1], cex.main = cex)
  info = image_info(image) 
  geometry = sprintf("%dx%d!", info$width, info$height) 
  
  # Create a blended / transparent version of the heatmap image
  png("heatmap_for_overlay.png", width = 10 * dim(heatmap)[1], height = 10 * dim(heatmap)[1], bg = NA)
  par(mar = c(0,0,0,0))
  image(t(apply(heatmap, 2, rev)), axes = FALSE, asp = 1, 
        col = paste0(colorRampPalette(c("blue","yellow","red"))(dim(heatmap)[1]), 70))
  dev.off()
  
  # Overlay the heatmap
  image_read("heatmap_for_overlay.png") %>% 
    image_resize(geometry, filter = "quadratic") %>% 
    image_composite(image_resize(image, geometry), operator = "blend", compose_args = "40") %>%
    plot()
  title(titles[2], cex.main = cex)
  box()
}

# Clear lower level access
#k_clear_session()

par(mfrow = c(4,4), mar = c(1,0,1,0)) 
for (i in 1:8) {
  plot_cam(img_path = paste0(dataloc,"test/",files[i.img_high[i]]),
           titles = c(paste0("Class = ",generator.test$classes[i.img_high[i]]),
                      paste0("yhat = ",colnames(yhat)[which.max(yhat[i.img_high[i],])])))
}



#######################################################################################################################-
#|||| Feature extraction 1 (WITHOUT data augmentation) ||||----
#######################################################################################################################-

## Get pretrained vgg16 convbase
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(target_size, 3)
)
conv_base



## Create feature matrix

# Get output from last layer
extract_features = function(directory = paste0(dataloc,"train"), 
                            sample_count = 2000, size = 100, last_layer_size = 3 * 3 * 512) {
  
  # Initialize x and y
  a.x = array(0, dim = c(sample_count, last_layer_size))  
  a.y = array(0, dim = c(sample_count, 10))
  
  # Define generator
  generator = flow_images_from_directory(
    directory,  
    image_data_generator(rescale = 1/255),  
    target_size = target_size,  
    batch_size = size,  
    class_mode = "categorical")
  
  # Run through images and fill featurs and y
  i = 1
  while(i * size <= sample_count) {
    print(i * size)
    
    batch = generator_next(generator)
    index_range = ((i-1) * size +1):(i * size)
    a.x[index_range,] = conv_base %>% 
      predict(batch[[1]]) %>% #Get last layer
      array_reshape(dim = c(size, last_layer_size)) #Flatten
    a.y[index_range,] <- batch[[2]]
    
    i = i + 1
  }
  
  list(x = a.x, y = a.y)
}
train = extract_features(paste0(dataloc,"train"), n.train)
test = extract_features(paste0(dataloc,"test"), n.test)
#validate = extract_features(paste0(dataloc,"validate"), n.validate)
validate = test
l.save = list("train" = train, "validate" = validate, "test" = test)
save(l.save, file = paste0(type,"_features.RDATA"))


## Train small model on x (could also be done with other ML algorithm)
model.2 = keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", 
              input_shape = 3 * 3 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 10, activation = "softmax") %>% 
  compile(
    optimizer = optimizer_rmsprop(lr = 1e-5),
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
model.2
fit.2 = model.2 %>% fit(
  train$x, train$y,
  batch_size = batchsize,
  #initial_epoch = 1,
  epochs = 30,
  validation_data = list(validate$x, validate$y)
)
plot(fit.2)
model.2 %>% evaluate(test$x, test$y)
yhat = predict_proba(model.2, test$x)
colnames(yhat) = paste0("d",0:9)
y = as.factor(paste0("d",generator.test$classes))
mysummary(data.frame(yhat, y = y))

# Save
#model.2 %>% save_model_hdf5(paste(type,"_model2.h5"))
#model.2 = load_model_hdf5(paste(type,"_model2.h5"))



#######################################################################################################################-
#|||| Feature extraction 2 (with data augmentation) ||||----
#######################################################################################################################-

# Freeze conv_base
freeze_weights(conv_base)

# Enlarge with dense layers
model.3 = keras_model_sequential() %>% 
  conv_base %>% 
  #a %>% 
  #layer_conv_2d(filters = 512, activation = "relu") %>%
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax") %>% 
  compile(
    loss = "categorical_crossentropy",
    #optimizer = optimizer_rmsprop(lr = 2e-5),
    optimizer = optimizer_adam(lr = 2e-5),
    metrics = c("acc")
  )
model.3
fit.3 = model.3 %>% fit_generator(
  generator.train,
  steps_per_epoch = n.train/batchsize, 
  #initial_epoch = 90,
  epochs = 10, 
  validation_data = generator.validate,
  validation_steps = n.validate/batchsize 
)
plot(fit.3)
model.3 %>% evaluate_generator(generator.test, steps = n.test/batchsize)
yhat = predict_generator(model.3, generator.test, steps = n.test/batchsize)
colnames(yhat) = paste0("d",0:9)
y = as.factor(paste0("d",generator.test$classes))
mysummary(data.frame(yhat, y = y))

# Save
model.3 %>% save_model_hdf5(paste(type,"_model3.h5"))
#model.3 = load_model_hdf5(paste(type,"_model3.h5"))

## xgboost
df.train = as.data.frame(train$x)
predictors = colnames(df.train)
df.train$target = as.factor(paste0("d", apply(train$y, 1, which.max) - 1))
formula = as.formula(paste("target", "~", paste(predictors, collapse = " + ")))
set.seed(999)
l.index = list(i = sample(1:nrow(df.train), floor(0.8*nrow(df.train))))
ctrl_idx_fff = trainControl(method = "cv", number = 1, index = l.index, 
                            returnResamp = "final", returnData = FALSE,
                            summaryFunction = mysummary, classProbs = TRUE, 
                            indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!
ctrl_idx_nopar_fff = trainControl(method = "cv", number = 1, index = l.index, 
                                  returnResamp = "final", returnData = FALSE,
                                  allowParallel = FALSE, #no parallel e.g. for xgboost on big data or with DMatrix
                                  summaryFunction = mysummary, classProbs = TRUE, 
                                  indexFinal = sample(1:nrow(df.train), 100)) #"Fast" final fit!!!
fit = train(formula, df.train[c("target",predictors)],
            trControl = ctrl_idx_nopar_fff,
            method = "xgbTree", 
            tuneGrid = expand.grid(nrounds = seq(100,2100, 200), max_depth = 12, 
                                   eta = c(0.01), gamma = 0, colsample_bytree = 0.3, 
                                   min_child_weight = 5, subsample = 0.5))
plot(fit)
# df.test = as.data.frame(test$x)
# yhat = predict(fit, df.test, type = "prob")
# y = as.factor(ifelse(test$y == 0, "N", "Y"))
# mysummary_class(data.frame(yhat = yhat, y = y))







#######################################################################################################################-
#|||| Fine Tuning ||||----
#######################################################################################################################-

# Unfreeze: IMPORTANT to first fit model.3
unfreeze_weights(conv_base, from = "block3_conv1")
model.4 = model.3

# Fit
model.4 %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  #optimizer = optimizer_adam(lr = 1e-6),
  metrics = c("accuracy")
)
fit.4 = model.4 %>% fit_generator(
  generator.train,
  steps_per_epoch = n.train/batchsize,
  epochs = 30,
  validation_data = generator.validate,
  validation_steps = n.validate/batchsize
)
plot(fit.4)
model.4 %>% evaluate_generator(generator.test, steps = n.test/batchsize)
yhat = predict_generator(model.4, generator.test, steps = n.test/batchsize)
y = as.factor(paste0("d",generator.test$classes))
mysummary_class(data.frame(yhat = yhat, y = y))
# plots = get_plot_performance_class(yhat, y)
# ggsave(paste0(plotloc, "model.4_performance.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
#        w = 18, h = 12)
# # Save
# model.4 %>% save_model_hdf5(paste(type,"_model4.h5"))
# #model.4 = load_model_hdf5(paste(type,"_model4.h5"))




#save.image(paste0(type,"_m.RData"))
