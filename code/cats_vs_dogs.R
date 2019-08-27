
#######################################################################################################################-
#|||| Initialize and ETL ||||----
#######################################################################################################################-

## Libraries
library(reticulate)
library(magick) 
library(viridis)



## Backend
#use_python("C:\\ProgramData\\Python\\Python36\\python.exe", required = TRUE)
#use_condaenv(condaenv = "cntk-py35", conda = "C:/local/Anaconda3-4.1.1-Windows-x86_64/Scripts/conda.exe", required = TRUE)
library(keras)
k_backend()
#use_backend("tensorflow")
#use_backend("cntk")
#cmd: "nvidia-smi -l" to monitor gpu-usage



## Functions
source("code/0_init.R")



## Parmeter
type = "small"
#load(paste0(type,"_cats_vs_dogs.RData"))

dataloc = paste0("./data/cats_vs_dogs/",type,"/")

(n.train = length(list.files(paste0(dataloc,"train"), recursive = TRUE)))
(n.validate = length(list.files(paste0(dataloc,"validate"), recursive = TRUE)))
(n.test = length(list.files(paste0(dataloc,"test"), recursive = TRUE)))


batchsize = 20




#######################################################################################################################-
#|||| Prepare ||||----
#######################################################################################################################-

# Get data --------------------------------------------------------------------------

## Validate and test
# Validate 
generator.validate = flow_images_from_directory(
  paste0(dataloc,"validate"),  
  image_data_generator(rescale = 1/255),
  target_size = c(150, 150),  
  batch_size = batchsize,  
  class_mode = "binary")

# Plot
par(mfrow = c(2,2), mar = c(2,0,2,0)) 
for (i in 1:4) {
  generator_next(generator.validate)[[1]][i,,,] %>% as.raster() %>% plot()
  title(paste0("Image\n",i), cex.main = 1)
}  
    
# Test 
generator.test = flow_images_from_directory(
  paste0(dataloc,"test"),  
  image_data_generator(rescale = 1/255),
  target_size = c(150, 150),  
  batch_size = batchsize,  
  class_mode = "binary",
  shuffle = FALSE) #no shuffle !!!



## Train
# Data augmentation
datagen.augment = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

# Generator
generator.train = flow_images_from_directory(
  paste0(dataloc,"train"),  
  datagen.augment,  
  target_size = c(150, 150),  
  batch_size = batchsize,  
  class_mode = "binary")

# Check augmentation
img = image_load(paste0(dataloc,"train/cats/00003.jpg"), target_size = c(150,150)) %>% 
  image_to_array() %>% 
  array_reshape(c(1,150,150,3))
par(mfrow = c(2,2), mar = c(2,0,2,0)) 
plot(as.raster(img[1,,,]/255)); title("Orig")
for (i in 1:3) {
  generator_next(flow_images_from_data(img, generator = datagen.augment, batch_size = 1))[1,,,] %>% 
  as.raster() %>% 
  plot()
}   






#######################################################################################################################-
#|||| Small convnet ||||----
#######################################################################################################################-

# Fit --------------------------------------------------------------------------

# Model definition
model.1 = keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
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
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
  )
model.1

# Fit
fit.1 = model.1 %>% fit_generator(
  generator.train,
  steps_per_epoch = n.train/batchsize, 
  #initial_epoch = 1, #must be less than epoch
  epochs = 15, 
  validation_data = generator.validate,
  validation_steps = n.validate/batchsize 
)
plot(fit.1)

# Evaluate
model.1 %>% evaluate_generator(generator.test, steps = n.validate/batchsize)
yhat = predict_generator(model.1, generator.test, steps = n.test/batchsize)
y_num = as.vector(generator.test$classes)
y = factor(ifelse(y_num == 0, "N", "Y"))
performance_summary(data.frame(yhat = data.frame("Y"=yhat, "N"=1-yhat), y = y))
plots = plot_all_performances(yhat = data.frame("Y"=yhat, "N"=1-yhat), y = y)
ggsave(paste0(plotloc, "model.1_performance.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)

# Save
#model.1 %>% save_model_hdf5(paste(type,"_model1.h5"))
#model.1 = load_model_hdf5(paste(type,"_model1.h5"))




# Interpret --------------------------------------------------------------------------

## Plot images with low and high residuals
# Get residuals
res = abs(yhat - y_num)
order(res)
k = 9
i.img_low = order(res)[1:k]
res[i.img_low]
i.img_high = order(res, decreasing = TRUE)[1:k]
res[i.img_high]

# # Plot
# files = list.files(paste0(dataloc,"test"), recursive = TRUE)
# par(mfrow = c(3,3), mar = c(1,0,1,0)) 
# print("low residuals")
# generator.test$classes[i.img_low]
# for (i in 1:k) {
#   plot(image_read(paste0(dataloc,"test/",files[i.img_low[i]])))
#   title(paste0("Class = ",generator.test$classes[i.img_low[i]],", yhat = ",round(yhat[i.img_low[i]],3)))
#   
# }
# print("high residuals")
# generator.test$classes[i.img_high]
# for (i in 1:k) {
#   plot(image_read(paste0(dataloc,"test/",files[i.img_high[i]])))
#   title(paste0("Class = ",generator.test$classes[i.img_high[i]],", yhat = ",round(yhat[i.img_high[i]],3)))
#   
# }
# #dev.off()



## Create CAM ("class activation map")

# Clear lower level access
#k_clear_session()

i.img = i.img_high
pdf(paste0(plotloc,"residuals_high_model1.pdf"))
par(mfrow = c(4,4), mar = c(1,0,1,0)) 
for (i in 1:k) {
  plot_cam(img_path = paste0(dataloc,"test/",files[i.img[i]]),
           model = model.1,
           layer_name = "conv2d_4",
           titles = c(paste0("Class = ",y_num[i.img[i]]),
                      paste0("yhat = ",round(yhat[i.img[i]],3))),
           target_class = 1 - y_num[i.img[i]])
}
dev.off()

i.img = i.img_low
pdf(paste0(plotloc,"residuals_low_model1.pdf"))
par(mfrow = c(4,4), mar = c(1,0,1,0)) 
for (i in 1:k) {
  plot_cam(img_path = paste0(dataloc,"test/",files[i.img[i]]),
           model = model.1,
           layer_name = "conv2d_4",
           titles = c(paste0("Class = ",y_num[i.img[i]]),
                      paste0("yhat = ",round(yhat[i.img[i]],3))),
           target_class = y_num[i.img[i]])
}
dev.off()


#######################################################################################################################-
#|||| Feature extraction 1 (WITHOUT data augmentation) ||||----
#######################################################################################################################-

## Get pretrained vgg16 convbase
conv_base <- application_densenet121(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)
conv_base



## Create feature matrix

# Get output from last layer
extract_features = function(directory = paste0(dataloc,"train"), 
                            sample_count = 2000, size = 100, last_layer_size = 4 * 4 * 512) {
  
  # Initialize x and y
  a.x = array(0, dim = c(sample_count, last_layer_size))  
  a.y = array(0, dim = c(sample_count))
  
  # Define generator
  generator = flow_images_from_directory(
    directory,  
    image_data_generator(rescale = 1/255),  
    target_size = c(150, 150),  
    batch_size = size,  
    class_mode = "binary")
  
  # Run through images and fill featurs and y
  i = 1
  while(i * size <= sample_count) {
    print(i * size)
    
    batch = generator_next(generator)
    index_range = ((i-1) * size +1):(i * size)
    a.x[index_range,] = conv_base %>% 
      predict(batch[[1]]) %>% #Get last layer
      array_reshape(dim = c(size, last_layer_size)) #Flatten
    a.y[index_range] <- batch[[2]]
    
    i = i + 1
  }
  
  list(x = a.x, y = a.y)
}
train = extract_features(paste0(dataloc,"train"), n.train)
validate = extract_features(paste0(dataloc,"validate"), n.validate)
test = extract_features(paste0(dataloc,"test"), n.test)
l.save = list("train" = train, "validate" = validate, "test" = test)
save(l.save, file = paste0(type,"_features.RDATA"))


## Train small model on x (could also be done with other ML algorithm)
model.2 = keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", 
              input_shape = 4 * 4 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = optimizer_rmsprop(lr = 1e-5),
    loss = "binary_crossentropy",
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
y_num = test$y
y = factor(ifelse(y_num == 0, "N", "Y"))
performance_summary(data.frame(yhat = data.frame("Y"=yhat, "N"=1-yhat), y = y))
plots = plot_all_performances(yhat = data.frame("Y"=yhat, "N"=1-yhat), y = y) 
ggsave(paste0(plotloc, "model.2_performance.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)
# Save
model.2 %>% save_model_hdf5(paste(type,"_model2.h5"))
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
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    loss = "binary_crossentropy",
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
y_num = as.vector(generator.test$classes)
y = factor(ifelse(y_num == 0, "N", "Y"))
performance_summary(data.frame(yhat = data.frame("Y"=yhat, "N"=1-yhat), y = y))
plots = plot_all_performances(yhat = data.frame("Y"=yhat, "N"=1-yhat), y = y) 
ggsave(paste0(plotloc, "model.3_performance.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)

# Save
model.3 %>% save_model_hdf5(paste(type,"_model3.h5"))
#model.3 = load_model_hdf5(paste(type,"_model3.h5"))




#######################################################################################################################-
#|||| Fine Tuning ||||----
#######################################################################################################################-

# Unfreeze: IMPORTANT to first fit model.3
unfreeze_weights(conv_base, from = "block3_conv1")
model.4 = model.3
  
# Fit
model.4 %>% compile(
  loss = "binary_crossentropy",
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
y_num = as.vector(generator.test$classes)
y = factor(ifelse(y_num == 0, "N", "Y"))
performance_summary(data.frame(yhat = data.frame("Y"=yhat, "N"=1-yhat), y = y))
plots = plot_all_performances(yhat = data.frame("Y"=yhat, "N"=1-yhat), y = y) 
ggsave(paste0(plotloc, "model.4_performance.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)
# Save
model.4 %>% save_model_hdf5(paste(type,"_model4.h5"))
#model.4 = load_model_hdf5(paste(type,"_model4.h5"))



#######################################################################################################################-
#|||| Fine Tuning with interpretation ||||----
#######################################################################################################################-

# Remove last conv_layer + maxpool_layer from conv_base: !!! RUN THIS after model.3 to just restimate last 2 layer
tmp = keras_model(conv_base$inputs, conv_base$layers[[14]]$output)
tmp
freeze_weights(tmp)

# Enlarge with dense layers
model.5 = keras_model_sequential() %>% 
  tmp %>% 
  layer_conv_2d(filters = 512,  kernel_size = c(3, 3), padding = "same", activation = "relu") %>% 
  layer_conv_2d(filters = 512,  kernel_size = c(3, 3), padding = "same", activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(lr = 2e-5),
    metrics = c("acc")
  )
model.5
fit.5 = model.5 %>% fit_generator(
  generator.train,
  steps_per_epoch = n.train/batchsize, 
  epochs = 20, 
  validation_data = generator.validate,
  validation_steps = n.validate/batchsize 
)
plot(fit.5)
model.5 %>% evaluate_generator(generator.test, steps = n.test/batchsize)
yhat = predict_generator(model.5, generator.test, steps = n.test/batchsize)
y_num = as.vector(generator.test$classes)
y = factor(ifelse(y_num == 0, "N", "Y"))
performance_summary(data.frame(yhat = data.frame("Y"=yhat, "N"=1-yhat), y = y))
plots = plot_all_performances(yhat = data.frame("Y"=yhat, "N"=1-yhat), y = y) 
ggsave(paste0(plotloc, "model.5_performance.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)

# Save
#model.5 %>% save_model_hdf5(paste(type,"_model5.h5"))
#model.5 = load_model_hdf5(paste(type,"_model5.h5"))


# Interpret --------------------------------------------------------------------------

## Plot images with low and high residuals
# Get residuals
res = abs(yhat - y_num)
order(res)
k = 16
(i.img_low = order(res)[10+(1:k)])
res[i.img_low]
(i.img_high = order(res, decreasing = TRUE)[0+(1:k)])
res[i.img_high]


## Plot CAM

# High residuals
i.img = i.img_high
pdf(paste0(plotloc,"residuals_high_model5.pdf"))
par(mfrow = c(4,4), mar = c(1,0,1,0)) 
for (i in 1:k) {
  plot_cam(img_path = paste0(dataloc,"test/",files[i.img[i]]),
           model = model.5,
           layer_name = "conv2d_6",
           titles = c(paste0("Class = ",y_num[i.img[i]]),
                      paste0("yhat = ",round(yhat[i.img[i]],3))),
           target_class = 1 - y_num[i.img[i]]) #target_class of prediction for high residuals
}
dev.off()

i.img = i.img_low
pdf(paste0(plotloc,"residuals_low_model5.pdf"))
par(mfrow = c(4,4), mar = c(1,0,1,0)) 
for (i in 1:k) {
  plot_cam(img_path = paste0(dataloc,"test/",files[i.img[i]]),
           model = model.5,
           layer_name = "conv2d_6",
           titles = c(paste0("Class = ",y_num[i.img[i]]),
                      paste0("yhat = ",round(yhat[i.img[i]],3))),
           target_class = y_num[i.img[i]])
}
dev.off()




#save.image(paste0(type,"_cats_vs_dogs.RData"))
