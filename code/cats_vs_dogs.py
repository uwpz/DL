
#######################################################################################################################-
#|||| Initialize and ETL ||||----
#######################################################################################################################-

#cmd: "nvidia-smi -l" to monitor gpu-usage

# General libraries, parameters and functions
import os
import sys
sys.path.append(os.getcwd() + "\\code") #not needed if code is marked as "source" in pycharm
from initialize import *

# Specific libraries
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers
from keras.models import load_model
from keras.models import clone_model
from keras.applications import VGG16
import vis
from vis.visualization import visualize_cam
import numpy as np
import matplotlib.pyplot as plt

# Parameter
type = "small"
dataloc = "./data/cats_vs_dogs/" + type + "/"
n_train = len([name for path, subdirs, files in os.walk(dataloc + "train") for name in files])
n_validate = len([name for path, subdirs, files in os.walk(dataloc + "validate") for name in files])
n_test = len([name for path, subdirs, files in os.walk(dataloc + "test") for name in files])
batchsize = 20
targetsize = (300, 300)


# ######################################################################################################################
# Prepare
# ######################################################################################################################

# --- Validate ------------------------------------------------------------------------------------
generator_validate = ImageDataGenerator(rescale=1/255).flow_from_directory(
    dataloc + "validate",
    target_size=targetsize,
    batch_size=batchsize,
    class_mode="binary")

x_batch, y_batch = next(generator_validate)
fig, ax = plt.subplots(2,2)
for i in np.arange(4):
    ax_act = ax.flat[i]
    ax_act.imshow(image.array_to_img(x_batch[i]))
fig.tight_layout()


# --- Test: no shuffle ------------------------------------------------------------------------------------
generator_test = ImageDataGenerator(rescale=1/255).flow_from_directory(
    dataloc + "test",
    target_size=targetsize,
    batch_size=batchsize,
    class_mode="binary",
    shuffle=False) #no shuffle !!!


# --- Train: data augmentation ------------------------------------------------------------------------------------
datagen_augment = ImageDataGenerator(
  rescale=1/255,
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode="nearest"
)
# Check augmentation
x_batch_orig, y_batch_orig = next(generator_test)
for i in np.arange(x_batch_orig.shape[0]):
    x_batch_orig[i,:,:,:] = x_batch_orig[0,:,:,:]
x_batch = 255 * next(datagen_augment.flow(x_batch_orig, batch_size=batchsize))
fig, ax = plt.subplots(2,2)
for i in np.arange(4):
    ax_act = ax.flat[i]
    img = x_batch_orig[0,:,:,:] if i == 0 else x_batch[i,:,:,:]
    ax_act.imshow(image.array_to_img(img))
fig.tight_layout()

# Generator
generator_train = datagen_augment.flow_from_directory(
    dataloc + "train",
    target_size=targetsize,
    batch_size=batchsize,
    class_mode="binary"
)



# ######################################################################################################################
# Small convnet
# ######################################################################################################################

# --- Fit --------------------------------------------------------------------------

# Model definition
model1 = models.Sequential()
model1.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=targetsize + (3,)))
model1.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model1.add(layers.MaxPool2D(pool_size=(2, 2)))
model1.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model1.add(layers.MaxPool2D(pool_size=(2, 2)))
model1.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu"))
model1.add(layers.MaxPool2D(pool_size=(2, 2)))
model1.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
model1.add(layers.MaxPool2D(pool_size=(2, 2)))
model1.add(layers.Flatten())
model1.add(layers.Dropout(rate=0.5))
model1.add(layers.Dense(units=512, activation="relu"))
model1.add(layers.Dense(units=1, activation="sigmoid"))
model1.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])
model1.summary()

# Fit
fit1 = model1.fit_generator(
    generator_train,
    steps_per_epoch=n_train/batchsize,
    # initial_epoch = 1, #must be less than epoch
    epochs=15,
    validation_data=generator_validate,
    validation_steps=n_validate/batchsize,
    verbose=2
)
plot_fit(fit1)

# Evaluate
print("[loss, acc]:", model1.evaluate_generator(generator_test, steps=n_test/batchsize))
tmp = model1.predict_generator(generator_test, steps=n_test/batchsize)[:,0]
y = generator_test.classes
yhat = np.stack((1-tmp, tmp), axis=1)
plot_all_performances(y, yhat,
                      w=12, h=8, pdf=plotloc + "model1_performance.pdf")

# Save
model1.save(type + "_model1_python.h5")
# model1 = load_model(type + "_model1_python.h5")


# Interpret -------------------------------------------------------------------------------------------------------

# Plot images with low and high residuals
res = np.abs(yhat[:, 1] - y)
res_order = np.argsort(res)
k = 9
i_img_low = res_order[:k]
print(res[i_img_low])
i_img_high = res_order[-k:][::-1]
print(res[i_img_high])

# High Residuals
plot_cam(model=model1, img_path=dataloc + "test/", img_idx=i_img_high, yhat=yhat, y=y,
         ncol=4, nrow=3, w=12, h=8, pdf=plotloc + "res_high_model1.pdf")

# Low Residuals
plot_cam(model=model1, img_path=dataloc + "test/", img_idx=i_img_low, yhat=yhat, y=y,
         ncol=4, nrow=3, w=12, h=8, pdf=plotloc + "res_low_model1.pdf")


# ######################################################################################################################
# Feature extraction (with data augmentation)
# ######################################################################################################################

# Get pretrained vgg16 convbase
conv_base = VGG16(
  weights="imagenet",
  include_top=False,
  input_shape=targetsize + (3,)
)
conv_base.summary()

# Freeze conv_base
conv_base.trainable = False

# Enlarge with dense layers
model3 = models.Sequential()
model3.add(conv_base)
model3.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation="relu"))
model3.add(layers.MaxPool2D(pool_size=(2, 2)))
model3.add(layers.Flatten())
model3.add(layers.Dense(units=256, activation="relu"))
model3.add(layers.Dense(units=1, activation="sigmoid"))
model3.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=2e-5),
              #optimizer=optimizers.Adam(lr=2e-5),
              metrics=["acc"])
model3.summary()

# Fit
fit3 = model3.fit_generator(
    generator_train,
    steps_per_epoch=n_train/batchsize,
    # initial_epoch = 1, #must be less than epoch
    epochs=10,
    validation_data=generator_validate,
    validation_steps=n_validate/batchsize,
    verbose=2
)
plot_fit(fit3)

# Evaluate
print("[loss, acc]:", model3.evaluate_generator(generator_test, steps=n_test/batchsize))
tmp = model3.predict_generator(generator_test, steps=n_test/batchsize)[:,0]
y = generator_test.classes
yhat = np.stack((1-tmp, tmp), axis=1)
plot_all_performances(y, yhat,
                      pdf=plotloc + "model3_performance.pdf")

# Save
model3.save(type + "_model3_python.h5")
# model3 = load_model(type + "_model3_python.h5")


# ######################################################################################################################
# Fine Tuning
# ######################################################################################################################

# IMPORTANT to use model3, because: "First train dense layers (see above) before fine tuning"
model4 = clone_model(model3)
model4.set_weights(model3.get_weights())
conv_base = model4.get_layer("vgg16")

# Freeze all except last 2 conv_layer + maxpool_layer from conv_base
conv_base.trainable = True
set_trainable = False
conv_base.summary()  # get layer name from which to retrain
for layer in conv_base.layers:
    if layer.name == "block4_conv1":
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
model4.compile(loss="binary_crossentropy",
               optimizer=optimizers.RMSprop(lr=1e-5),
               #optimizer=optimizers.Adam(lr=1e-6),
               metrics=["acc"])
model4.summary()

# Fit
fit4 = model4.fit_generator(
    generator_train,
    steps_per_epoch=n_train/batchsize,
    # initial_epoch = 1, #must be less than epoch
    epochs=10,
    validation_data=generator_validate,
    validation_steps=n_validate/batchsize,
    verbose=2
)
plot_fit(fit4)

# Evaluate
print("[loss, acc]:", model4.evaluate_generator(generator_test, steps=n_test/batchsize))
tmp = model4.predict_generator(generator_test, steps=n_test/batchsize)[:,0]
y = generator_test.classes
yhat = np.stack((1-tmp, tmp), axis=1)
plot_all_performances(y, yhat,
                      pdf=plotloc + "model4_performance.pdf")

# Save
model4.save(type + "_model4_python.h5")
# model4 = load_model(type + "_model4_python.h5")


# Interpret -------------------------------------------------------------------------------------------------------

# Plot images with low and high residuals
res = np.abs(yhat[:, 1] - y)
res_order = np.argsort(res)
k = 390
i_img_low = res_order[:k]
print(res[i_img_low])
i_img_high = res_order[-k:][::-1]
print(res[i_img_high])

# High Residuals
plot_cam(model=model4, img_path=dataloc + "test/", img_idx=i_img_high, yhat=yhat, y=y,
         ncol=4, nrow=3, w=12, h=8, pdf=plotloc + "res_high_model4.pdf")

# Low Residuals
plot_cam(model=model4, img_path=dataloc + "test/", img_idx=i_img_low, yhat=yhat, y=y,
         ncol=4, nrow=3, w=12, h=8, pdf=plotloc + "res_low_model4.pdf")
