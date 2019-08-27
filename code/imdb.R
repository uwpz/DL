
#######################################################################################################################-
#|||| Initialize and ETL ||||----
#######################################################################################################################-

# Libraries
library(keras)
library(hashFunction)

# Functions
source("code/0_init.R")

# Get data
imdb <- dataset_imdb(num_words = 10000) # Take only top 10000 words
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb
max(map_int(train_data, ~ max(.)))
quantile(map_int(train_data, ~ length(.)), seq(0,1,0.01))




#######################################################################################################################-
#|||| First model from one-hot encoding ||||----
#######################################################################################################################-

# Prepare --------------------------------------------------------------------------

# Get "data", i.e. index
word_index <- dataset_imdb_word_index()
word_index[1:3]
decod = c(rep("?", 3), names(word_index))
names(decod) = c(c(-2,-1,0), as.numeric(word_index))
decod[1:6]
decod[as.character(1:10)]
n = 14
paste(decod[as.character(train_data[[n]] - 3)], collapse = " ")

# One-hot encode
vectorize_sequences <- function(sequences, dimension = 10000) {
  # Create an all-zero matrix of shape (len(sequences), dimension)
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1
  results
}
x_train <- vectorize_sequences(train_data)
y_train = train_labels
x_test <- vectorize_sequences(test_data)
y_test = test_labels



# Fit --------------------------------------------------------------------------

# Model definition
model.1 <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )

# Tune
i.train = sample(1:length(y_train), 10e3)
fit = model.1 %>% fit(
  x_train[i.train,], y_train[i.train], 
  epochs = 10, 
  batch_size = 512,
  validation_data = list(x_train[-i.train,], y_train[-i.train])
  )
plot(fit)

# Train
fit.final = model.1 %>% fit(x_train, y_train, epochs = 4, batch_size = 512)   
plot(fit.final)

# Evaluate
results <- model.1 %>% evaluate(x_test, y_test)
results
yhat = predict(model.1, x_test)
y = factor(ifelse(y_test == 0, "N", "Y"))
mysummary_class(data.frame(yhat = yhat, y = y))
plots = get_plot_performance_class(yhat, y)
ggsave(paste0(plotloc,"imdb","model.1_performance.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)


#######################################################################################################################-
#|||| With embedding layer ||||----
#######################################################################################################################-

# Prepare --------------------------------------------------------------------------

# Get data
read_imdb = function(fold) {
  labels <- c()
  texts <- c()
  for (label_type in c("neg", "pos")) {
    label <- switch(label_type, neg = 0, pos = 1)
    dir_name <- file.path("./data/imdb/",fold, label_type)
    for (fname in list.files(dir_name, pattern = glob2rx("*.txt"), 
                             full.names = TRUE)) {
      texts <- c(texts, readChar(fname, file.info(fname)$size))
      labels <- c(labels, label)
    }
  }
  list(texts, labels)
}
c(texts,labels) %<-% read_imdb("train")

# Tokenize
max_words <- 10000  # Take only top 10000 words          
tokenizer <- text_tokenizer(num_words = max_words) %>% 
  fit_text_tokenizer(texts)
sequences <- texts_to_sequences(tokenizer, texts)
word_index = tokenizer$word_index
cat("Found", length(word_index), "unique tokens.\n")

# Pad
maxlen <- 1000 # We will cut reviews after 100 words
data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(labels)
cat("Shape of data tensor:", dim(data), "\n")



# Fit --------------------------------------------------------------------------

# Model definition
embedding_dim <- 100
model.2 <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, 
                  input_length = maxlen) %>% 
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>%
  compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
  )

# Tune
set.seed(999)
i.train = sample(1:length(labels), 10e3)
fit <- model.2 %>% fit(
  data[i.train,], labels[i.train],
  epochs = 10,
  batch_size = 512,
  validation_data = list(data[-i.train,], labels[-i.train])
)
plot(fit)

# Train
fit.final = model.2 %>% fit(data, labels, epochs = 4, batch_size = 512)   
plot(fit.final)

# Evaluate
c(texts_test, labels_test)  %<-% read_imdb("test")
x_test = pad_sequences(texts_to_sequences(tokenizer, texts_test), maxlen = maxlen)
y_test = as.array(labels_test)
results <- model.2 %>% evaluate(x_test, y_test)
results
yhat = predict(model.2, x_test)
y = factor(ifelse(y_test == 0, "N", "Y"))
mysummary_class(data.frame(yhat = yhat, y = y))
plots = get_plot_performance_class(yhat, y)
ggsave(paste0(plotloc,"imdb","model.2_performance.pdf"), marrangeGrob(plots, ncol = 4, nrow = 2, top = NULL), 
       w = 18, h = 12)




#######################################################################################################################-
#|||| With glove embedding ||||----
#######################################################################################################################-

# Prepare --------------------------------------------------------------------------

lines <- readLines("./data/imdb/glove.6B.100d.txt")
embeddings_index <- new.env(hash = TRUE, parent = emptyenv())
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}

embedding_matrix <- array(0, c(max_words, embedding_dim))
for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
      # Words not found in the embedding index will be all zeros.
      embedding_matrix[index+1,] <- embedding_vector
  }
}


# Fit --------------------------------------------------------------------------

# Freeze weights by glove embedding and compile again
model.3 <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, 
                  input_length = maxlen) %>% 
  layer_flatten() %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") 
get_layer(model.3, index = 1) %>% 
  set_weights(list(embedding_matrix)) %>% 
  freeze_weights()
model.3 %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("acc")
  )

# Tune
fit <- model.3 %>% fit(
  data[i.train,], labels[i.train],
  epochs = 30,
  batch_size = 512,
  validation_data = list(data[-i.train,], labels[-i.train])
)
plot(fit)

# Train
fit.final = model.3 %>% fit(data, labels, epochs = 4, batch_size = 512)   
plot(fit.final)

# Evaluate
results <- model.3 %>% evaluate(x_test, y_test)
results




#######################################################################################################################-
#|||| RNN ||||----
#######################################################################################################################-

# Prepare & fit  --------------------------------------------------------------------------

input_train <- pad_sequences(train_data, maxlen = maxlen)
input_test <- pad_sequences(test_data, maxlen = maxlen)

model.4 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = 100) %>%
  #layer_lstm(units = 32) %>%
  bidirectional(
    layer_gru(units = 64, dropout = 0.2, recurrent_dropout = 0.2) 
  ) %>%
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(
    optimizer =  optimizer_rmsprop(lr = 5e-4),
    loss = "binary_crossentropy",
    metrics = c("acc")
  )
fit <- model.4 %>% fit(
  input_train, train_labels,
  epochs = 10,
  batch_size = 256,
  validation_split = 0.2
)



#######################################################################################################################-
#|||| 1D-convnet ||||----
#######################################################################################################################-

# Prepare & fit  --------------------------------------------------------------------------

model.5 <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, output_dim = 100,
                  input_length = maxlen) %>% 
  layer_conv_1d(filters = 16, kernel_size = 5, activation = "relu") %>% 
  layer_max_pooling_1d(pool_size = 5) %>% 
  layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu") %>% 
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 1) %>% 
  compile(
    optimizer =  optimizer_rmsprop(lr = 5e-5),
    loss = "binary_crossentropy",
    metrics = c("acc")
  )

# tensorboard("./output/my_log_dir")
# callbacks = list(callback_tensorboard(
#   log_dir = "./output/my_log_dir",
#   histogram_freq = 1,
#   embeddings_freq = 1
# ))

fit <- model.5 %>% fit(
  input_train, train_labels,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2#,
  #callbacks = callbacks
)
