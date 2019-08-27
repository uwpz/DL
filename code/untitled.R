

# From scratch by tokenizer
samples <- c("The cat sat on the mat.", "The dog ate my homework.")
tokenizer <- text_tokenizer(num_words = 1000) %>%
  fit_text_tokenizer(samples)
sequences <- texts_to_sequences(tokenizer, samples)
word_index <- tokenizer$word_index


## Hashing (= approximating one-hot encoding)
dimensionality <- 1000
max_length <- 10
results <- array(0, dim = c(length(samples), max_length, dimensionality))
for (i in 1:length(samples)) {
  sample <- samples[[i]]
  words <- head(strsplit(sample, " ")[[1]], n = max_length)
  for (j in 1:length(words)) {
    # Hash the word into a "random" integer index
    # that is between 0 and 1,000
    index <- abs(spooky.32(words[[j]])) %% dimensionality
    results[[i, j, index]] <- 1
  }
}
apply(results[1,,], 1, function(x) {which(x>0)})

## 
max_length = 1000
x_train2 <- pad_sequences(train_data, maxlen = max_length)
x_test2 <- pad_sequences(test_data, maxlen = max_length)

model <- keras_model_sequential() %>% 
  # We specify the maximum input length to our Embedding layer
  # so we can later flatten the embedded inputs
  layer_embedding(input_dim = 10000, output_dim = 8, 
                  input_length = max_length) %>% 
  # We flatten the 3D tensor of embeddings 
  # into a 2D tensor of shape `(samples, maxlen * 8)`
  layer_flatten() %>% 
  # We add the classifier on top
  layer_dense(units = 1, activation = "sigmoid") 
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  x_train2, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

