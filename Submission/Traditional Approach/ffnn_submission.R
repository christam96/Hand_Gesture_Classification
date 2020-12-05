# Imports
library(keras)
library(tensorflow)
library(plotly)
library(igraph)
library(BatchGetSymbols)
require(Metrics)
require(DMwR)

########################################
# PREPARING THE DATA
########################################

# Read in the data
test <- read.csv("~/Desktop/Neural Networks Gesture Classification/test_set.csv")
train <- read.csv("~/Desktop/Neural Networks Gesture Classification/training_set.csv")
valid <- read.csv("~/Desktop/Neural Networks Gesture Classification/validation_set.csv")

# Standardize the inputs
colMeans = colMeans(train) 	# column means
col_stdev <- apply(train, 2, sd) # standard deviation
trainNorm <- scale(train, center=colMeans, scale=col_stdev )
valNorm <- scale(valid, center=colMeans, scale=col_stdev )
testNorm <- scale(test,  center=colMeans, scale=col_stdev)

# Split x and y components of the data
xtrain = trainNorm[,1:56]
ytrain_target = train[,57]
xtest = testNorm[,1:56]
ytest_target = test[,57]
xval = valNorm[,1:56]
yval_target = valid[,57]

# One-hot encode y's
yval = valid[,58:61]
yval = data.matrix(yval) 
ytrain = train[,58:61]
ytrain = data.matrix(ytrain)
ytest = test[,58:61]
ytest = data.matrix(ytest)

########################################
# TUNING THE NETWORK STRUCTURE
########################################
numEpoch = 120

# Total number of layers: 2
# Neurons structure: 56-4
soft_2 <- keras_model_sequential() #initialize
soft_2 %>%
  layer_dense(units = ncol(xtrain), activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

soft_2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = 0.1), #lr is learning rate
  metrics = 'accuracy'
)
history <- soft_2 %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = numEpoch,
  batch_size = 311,
  validation_data = list(xval,yval),
)

# Total number of layers: 3
# Neurons structure: 56-4-4
soft_3_4 <- keras_model_sequential() #initialize
soft_3_4 %>%
  layer_dense(units = ncol(xtrain), activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

soft_3_4 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = 0.1), #lr is learning rate
  metrics = 'accuracy'
)

history <- soft_3_4 %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = numEpoch,
  batch_size = 311,
  validation_data = list(xval,yval),
)

# Total number of layers: 3
# Neurons structure: 56-8-4
soft_3_8 <- keras_model_sequential() #initialize
soft_3_8 %>%
  layer_dense(units = ncol(xtrain), activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 8, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

soft_3_8 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = 0.1), #lr is learning rate
  metrics = 'accuracy'
)

history <- soft_3_8 %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = numEpoch,
  batch_size = 311,
  validation_data = list(xval,yval),
)


# Total number of layers: 3
# Neurons structure: 56-12-4
soft_3_12 <- keras_model_sequential() #initialize
soft_3_12 %>%
  layer_dense(units = ncol(xtrain), activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 12, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

soft_3_12 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = 0.1), #lr is learning rate
  metrics = 'accuracy'
)

history <- soft_3_12 %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = numEpoch,
  batch_size = 311,
  validation_data = list(xval,yval),
)

# Total number of layers: 3
# Neurons structure: 56-28-4
soft_3_28 <- keras_model_sequential() #initialize
soft_3_28 %>%
  layer_dense(units = ncol(xtrain), activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 28, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

soft_3_28 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = 0.1), #lr is learning rate
  metrics = 'accuracy'
)

history <- soft_3_28 %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = numEpoch,
  batch_size = 311,
  validation_data = list(xval,yval),
)

# Total number of layers: 3
# Neurons structure: 56-100-4
soft_3_100 <- keras_model_sequential() #initialize
soft_3_100 %>%
  layer_dense(units = ncol(xtrain), activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 100, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

soft_3_100 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = 0.1), #lr is learning rate
  metrics = 'accuracy'
)

history <- soft_3_100 %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = numEpoch,
  batch_size = 311,
  validation_data = list(xval,yval),
)

# Total number of layers: 4
# Neurons structure: 56-4-4-4
soft_4_4_4 <- keras_model_sequential() #initialize
soft_4_4_4 %>%
  layer_dense(units = ncol(xtrain), activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

soft_4_4_4 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = 0.1), #lr is learning rate
  metrics = 'accuracy'
)

history <- soft_4_4_4 %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = numEpoch,
  batch_size = 311,
  validation_data = list(xval,yval),
)

# Total number of layers: 4
# Neurons structure: 56-12-8-4
soft_4_8_12 <- keras_model_sequential() #initialize
soft_4_8_12 %>%
  layer_dense(units = ncol(xtrain), activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 12, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 8, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

soft_4_8_12 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = 0.1), #lr is learning rate
  metrics = 'accuracy'
)

history <- soft_4_8_12 %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = numEpoch,
  batch_size = 311,
  validation_data = list(xval,yval),
)

# Total number of layers: 4
# Neurons structure: 56-28-14-4
soft_4_14_28 <- keras_model_sequential() #initialize
soft_4_14_28 %>%
  layer_dense(units = ncol(xtrain), activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 28, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 14, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

soft_4_14_28 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = 0.1), #lr is learning rate
  metrics = 'accuracy'
)

history <- soft_4_14_28 %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = numEpoch,
  batch_size = 311,
  validation_data = list(xval,yval),
)

yval_pred <- soft_4_14_28 %>% predict_classes(xval)
yval_pred

# Total number of layers: 4
# Neurons structure: 56-56-28-4
soft_4_28_56 <- keras_model_sequential() #initialize
soft_4_28_56 %>%
  layer_dense(units = ncol(xtrain), activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 56, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 28, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

soft_4_28_56 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = 0.1), #lr is learning rate
  metrics = 'accuracy'
)

history <- soft_4_28_56 %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = numEpoch,
  batch_size = 311,
  validation_data = list(xval,yval),
)

# Total number of layers: 4
# Neurons structure: 56-112-56-4
soft_4_56_112 <- keras_model_sequential() #initialize
soft_4_56_112 %>%
  layer_dense(units = ncol(xtrain), activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 112, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 56, activation = 'sigmoid', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

soft_4_56_112 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(lr = 0.1), #lr is learning rate
  metrics = 'accuracy'
)

history <- soft_4_56_112 %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = numEpoch,
  batch_size = 311,
  validation_data = list(xval,yval),
)

########################################
# FURTHER TUNING OF THE MODEL (GRID SEARCH)
########################################
# From previous trials with default settings and non-recorded trials, 120 epoch is sufficient

# All models will:
#   - Use softmax for the output layer, as standard
#   - Use L2 regulation to help prevent overfitting
#   - Use default initialization setting

# The parameters to be tuned are the learning rate, learning function, and activation function
# Two network architectures will be tried - 2 layers and 3 layers
# The three layer one will have 4 neurons in the hidden layer because of the formula
#   (a should be between 5-10, resulting in 3-6 neurons. Choose 4.)

# Store the results in two 3D matrices - 1 for the 2 layer network, 1 for the 3 layer network
# j == learning rate (0.001,0.01,0.05,0.1,0.2,0.3,0.4)
# k == activation function (relu, elu, sigmoid)

# Tuning space
learn_fn = c("sgd","adam")
learn_rate = c(0.001,0.01,0.05,0.1,0.2,0.3,0.4) #7 options
act_fn = c("relu","elu","sigmoid") #3 options
numEpoch = 120

# Results matrix
tuning_2 <- array(data = NA, dim = c(2,length(learn_rate),length(act_fn)))
tuning_3 <- array(data = NA, dim = c(2,length(learn_rate),length(act_fn)))

# Sgd learning function
for (j in 1:length(learn_rate)){
  for(k in 1:length(act_fn)){
    
    learn_rate[j]
    act_fn[k]
    
    #make and train the 2 layer model
    tuning_model <- keras_model_sequential() #initialize
    tuning_model %>%
      layer_dense(units = ncol(xtrain), activation = act_fn[k], kernel_regularizer = regularizer_l2(0.01))%>%
      layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))
    
    tuning_model %>% compile(
      loss = 'categorical_crossentropy',
      optimizer = optimizer_sgd(learn_rate[j]), #lr is learning rate
      metrics = 'accuracy'
    )
    
    history <- tuning_model %>% fit(
      x = xtrain,
      y = ytrain,
      epochs = numEpoch,
      batch_size = 311,
      validation_data = list(xval,yval),
    )
    
    #get the accuracy of the prediction for the 2 layer one
    yval_pred <- tuning_model %>% predict_classes(xval)
    yval_pred
    pred = yval_pred
    target = yval_target
    
    numCor = 0
    for(i in 1:length(pred)){
      if(pred[i]==target[i]){numCor = numCor+1}
    }
    perCor = numCor/length(pred)*100
    tuning_2[1,j,k] = perCor
    
    #make and train the 3 layer model
    tuning_model <- keras_model_sequential() #initialize
    tuning_model %>%
      layer_dense(units = ncol(xtrain), activation = act_fn[k], kernel_regularizer = regularizer_l2(0.01))%>%
      layer_dense(units = 4, activation = act_fn[k], kernel_regularizer = regularizer_l2(0.01))
    layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))
    
    tuning_model %>% compile(
      loss = 'categorical_crossentropy',
      optimizer = optimizer_sgd(learn_rate[j]), #lr is learning rate
      metrics = 'accuracy'
    )
    
    history <- tuning_model %>% fit(
      x = xtrain,
      y = ytrain,
      epochs = numEpoch,
      batch_size = 311,
      validation_data = list(xval,yval),
    )
    
    #get the accuracy of the prediction for the 2 layer one
    yval_pred <- tuning_model %>% predict_classes(xval)
    yval_pred
    pred = yval_pred
    target = yval_target
    
    numCor = 0
    for(i in 1:length(pred)){
      if(pred[i]==target[i]){numCor = numCor+1}
    }
    perCor = numCor/length(pred)*100
    tuning_3[1,j,k] = perCor
    
  }
}# End of sgd

# Adam learning function
for (j in 1:length(learn_rate)){
  for(k in 1:length(act_fn)){
    
    #make and train the 2 layer model
    tuning_model <- keras_model_sequential() #initialize
    tuning_model %>%
      layer_dense(units = ncol(xtrain), activation = act_fn[k], kernel_regularizer = regularizer_l2(0.01))%>%
      layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))
    
    tuning_model %>% compile(
      loss = 'categorical_crossentropy',
      optimizer = optimizer_adam(learn_rate[j]), #lr is learning rate
      metrics = 'accuracy'
    )
    
    history <- tuning_model %>% fit(
      x = xtrain,
      y = ytrain,
      epochs = numEpoch,
      batch_size = 311,
      validation_data = list(xval,yval),
    )
    
    #get the accuracy of the prediction for the 2 layer one
    yval_pred <- tuning_model %>% predict_classes(xval)
    yval_pred
    pred = yval_pred
    target = yval_target
    
    numCor = 0
    for(i in 1:length(pred)){
      if(pred[i]==target[i]){numCor = numCor+1}
    }
    perCor = numCor/length(pred)*100
    tuning_2[2,j,k] = perCor
    
    #make and train the 3 layer model
    tuning_model <- keras_model_sequential() #initialize
    tuning_model %>%
      layer_dense(units = ncol(xtrain), activation = act_fn[k], kernel_regularizer = regularizer_l2(0.01))%>%
      layer_dense(units = 4, activation = act_fn[k], kernel_regularizer = regularizer_l2(0.01))
    layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))
    
    tuning_model %>% compile(
      loss = 'categorical_crossentropy',
      optimizer = optimizer_adam(learn_rate[j]), #lr is learning rate
      metrics = 'accuracy'
    )
    
    history <- tuning_model %>% fit(
      x = xtrain,
      y = ytrain,
      epochs = numEpoch,
      batch_size = 311,
      validation_data = list(xval,yval),
    )
    
    #get the accuracy of the prediction for the 2 layer one
    yval_pred <- tuning_model %>% predict_classes(xval)
    yval_pred
    pred = yval_pred
    target = yval_target
    
    numCor = 0
    for(i in 1:length(pred)){
      if(pred[i]==target[i]){numCor = numCor+1}
    }
    perCor = numCor/length(pred)*100
    tuning_3[2,j,k] = perCor
    
  }
} # End of Adam

########################################
# PLOT THE RESULTS
########################################
plot_ly(x = learn_rate, y = learn_fn, z = tuning_2[,,1], type = "heatmap")
plot_ly(x = learn_rate, y = learn_fn, z = tuning_2[,,2], type = "heatmap")
plot_ly(x = learn_rate, y = learn_fn, z = tuning_2[,,3], type = "heatmap")
plot_ly(x = learn_rate, y = learn_fn, z = tuning_3[,,1], type = "heatmap")
plot_ly(x = learn_rate, y = learn_fn, z = tuning_3[,,2], type = "heatmap")
plot_ly(x = learn_rate, y = learn_fn, z = tuning_3[,,3], type = "heatmap")

########################################
# ANALYSIS OF BEST PERFORMING MODEL
########################################
# Best performer: 2 layer, relu, 0.2, sgd

# Recreate and train the best model
final_model <- keras_model_sequential() #initialize
final_model %>%
  layer_dense(units = ncol(xtrain), activation = 'relu', kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dense(units = 4, activation = 'softmax', kernel_regularizer = regularizer_l2(0.01))

final_model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_sgd(0.02), #lr is learning rate
  metrics = 'accuracy'
)

history <- final_model %>% fit(
  x = xtrain,
  y = ytrain,
  epochs = 150,
  batch_size = 311,
  validation_data = list(xval,yval),
)

# Get accuracy and prediction
yval_pred <- final_model %>% predict_classes(xval)
yval_pred
pred = yval_pred
target = yval_target

numCor = 0
for(i in 1:length(pred)){
  if(pred[i]==target[i]){numCor = numCor+1}
}
perCor = numCor/length(pred)*100
perCor

# Get the table for the validation data
table <- array(data = 0, dim = c(4,4))
pred = yval_pred
target = yval_target

# Rows are what the data should be, columns are what the data was predicted to be
for (i in 1:length(pred)){
  if(target[i]==0&&pred[i]==0){table[1,1] = table[1,1] + 1}
  if(target[i]==0&&pred[i]==1){table[1,2] = table[1,2] + 1}
  if(target[i]==0&&pred[i]==2){table[1,3] = table[1,3] + 1}
  if(target[i]==0&&pred[i]==3){table[1,4] = table[1,4] + 1}
  
  if(target[i]==1&&pred[i]==0){table[2,1] = table[2,1] + 1}
  if(target[i]==1&&pred[i]==1){table[2,2] = table[2,2] + 1}
  if(target[i]==1&&pred[i]==2){table[2,3] = table[2,3] + 1}
  if(target[i]==1&&pred[i]==3){table[2,4] = table[2,4] + 1}
  
  if(target[i]==2&&pred[i]==0){table[3,1] = table[3,1] + 1}
  if(target[i]==2&&pred[i]==1){table[3,2] = table[3,2] + 1}
  if(target[i]==2&&pred[i]==2){table[3,3] = table[3,3] + 1}
  if(target[i]==2&&pred[i]==3){table[3,4] = table[3,4] + 1}
  
  if(target[i]==3&&pred[i]==0){table[4,1] = table[4,1] + 1}
  if(target[i]==3&&pred[i]==1){table[4,2] = table[4,2] + 1}
  if(target[i]==3&&pred[i]==2){table[4,3] = table[4,3] + 1}
  if(target[i]==3&&pred[i]==3){table[4,4] = table[4,4] + 1}
}
table

# Get the stats for the test data
ytest_pred <- final_model %>% predict_classes(xtest)
ytest_pred
pred = ytest_pred
target = ytest_target
numCor = 0
for(i in 1:length(pred)){
  if(pred[i]==target[i]){numCor = numCor+1}
}
perCor = numCor/length(pred)*100
perCor

# Rows are what the data should be, columns are what the data was predicted to be
table <- array(data = 0, dim = c(4,4))
for (i in 1:length(pred)){
  if(target[i]==0&&pred[i]==0){table[1,1] = table[1,1] + 1}
  if(target[i]==0&&pred[i]==1){table[1,2] = table[1,2] + 1}
  if(target[i]==0&&pred[i]==2){table[1,3] = table[1,3] + 1}
  if(target[i]==0&&pred[i]==3){table[1,4] = table[1,4] + 1}
  
  if(target[i]==1&&pred[i]==0){table[2,1] = table[2,1] + 1}
  if(target[i]==1&&pred[i]==1){table[2,2] = table[2,2] + 1}
  if(target[i]==1&&pred[i]==2){table[2,3] = table[2,3] + 1}
  if(target[i]==1&&pred[i]==3){table[2,4] = table[2,4] + 1}
  
  if(target[i]==2&&pred[i]==0){table[3,1] = table[3,1] + 1}
  if(target[i]==2&&pred[i]==1){table[3,2] = table[3,2] + 1}
  if(target[i]==2&&pred[i]==2){table[3,3] = table[3,3] + 1}
  if(target[i]==2&&pred[i]==3){table[3,4] = table[3,4] + 1}
  
  if(target[i]==3&&pred[i]==0){table[4,1] = table[4,1] + 1}
  if(target[i]==3&&pred[i]==1){table[4,2] = table[4,2] + 1}
  if(target[i]==3&&pred[i]==2){table[4,3] = table[4,3] + 1}
  if(target[i]==3&&pred[i]==3){table[4,4] = table[4,4] + 1}
}
table

# Create results CSV 
write.csv(ytest_pred,"C:/Users/Admin/Desktop/FFNN_test_predicted.csv")
write.csv(ytest_target,"C:/Users/Admin/Desktop/FFNN_test_true.csv")

