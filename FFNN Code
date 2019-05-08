#install.packages("sigmoid")
#install.packages("neuralnet")
setwd("C:/Users/mfrangos2016/Desktop/Neural Network")
library(sigmoid) #Allows use of sigmoid function, a possible replacement for relu
library(neuralnet)
library(numDeriv)
library(Ryacas)

#                    (# of input nodes, nodes in hidden layer, # of output nodes)
NodeConfiguration = c(2,7,1)
OutputVariables = 2
i=0

#Data Normalization Function (Flattens data from 1 to 0)
normalize.flatten = function(x){
  (x - min(x)) / (max(x) - min(x)) #Normalizes between 0 and 1
}

#Approximate Relu Function
relu = function(x){
  log(1.0 + exp(1)^x) #Normalizes between 0 and 1. exp(1) is number e
}

deriv.relu = function(x){ 1.0 / (1.0 + exp(1)^(-x))}

data = read.csv("Data.csv")
data.normalized = apply(data,2,normalize.flatten) #2 signifies to apply function on columns

#Initiate weight matrix with random weights.
W1 = matrix(rnorm(NodeConfiguration[2]*NodeConfiguration[1],mean=0,sd=(1/3)),nrow = NodeConfiguration[2],ncol = NodeConfiguration[1])
W2 = matrix(rnorm(NodeConfiguration[3]*NodeConfiguration[2],mean=0,sd=(1/3)),nrow = NodeConfiguration[3],ncol = NodeConfiguration[2])

#bias. Initial Setting is random
b1<<-rnorm(NodeConfiguration[2],mean=0,sd=(1/3))
b2<<-rnorm(NodeConfiguration[3],mean=0,sd=(1/3))

#END OF INITIAL SETTINGS (EVERY ITERATION FROM HERE ON IS TRAINING)
################################################################################################################################
iterations = 100000
#INITIATE DATA.FRAME FOR TRACKING ERROR THROUGHOUT THE TRAINING
Sqd.Error_df <- data.frame(
  iteration = 1:iterations,
  Sqd.Error = vector("numeric", length = iterations)
)

for(i in seq_len(iterations)){
  
#Creates an index for random sampling the data
rowindex = sample(1:nrow(data.normalized),1)
a0.descaled = c(data[rowindex,"Aye"],data[rowindex,"Bee"])
a0 <<- c(data.normalized[rowindex,"Aye"],data.normalized[rowindex,"Bee"])
actual = data.normalized[rowindex,"z"]  
#Feed forward functions for Layer 1
Z1 <<- W1%*%a0+b1
a1 <<- relu(Z1) # %*% operator does matrix multiplication. 

#Feed forward functions for output
Z2 <<- W2%*%a1+b2
a2 <<- relu(Z2)


#Function. COPY PASTA THIS INTO THE DEFINE FUNCTION LINE 54
#OriginalFunction = function(actual,a2){
#  (actual - a2)**2
#  }

#Partial Derivatives for backwards propagation
#components for dCdW2 
dC.da2<<-2*(a2-actual)
da2.dz2 <<- deriv.relu(Z2) 
dz2.dW2 <<- a1
dz2.db2 <<- 1

#components for dCdW1
dz2.da1 <<- W2
da1.dW1 <<- a0
da1.db1 <<- 1
#This is how the cost function (error) varies with changes in weights (layer two)
dCdW2 <<- dz2.dW2%*%da2.dz2%*%dC.da2 #Derivative of 
dCdb2 <<- dz2.db2%*%da2.dz2%*%dC.da2
#This is how the cost function (error) varies with changes in weights (layer one)
dCdW1 <<- da1.dW1%*%da2.dz2%*%dC.da2%*%dz2.da1
dCdb1 <<- da1.db1%*%da2.dz2%*%dC.da2%*%dz2.da1
###################################################################################################################################
#Define jacobian of cost function. #Takes two inputs for z in the form of c()
x<<-a2
y<<-actual
DescentStrength <<- 0.001 #Arbitrary number. Too large will overshoot and never find the solution. Too little and you need too much computer power.


#Used for graphing. The diferentials of this equation really do all of the weight adjustments.
Error.Cost.Function <- function(z) { 
  x <- z[1] #Separates
  y <- z[2] #Separates
  #c((actual-a2)^2)#COST FUNCTION - The function to use Jacobian on. Can use c() for two functions.
   c((x     - y)^2) #Same as line above
} 
################################################################################################################################

rotate <- function(x) t(apply(x, 2, rev))


  z <- cbind(x, y)
  #UPDATE WEIGHTS
  W1<<- W1 - DescentStrength*rotate(dCdW1)       
  W2<<- W2 - DescentStrength*rotate(dCdW2)
  b2<<- as.numeric(b2 - DescentStrength*dCdb2)
  b1<<- as.numeric(b1 - DescentStrength*dCdb1)
 
  
 #Print every iteration's error
  print(paste("Squared Error: ", Error.Cost.Function(c(x,y))))
  Sqd.Error_df$Sqd.Error[i] <- Error.Cost.Function(c(x,y))
  print("")
  print(paste("difference:",(actual-a2)))
}
#Graph the error (shows learning over time)
library(ggplot2)  
ggplot(data = Sqd.Error_df, aes(x = iteration, y = Sqd.Error)) +
  geom_line()
#Average Error
print("Average of Last 500 errors")  
mean(Sqd.Error_df[(nrow(Sqd.Error_df)-500):nrow(Sqd.Error_df),"Sqd.Error"])







######################################################################################################
#Now that the model has been created, we can test out some inputs using the below function

# #ofParameters should be the same as number of inputs to neural net.

#Get the max of every column
maxValue = apply(data,2,max)
#Get the min of every column
minValue = apply(data,2,min)
allVars = colnames(data)
TargetVariable = "z"
#Obviously, the # of parameters entered must match the number of input variables.
#FEED PARAMETERS INTO THIS FUNCTION using c(param.1,param.2, etc). For example: Neural.Predict(c(1,2)) will predict the outcome at point (1,2).
Neural.Predict = function(parameters){
  i=1 #Initialize Variable
  normalized.input=matrix(0,ncol=length(parameters), nrow = 1) #Initialize Matrix with correct length (same number as parameters)
  
  while (i<=length(parameters)) {
    normalized.input[,i] = (parameters[i] - minValue[i]) / (maxValue[i] - minValue[i]) #Normalizes the input parameters, crushing them between 0 & 1.
    names(normalized.input) = names(data[,!allVars%in%TargetVariable]) #Sets names of the parameters to equal the variables
    
    i=i+1
  }
  a0 = normalized.input
  Z1 <<- W1%*%as.numeric(a0)+b1
  a1 <<- relu(Z1) # %*% operator does matrix multiplication. 
  #Feed forward functions for output
  Z2 <<- W2%*%a1+b2
  
  normalized.output = relu(Z2) #output, but it's still nonmeaningful because it's normalized between 1 & 0.
  a2 = normalized.output
  #reverses normalization for a meaningful value.
  descaled.output = a2*(max(data[,TargetVariable])-min(data[,TargetVariable])) + min(data[,TargetVariable])
  
  paste("The Neural network predicts",descaled.output)
}
