crossval = function(lambda_vec,
                    penalty, alpha, ..., 
                    K = 50, data_sampled = cbind(x_sampled2, y_sampled2)){
  p = ncol(data_sampled)-2
  score_vec = rep(0, length(lambda_vec))
  for(lambdaidx in 1:length(lambda_vec)){
    #Randomly shuffle the data
    data_sampled <-data_sampled[sample(nrow(data_sampled)),]
    
    #Create 10 equally size folds
    folds <- cut(seq(1,nrow(data_sampled)),breaks=K,labels=FALSE)
    
    #Perform 10 fold cross validation
    error_vec = rep(0, K)
    for(i in 1:K){
      #Segement your data by fold using the which() function 
      testIndexes <- which(folds==i,arr.ind=TRUE)
      testData <- data_sampled[testIndexes, ]
      trainData <- data_sampled[-testIndexes, ]
      
      #Use the test and train data partitions however you desire...
      fit = ncvfit(trainData[,1:(p+1)], trainData[,p+2], penalty = penalty,
                   alpha = alpha, lambda = lambda_vec[lambdaidx])
      error_vec[i] = sum((testData[,1:(p+1)] %*% unname(fit$beta) - testData[,p+2])^2)
    }
    
    score_vec[lambdaidx] = mean(error_vec)
  }
  return(list(score_vec = score_vec,
              lambda.min = lambda_vec[which.min(score_vec)]))
}

# lambda_vec = 1.2^(-20:20)
# plot(crossval(lambda_vec = lambda_vec, penalty = "lasso", 
#               alpha = 0.000001)$score_vec ~ log(lambda_vec))
# 
# lambda_vec = 1.2^(20:-30)
# plot(crossval(lambda_vec = lambda_vec, penalty = "SCAD", 
#               alpha = 1)$score_vec ~ log(lambda_vec))

