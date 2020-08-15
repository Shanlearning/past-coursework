###############################################################
###############################################################
# Moving-block bootstrap for time series: Consider the pivot 
# T_n = sqrt{n}(\bar X_n - \mu) and use MBB to build confidence 
# intervals for \mu.  In addition, estimate Var(T_n) with MBB.
###############################################################
###############################################################


# get mean and variance from block length l
get.mean.var.analytic<-function(l,X){
  n<-length(X)
  
  X.blocks <- matrix(NA,n-l+1,l)
  
  for( j in 1:(n-l+1)){
    X.blocks[j,] <- X[j:(j+l-1)]
  }
  
  # compute the mean of the MBB-induced distribution:
  X.bar.MBB <- mean(X.blocks)
  
  # compute MBB bootstrap estimator of var.T analytically
  accum <- 0
  for( j in 1:(n-l+1) )
    for(k in 1:l)
      for(m in 1:l){
        accum <- accum + (X.blocks[j,k] - X.bar.MBB)*(X.blocks[j,m] - X.bar.MBB)
      }
  
  var.T.hat.analytic <- accum / (l*(n-l+1))
  return(c(X.bar.MBB,var.T.hat.analytic))
}



get.mean.var.bootstrap<-function(l,X,B){
  n<-length(X)
  
  # compute the mean
  X.bar<- mean(X)
  
  
  X.blocks <- matrix(NA,n-l+1,l)
  
  for( j in 1:(n-l+1)){
    X.blocks[j,] <- X[j:(j+l-1)]
  }
  
  # compute the mean of the MBB-induced distribution:
  X.bar.MBB <- mean(X.blocks)
  
  # compute MBB bootstrap estimator of var.T analytically
  accum <- 0
  for( j in 1:(n-l+1) )
    for(k in 1:l)
      for(m in 1:l){
        accum <- accum + (X.blocks[j,k] - X.bar.MBB)*(X.blocks[j,m] - X.bar.MBB)
      }
  
  var.T.hat.analytic <- accum / (l*(n-l+1))
  
  #simulate by bootstrap
      T.star <- numeric(B)
      n <- length(X)
      X.bar.star<-numeric(B)
  
  # run simulation
  for( b in 1:B){
    n.blocks <- floor(n/l) + 1 # choose an extra block if n/l is non-integer
    which.blocks <- sample(1:(n-l+1),n.blocks,replace=TRUE)
    X.star.mat <- X.blocks[which.blocks,]
    X.star <- as.vector(t(X.star.mat))[1:n] # truncate to length n
    X.bar.star[b] <- mean(X.star)
    T.star[b] <- sqrt(n)*( X.bar.star[b] - X.bar.MBB )
  }

  # compute MC version of Var(T_n) in order to compare it
  var.T.hat.MC <- var(T.star)
  # get Monte-Carlo-approximated alpha/2 and 1-alpha/2 quantiles
  alpha <- 0.05
  G.alphaby2 <- sort(T.star)[floor(B*(1-alpha/2))]
  G.1minusalphaby2 <- sort(T.star)[floor(B*(alpha/2))]

  # construct (1-alpha)*100% CI for the mean using the MBB
  # estimates of the quantiles; record whether the interval
  # contains the true mean.
  lo.ci <- X.bar - G.alphaby2 / sqrt(n)
  up.ci <- X.bar - G.1minusalphaby2 / sqrt(n)
  covered <- (lo.ci < 0) & (up.ci > 0)
  return(c(var.T.hat.analytic,var.T.hat.MC,covered))
}

get.meanvar.bootstrap<-function(l,X,B){
  n<-length(X)
  
  # compute the mean
  X.bar<- mean(X)
  
  
  X.blocks <- matrix(NA,n-l+1,l)
  
  for( j in 1:(n-l+1)){
    X.blocks[j,] <- X[j:(j+l-1)]
  }
  
  # compute the mean of the MBB-induced distribution:
  X.bar.MBB <- mean(X.blocks)
  
  # compute MBB bootstrap estimator of var.T analytically
  accum <- 0
  for( j in 1:(n-l+1) )
    for(k in 1:l)
      for(m in 1:l){
        accum <- accum + (X.blocks[j,k] - X.bar.MBB)*(X.blocks[j,m] - X.bar.MBB)
      }
  
  var.T.hat.analytic <- accum / (l*(n-l+1))
  
  #simulate by bootstrap
  T.star <- numeric(B)
  n <- length(X)
  X.bar.star<-numeric(B)
  
  # run simulation
  for( b in 1:B){
    n.blocks <- floor(n/l) + 1 # choose an extra block if n/l is non-integer
    which.blocks <- sample(1:(n-l+1),n.blocks,replace=TRUE)
    X.star.mat <- X.blocks[which.blocks,]
    X.star <- as.vector(t(X.star.mat))[1:n] # truncate to length n
    X.bar.star[b] <- mean(X.star)
    T.star[b] <- sqrt(n)*( X.bar.star[b] - X.bar.MBB )
  }
  
  # compute MC version of Var(T_n) in order to compare it
  var.T.hat.MC <- var(T.star)
  # get Monte-Carlo-approximated alpha/2 and 1-alpha/2 quantiles
  alpha <- 0.05
  G.alphaby2 <- sort(T.star)[floor(B*(1-alpha/2))]
  G.1minusalphaby2 <- sort(T.star)[floor(B*(alpha/2))]
  
  # construct (1-alpha)*100% CI for the mean using the MBB
  # estimates of the quantiles; record whether the interval
  # contains the true mean.
  lo.ci <- X.bar - G.alphaby2 / sqrt(n)
  up.ci <- X.bar - G.1minusalphaby2 / sqrt(n)
  return(c(X.bar,X.bar.MBB,var.T.hat.analytic,var.T.hat.MC,lo.ci,up.ci))
}


