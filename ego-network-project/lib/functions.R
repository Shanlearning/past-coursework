shan_zhong_matrix_data_transform <- function(data) {
    data1<-data
    # duplicate the from and to, and combine them together
    data1$FROM<-data$TO
    data1$TO<-data$FROM
    
    data2<-rbind(data,data1)
    data2<-cbind(data2,1)
    colnames(data2) <-c("FROM", "TO","Value")
    
    # expand data into a big matrix
    matrixdata<-acast(data2, FROM~TO, value.var="Value")
    matrixdata[is.na(matrixdata)]<-0
    matrixdata<-matrixdata+diag(dim(matrixdata)[1])
 return(matrixdata)
}

shan_zhong_data_forggplot2 <- function(data) {
  data1<-data
  # duplicate the from and to, and combine them together
  data1$FROM<-data$TO
  data1$TO<-data$FROM
  
  data2<-rbind(data,data1)
  data2<-cbind(data2,1)
  colnames(data2) <-c("FROM", "TO","Value")
  return(data2)
}  
    
shan_zhong_data_formixMem <- function(data,K){
  
  # sample size
  Total<- dim(data)[1]
  # number of variables
  J <-dim(data)[2]
  # we only have one replicate for each of the variables
  Rj<-rep(1,J)
  # Nijr indicates the number of ranking levels for each variable.
  # Since all our data is multinomial it should be an array of all 1s
  Nijr <- array(1, dim = c(Total, J, max(Rj)))
  # Number of sub-populations
  #K<-5
  # There are 2 choices for each of the variables ranging from 0 to 1.
  Vj <- rep(2, J)
  # we initialize alpha to .2
  alpha <- rep(.2, K)
  # All variables are multinomial
  dist <- rep("multinomial", J)
  # obs are the observed responses. it is a 4-d array indexed by i,j,r,n
  # note that obs ranges from 0 to 2 for each response
  obs <- array(0, dim = c(Total, J, max(Rj), max(Nijr)))
  obs[, , 1, 1] <- as.matrix(data)
  
  # Initialize theta randomly with Dirichlet distributions
  theta <- array(0, dim = c(J, K, max(Vj)))
  for (j in 1:J) {
    theta[j, , ] <- gtools::rdirichlet(K, rep(.8, Vj[j]))
  }
  
  # Create the mixedMemModel
  # This object encodes the initialization points for the variational EM algorithim
  # and also encodes the observed parameters and responses
  initial <- mixedMemModel(Total = Total, J = J, Rj = Rj,
                           Nijr = Nijr, K = K, Vj = Vj, alpha = alpha,
                           theta = theta, dist = dist, obs = obs)
  return(initial)
}
