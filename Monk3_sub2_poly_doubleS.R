################# MONK's Problem 1 ############################### double selection
#subspace dim = 2
data = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.train")
data2 = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-3.test")

alldata = rbind(data, data2)

set.seed(0);alldata$y = (alldata$V1 *10) + rnorm((dim(alldata)[1]), mean=0, sd=5)
head(alldata)

data.lhsk <- alldata[, -c(1,8)]
head(data.lhsk)

########### Or data (may need to change the percentage of selection criterion) ##########
#data = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train")
#data2 = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test")
#alldata = rbind(data, data2)
#set.seed(0);alldata$y = (alldata$V1 *10) + rnorm((dim(alldata)[1]), mean=0, sd=5)
#head(alldata)
#data.lhsk <- alldata[, -c(1,8)]
#head(data.lhsk)

library(e1071)
library(tictoc)
library(kernlab)
library(matlib)
library(lava)

tic()
seed <- 0
no.col <- 2  #subspace dim = 2
k <- 5
all.testerr <- c()
alp <- 0.01

crit.col <- rep(list(c()), 5)
cri_varnum <- rep(list(c()), 5)
optpar <- c()



degree=c(1,2,3)
epsilon =c(0.01,0.05,0.1,0.2,0.5)
cost=c(0.01,0.1,1,10,100) # Double selection


#gamma = c(1/no.col)
par.test = expand.grid(degree, epsilon, cost)
#par.test = expand.grid(degree, epsilon, cost,gamma)


train.set4 <- data.lhsk # use all data

### DATA NORMALIZATION ########################################################
x.mean <- apply(train.set4[,-ncol(train.set4)], 2, mean)
x.sd <- apply(train.set4[,-ncol(train.set4)], 2, sd)
train.set4[,-ncol(train.set4)] <- t(apply(train.set4[,-ncol(train.set4)], 1, function(x) (x-x.mean)/x.sd))
#test.set4[,-ncol(test.set4)] <- t(apply(test.set4[,-ncol(test.set4)], 1, function(x) (x-x.mean)/x.sd))
###############################################################################

#all.col <- c()
new.form <- c()

# Create 5 fold indices for selection criterion
#seed <- 1
set.seed(seed);id.crit <- sample(nrow(train.set4))
seed <- seed+1

folds <- cut(seq(1,nrow(train.set4)),breaks=k,labels=FALSE)


gcv_store = c()
store_testerror <- c()
opt_col =rep(list(c()), nrow(par.test))
store_varnum <- rep(list(c()), nrow(par.test))
for (w in 1: nrow(par.test)) {
  print(paste0("Parameter Iteration: ",w,"********"))
  varnum <-c()
  par = as.numeric(par.test[w,])
  
  # Find the baseline values
  ave0.rmse <- c()
  resid.tune <- rep(list(c()), k)
  resid.hold <- rep(list(c()), k)
  fitted.tune <- rep(list(c()), k)
  fitted.hold <- rep(list(c()), k)
  for(e in 1:k){
    #Segement your data by fold using the which() function 
    holdoutIndexes <- id.crit[which(folds==e,arr.ind=TRUE)]
    TUNE <- train.set4[-holdoutIndexes, ]
    HOLDOUT <- train.set4[holdoutIndexes, ]
    
    mod0 <- lm(y~1, data = TUNE)
    pred0 <- predict(mod0, HOLDOUT)
    
    ave0.rmse <- c(ave0.rmse, sqrt(mean((HOLDOUT$y - pred0)^2)))
    
    resid.tune[[e]] <- TUNE$y
    resid.hold[[e]] <- HOLDOUT$y
  }
  rmse0 <- mean(ave0.rmse)
  
  # Iterative subspace search
  iter <- 0
  all.col <- c()
  tmpsumtrace <- rep(0, k)
  #seed <- 15 #17 15
  seed <- 25 #640
  
  repeat {
    iter <- iter + 1
    
    set.seed(seed);three.col <- sample(1:ncol(train.set4[,-ncol(train.set4)]),no.col)
    seed <- seed+1
    
    var.tmp <- names(train.set4[, three.col])
    tmp.form <- as.formula(paste0("res~",paste(var.tmp, collapse = "+")))
    
    rmse <- c()
    pvalue_list <- c()
    for(e in 1:k){
      res <- resid.tune[[e]]
      
      holdoutIndexes <- id.crit[which(folds==e,arr.ind=TRUE)]
      TUNE <- train.set4[-holdoutIndexes, ]
      HOLDOUT <- train.set4[holdoutIndexes, ]
      
      mod.sel <- svm(tmp.form, data = TUNE, kernel = "polynomial", degree = par[1], epsilon= par[2], cost= par[3])
      fitted.tune[[e]] <- predict(mod.sel, TUNE)
      fitted.hold[[e]] <- predict(mod.sel, HOLDOUT)
      rmse <- c(rmse, sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2)))
      threermse <- sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2))
    }
    avg.rmse <- mean(rmse)
    
    # termination criterion
    if( (abs((rmse0-avg.rmse)/rmse0) < .0001 | iter > 20)&length(all.col)>0 ){
      
      resid.tr <- train.set4$y
      #resid.ts <- test.set4$y
      s_store = c()
      
      # add critical subspace (different dimension)
      for (t in 1:length(varnum)) {
        if(t==1){
          var.set <- all.col[1: (varnum[t])]
        }else{
          var.set <- all.col[(sum(varnum[1:(t-1)])+1) : (sum(varnum[1:t]))]
        }
        
        # model building
        form <- as.formula(paste0("resid.tr~",paste(var.set, collapse = "+")))
        mod <- svm(form, data = train.set4, kernel = "polynomial", degree = par[1], epsilon= par[2], cost= par[3])
        
        ##### calculate trace(s),inner product
        polykernel <- polydot(degree = par[1], scale = 1/length(var.set), offset = 0)
        TRAINt <- as.data.frame(train.set4[,var.set])
        
        matrixx <- matrix(as.numeric(unlist(TRAINt)),nrow=nrow(TRAINt), ncol= ncol(TRAINt))
        inner_product <- kernelMatrix(polykernel, matrixx)
        alpha_y <- rep(0, length(resid.tr))
        alpha_y[mod$index] <- as.numeric(mod$coefs)
        #st <- inner_product %*% diag(alpha_y/resid.tr)
        st <- solve(inner_product + diag(1/par[3], nrow(inner_product))) %*% inner_product
        trace_st <- tr(st)
        
        s_store <- c(s_store, trace_st)
        
        resid.tr <- resid.tr - predict(mod, train.set4)
        #resid.ts <- resid.ts - predict(mod, test.set4)
      }
      
      #test.error <- sqrt(mean((resid.ts)^2))
      s_sum <- sum(s_store)
      
      # GCV calculation
      gcv_cal <- mean((resid.tr/(1-s_sum/nrow(train.set4)))**2)
      gcv_store <- c(gcv_store, gcv_cal)
      
      #store_testerror = c(store_testerror, test.error)
      
      opt_col[[w]] <- all.col
      store_varnum[[w]] <- varnum
      
      print(iter);print(rmse0);print(avg.rmse);print(all.col);print(gcv_cal);print(varnum)
      break
      
    }
    
    
    
    # calculate p-value and avg.rmse
    tmptrace <- c()
    for(e in 1:k){
      res <- resid.tune[[e]]
      
      holdoutIndexes <- id.crit[which(folds==e,arr.ind=TRUE)]
      TUNE <- train.set4[-holdoutIndexes, ]
      HOLDOUT <- train.set4[holdoutIndexes, ]
      
      mod.sel <- svm(tmp.form, data = TUNE, kernel = "polynomial", degree = par[1], epsilon= par[2], cost= par[3])
      fitted.tune[[e]] <- predict(mod.sel, TUNE)
      fitted.hold[[e]] <- predict(mod.sel, HOLDOUT)
      rmse <- c(rmse, sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2)))
      threermse <- sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2))
      
      ##### calculate trace(s),inner product
      polykernel <- polydot(degree = par[1], scale = 1/no.col, offset = 0)
      TRAIN <- as.data.frame(TUNE[,var.tmp])
      
      matrixx <- matrix(as.numeric(unlist(TRAIN)),nrow=nrow(TRAIN), ncol= ncol(TRAIN))
      inner_product <- kernelMatrix(polykernel, matrixx)
      alpha_y <- rep(0, nrow(TRAIN))
      alpha_y[mod.sel$index] <- as.numeric(mod.sel$coefs)
      
      s <- solve(inner_product + diag(1/par[3], nrow(inner_product))) %*% inner_product
      trace_s <- tr(s)
      
      tmptrace[e] <- trace_s
      tmpsumtrace[e] <- tmpsumtrace[e] + tmptrace[e]
      
      # Test A fstat ####
      fstat <-  ((rmse0**2 - threermse**2)/ trace_s) / (threermse**2/(nrow(HOLDOUT)- tmpsumtrace[e]))
      pvalue <- 1 - pf(fstat, trace_s, nrow(HOLDOUT) - tmpsumtrace[e])
      pvalue_list <- c(pvalue_list, pvalue)
      
    }
    avg.rmse <- mean(rmse)
    
    
    
    
    # selection criterion
    if( (min(pvalue_list) < alp)&(avg.rmse<rmse0) ){
      #take 1 var from 2 var
      vvar = combn(var.tmp, 1)
      #store the prediction error of 1 var
      onecombremse <- c()
      for (z in 1:no.col){
        comvar <- vvar[,z]
        tmp.form2 <- as.formula(paste0("res~",paste(comvar, collapse = "+")))
        
        #store the rmse of 5-fold CV (1 var)
        rmse2 <- c()
        for(e in 1:k){
          res <- resid.tune[[e]]
          
          holdoutIndexes <- id.crit[which(folds==e,arr.ind=TRUE)]
          TUNE <- train.set4[-holdoutIndexes, ]
          HOLDOUT <- train.set4[holdoutIndexes, ]
          
          mod.sel <- svm(tmp.form2, data = TUNE, kernel = "polynomial", degree = par[1], epsilon= par[2], cost= par[3])
          fitted.tune[[e]] <- predict(mod.sel, TUNE)
          fitted.hold[[e]] <- predict(mod.sel, HOLDOUT)
          rmse2 <- c(rmse2, sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2)))
          twormse <- sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2))
        }
        avg.rmse2 <- mean(rmse2)
        onecombremse <- c(onecombremse, avg.rmse2)
      }
      #find the minimum rmse of 1 var
      finalonecombrmse <- min(onecombremse)
      minione = which(onecombremse == min(onecombremse))
      if (finalonecombrmse > avg.rmse){
        #print("add 2 var")
        # update important vars
        all.col <- c(all.col, var.tmp)
        varnum <-c(varnum, 2)
        rmse0 <- avg.rmse
        resid.tune <- lapply(1:k, function(id) resid.tune[[id]] - fitted.tune[[id]])
        resid.hold <- lapply(1:k, function(id) resid.hold[[id]] - fitted.hold[[id]])
        
        #print(iter);print(rmse0);print(avg.rmse)#;print(pvalue_list)
        
      }else{
        # add one var # need to remove the calculation of 2-var p-value
        tmpsumtrace <- tmpsumtrace - tmptrace
        
        tmp.form1 <- as.formula(paste0("res~",paste(vvar[, minione], collapse = "+")))
        tmptrace <- c()
        for(e in 1:k){
          res <- resid.tune[[e]]
          
          holdoutIndexes <- id.crit[which(folds==e,arr.ind=TRUE)]
          TUNE <- train.set4[-holdoutIndexes, ]
          HOLDOUT <- train.set4[holdoutIndexes, ]
          
          mod.sel <- svm(tmp.form1, data = TUNE, kernel = "polynomial", degree = par[1], epsilon= par[2], cost= par[3])
          fitted.tune[[e]] <- predict(mod.sel, TUNE)
          fitted.hold[[e]] <- predict(mod.sel, HOLDOUT)
          #rmse1 <- c(rmse1, sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2)))
          onermse <- sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2))
          
          #calculate trace(s),inner product
          polykernel <- polydot(degree = par[1], scale = 1, offset = 0)
          TRAIN <- as.data.frame(TUNE[, vvar[, minione]])
          
          matrixx <- matrix(as.numeric(unlist(TRAIN)),nrow=nrow(TRAIN), ncol= ncol(TRAIN))
          inner_product <- kernelMatrix(polykernel, matrixx)
          alpha_y <- rep(0, nrow(TRAIN))
          alpha_y[mod.sel$index] <- as.numeric(mod.sel$coefs)
          s <- solve(inner_product + diag(1/par[3], nrow(inner_product))) %*% inner_product
          trace_s <- tr(s)
          
          tmptrace[e] <- trace_s
          tmpsumtrace[e] <- tmpsumtrace[e] + tmptrace[e]
          
          fstat <-  ((rmse0**2 - onermse**2)/ trace_s) / (onermse**2/(nrow(HOLDOUT)- tmpsumtrace[e]))
          pvalue <- 1 - pf(fstat, trace_s, nrow(HOLDOUT) - tmpsumtrace[e])
          pvalue_list <- c(pvalue_list, pvalue)
        }
        #update important vars##########
        all.col <- c(all.col, vvar[, minione])
        varnum <- c(varnum, 1)
        rmse0 <- finalonecombrmse
        resid.tune <- lapply(1:k, function(id) resid.tune[[id]] - fitted.tune[[id]])
        resid.hold <- lapply(1:k, function(id) resid.hold[[id]] - fitted.hold[[id]])
        #store the prediction error of 1 var
        #onecombremse <- c()
        #find the minimum rmse of 1 var
        #finalonecombrmse <- min(onecombremse)
        #minione = which(onecombremse == min(onecombremse))
        
      }
      
    } else{
      tmpsumtrace <- tmpsumtrace - tmptrace} #this subspace is not critical; remove the calculation of trace (p-value)
  }
  
  
}

mini = which(gcv_store == min(gcv_store))
print(mini);print(gcv_store[mini])
print(par.test[mini,])
print(opt_col[[mini]])
print(store_varnum[[mini]])
#onetesterror = store_testerror[mini]
#all.testerr <- c(all.testerr, onetesterror)
#crit.col[[m]] <- opt_col[[mini]]


#avg.testerr <- mean(all.testerr)
#print(all.testerr)
#print(avg.testerr)
#print(sd(all.testerr))
#print(crit.col)
toc()