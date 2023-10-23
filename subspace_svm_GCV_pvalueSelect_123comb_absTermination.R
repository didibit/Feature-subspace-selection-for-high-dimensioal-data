#setwd("F:/UTK/Research/MetalDamage/Di") 
#1/2/3---- 20comb
#Form 1_abs termination_ p-value as selection  criterion directly 
#SECOND PAPER without grouping info

library(e1071)
library(tictoc)
library(kernlab)
library(matlib)
library(lava)
tic()
x <- read.csv("mydata3.csv")
x <- x[1:200,]
y <- read.csv("y3.csv")[1:200,]
data.lhsk  <- cbind(x, y)
names(data.lhsk)[ncol(data.lhsk)] <- "y"

seed <- 0
no.col <- 3
k <- 5
all.testerr <- c()
alp <- 0.01

crit.col <- rep(list(c()), 5)
cri_varnum <- rep(list(c()), 5)
optpar <- c()

# Calculate test error based on 5-fold CV
set.seed(seed);id.trts <- sample(nrow(data.lhsk))
seed <- seed+1

folds.train <- cut(seq(1,nrow(data.lhsk)),breaks=5,labels=FALSE)

degree=c(1)
epsilon =c(0.01,0.05,0.1)
cost=c(1, 2, 5)
#gamma = c(1/no.col)
par.test = expand.grid(degree, epsilon, cost)
#par.test = expand.grid(degree, epsilon, cost,gamma)
for (m in 1:5) {
  print(paste0("Iteration: ",m,"******************************"))
  tic("each dataset")
  testIndexes <- id.trts[which(folds.train==m,arr.ind=TRUE)]
  train.set4 <- data.lhsk[-testIndexes, ]
  test.set4 <- data.lhsk[testIndexes, ]
  
  ### DATA NORMALIZATION ########################################################
  x.mean <- apply(train.set4[,-ncol(train.set4)], 2, mean)
  x.sd <- apply(train.set4[,-ncol(train.set4)], 2, sd)
  train.set4[,-ncol(train.set4)] <- t(apply(train.set4[,-ncol(train.set4)], 1, function(x) (x-x.mean)/x.sd))
  test.set4[,-ncol(test.set4)] <- t(apply(test.set4[,-ncol(test.set4)], 1, function(x) (x-x.mean)/x.sd))
  ###############################################################################
  
  # Create 5 fold indices for selection criterion
  set.seed(seed);id.crit <- sample(nrow(train.set4))
  seed <- seed+1
  
  folds <- cut(seq(1,nrow(train.set4)),breaks=k,labels=FALSE)
  
  gcv_store = c()
  store_testerror <- c()
  opt_col <- rep(list(c()), nrow(par.test))
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
    
    repeat {
      iter <- iter + 1
      
      set.seed(seed);three.col <- sample(1:ncol(train.set4[,-c(42)]),no.col)
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
      if( (abs((rmse0-avg.rmse)/rmse0) < .0001 | iter >10000)&length(all.col)>0 ){
        if(iter >10000){print("************* iteration is greater than 10000 **********")}
        resid.tr <- train.set4$y
        resid.ts <- test.set4$y
        s_store = c()
        
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
          resid.ts <- resid.ts - predict(mod, test.set4)
        }
        
        test.error <- sqrt(mean((resid.ts)^2))
        s_sum <- sum(s_store)
        
        # GCV calculation
        gcv_cal <- mean((resid.tr/(1-s_sum/nrow(train.set4)))**2)
        gcv_store <- c(gcv_store, gcv_cal)
        
        store_testerror = c(store_testerror, test.error)
        
        opt_col[[w]] <- all.col
        store_varnum[[w]] <- varnum
        
        print(iter);print(rmse0);print(avg.rmse);print(test.error);print(gcv_cal)#;print(opt_col)
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
        polykernel <- polydot(degree = par[1], scale = 1/3, offset = 0)
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
      if(min(pvalue_list) < alp){
        #take 2 var from 3 var
        vvar = combn(var.tmp, 2)
        #store the prediction error of 2 var
        twocombremse <- c()
        for (z in 1:3){
          comvar <- vvar[,z]
          tmp.form2 <- as.formula(paste0("res~",paste(comvar, collapse = "+")))
          
          #store the rmse of 5-fold CV (2 var)
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
          twocombremse <- c(twocombremse, avg.rmse2)
        }
        #find the minimum rmse of 2 var
        finaltwocombrmse <- min(twocombremse)
        minitwo = which(twocombremse == min(twocombremse))
        if (finaltwocombrmse > avg.rmse){
          #print("add 3 var")
          # update important vars
          all.col <- c(all.col, var.tmp)
          varnum <-c(varnum, 3)
          rmse0 <- avg.rmse
          resid.tune <- lapply(1:k, function(id) resid.tune[[id]] - fitted.tune[[id]])
          resid.hold <- lapply(1:k, function(id) resid.hold[[id]] - fitted.hold[[id]])
          
          #print(iter);print(rmse0);print(avg.rmse)#;print(pvalue_list)

          
        }else{
          tmpsumtrace <- tmpsumtrace - tmptrace
          
          combonevar = combn(var.tmp, 1)
          #store the prediction error of 1 var
          onecombremse <- c()
          for (onez in 1:3){
            onevar <- combonevar[, onez]
            tmp.form1 <- as.formula(paste0("res~",paste(onevar, collapse = "+")))
            
            #store the rmse of 5-fold CV (1 var)
            rmse1 <- c()
            for(e in 1:k){
              res <- resid.tune[[e]]
              
              holdoutIndexes <- id.crit[which(folds==e,arr.ind=TRUE)]
              TUNE <- train.set4[-holdoutIndexes, ]
              HOLDOUT <- train.set4[holdoutIndexes, ]
              
              mod.sel <- svm(tmp.form1, data = TUNE, kernel = "polynomial", degree = par[1], epsilon= par[2], cost= par[3])
              fitted.tune[[e]] <- predict(mod.sel, TUNE)
              fitted.hold[[e]] <- predict(mod.sel, HOLDOUT)
              rmse1 <- c(rmse1, sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2)))
              onermse <- sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2))
            }
            avg.rmse1 <- mean(rmse1)
            onecombremse <- c(onecombremse, avg.rmse1)
            
          }
          #find the minimum rmse of 1 var
          finalonecombrmse <- min(onecombremse)
          minione = which(onecombremse == min(onecombremse))
          if(finalonecombrmse <= finaltwocombrmse){
            #print("add 1 var")
            tmp.form1 <- as.formula(paste0("res~",paste(combonevar[, minione], collapse = "+")))
            tmptrace <- c()
            for(e in 1:k){
              res <- resid.tune[[e]]
              
              holdoutIndexes <- id.crit[which(folds==e,arr.ind=TRUE)]
              TUNE <- train.set4[-holdoutIndexes, ]
              HOLDOUT <- train.set4[holdoutIndexes, ]
              
              mod.sel <- svm(tmp.form1, data = TUNE, kernel = "polynomial", degree = par[1], epsilon= par[2], cost= par[3])
              fitted.tune[[e]] <- predict(mod.sel, TUNE)
              fitted.hold[[e]] <- predict(mod.sel, HOLDOUT)
              rmse1 <- c(rmse1, sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2)))
              onermse <- sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2))
              
              #calculate trace(s),inner product
              polykernel <- polydot(degree = par[1], scale = 1, offset = 0)
              TRAIN <- as.data.frame(TUNE[, combonevar[, minione]])
              
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
            all.col <- c(all.col, combonevar[, minione])
            varnum <- c(varnum, 1)
            rmse0 <- finalonecombrmse
            resid.tune <- lapply(1:k, function(id) resid.tune[[id]] - fitted.tune[[id]])
            resid.hold <- lapply(1:k, function(id) resid.hold[[id]] - fitted.hold[[id]])
            
            #print(iter);print(rmse0);print(finalonecombrmse)#;print(pvalue_list)
            
            
          }else{
            
           # tmpsumtrace <- tmpsumtrace - tmptrace
            
            
            #print("add 2 var")
            tmp.form2 <- as.formula(paste0("res~",paste(vvar[,minitwo], collapse = "+")))
            
            #store the rmse of 5-fold CV (2 var)
            rmse2 <- c()
            tmptrace <- c()
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
              
              # calculate trace(s),inner product
              polykernel <- polydot(degree = par[1], scale = 1/2, offset = 0)
              TRAIN <- as.data.frame(TUNE[, vvar[,minitwo]])
              
              matrixx <- matrix(as.numeric(unlist(TRAIN)),nrow=nrow(TRAIN), ncol= ncol(TRAIN))
              inner_product <- kernelMatrix(polykernel, matrixx)
              alpha_y <- rep(0, nrow(TRAIN))
              alpha_y[mod.sel$index] <- as.numeric(mod.sel$coefs)
              s <- solve(inner_product + diag(1/par[3], nrow(inner_product))) %*% inner_product
              trace_s <- tr(s)
              tmptrace[e] <- trace_s
              tmpsumtrace[e] <- tmpsumtrace[e] + tmptrace[e]
              
              fstat <-  ((rmse0**2 - twormse**2)/ trace_s) / (twormse**2/(nrow(HOLDOUT)- tmpsumtrace[e]))
              pvalue <- 1 - pf(fstat, trace_s, nrow(HOLDOUT) - tmpsumtrace[e])
              pvalue_list <- c(pvalue_list, pvalue)
            }
            #avg.rmse2 <- mean(rmse2)
            #twocombremse <- c(twocombremse, avg.rmse2)
            
            
            # update important vars##########
            all.col <- c(all.col, vvar[,minitwo])
            varnum <- c(varnum, 2)
            rmse0 <- finaltwocombrmse
            resid.tune <- lapply(1:k, function(id) resid.tune[[id]] - fitted.tune[[id]])
            resid.hold <- lapply(1:k, function(id) resid.hold[[id]] - fitted.hold[[id]])
            
            #print(iter);print(rmse0);print(finalonecombrmse)#;print(pvalue_list)
            
   
          }
          
        }
        
      } else{
        tmpsumtrace <- tmpsumtrace - tmptrace}
    }
  }
  mini = which(gcv_store == min(gcv_store))
  print(mini);print(gcv_store[mini])
  print(par.test[mini,])
  optpar <- c(optpar, par.test[mini,])
  cri_varnum[[m]] <- store_varnum[[mini]]
  onetesterror = store_testerror[mini]
  all.testerr <- c(all.testerr, onetesterror)
  crit.col[[m]] <- opt_col[[mini]]
  
  toc()
}

avg.testerr <- mean(all.testerr)
print(all.testerr)
print(avg.testerr)
print(sd(all.testerr))
print(crit.col)
print(cri_varnum)
print(optpar)

toc()