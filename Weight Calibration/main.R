#install.packages("ncvreg")
library(xtable)

library(ncvreg)
library(MASS)
library(glmnet)
library(foreach)
library(DiceKriging)

n = 100
N = 10000
r = 0.75 
#r = 1
rho = 0.2

#beta = c(1.0, rep(0, 10), rep(1.5, 10))
beta = c(1.0, rep(0, 10), rep(1.5, 10), rep(0, 10), rep(3, 10))
p = length(beta) - 1

B = 100 # 497

Y_res = matrix(nr= B, nc = 12)

ar1_cor <- function(n, rho) {
  exponent <- abs(matrix(1:n - 1, nrow = n, ncol = n, byrow = TRUE) - 
                    (1:n - 1))
  rho^exponent
}

#Y_res <- foreach(simnum = 1:B, .combine = "rbind", packages =
#          c("ncvreg", "MASS", "DiceKriging")) %dopar%{

for(simnum in 1:B){
  set.seed(simnum)
  print(simnum)
  e = rnorm(N)
  
  x = mvrnorm(n = N, rep(0, p), ar1_cor(p, rho))
  x = pnorm(x)
  #x = scale(x, T, T)
  x = cbind(1, x)
  
  y = x %*% beta + e
  
  if(r == 0){
    z = rnorm(n = N)
  }else{
    z = rnorm(n = N, 0, sqrt((1 - r) / r)) + e
  }
  z_sorted_idx = order(z)
  len = round(N / 4)
  
  SampleIdx = rep(0, n)
  w_sampled = rep(0, n)
  n_h = round(c(.15, .20, .30, .35) * n)
  n_h[4] = n - n_h[3] - n_h[2] - n_h[1]
  cumn_h = cumsum(n_h)
  for (i in 1:4){
    Idx_z = (len * (i - 1)+ 1): (len * i)
    Idx = z_sorted_idx[Idx_z]
    from = ifelse(i == 1, 0, cumn_h[i-1])
    SampleIdx[(from + 1) : cumn_h[i]] = sample(Idx, size = n_h[i], replace = FALSE)
    w_sampled[(from + 1) : cumn_h[i]] = len / n_h[i]
  }
  
  #SRS non-informative
  SampleIdx = sample(1:N, n, replace = FALSE)
  
  x_sampled = x[SampleIdx, ]
  y_sampled = y[SampleIdx]
  
  ## True parameter ############
  Y_t = sum(y)
  X_t = apply(x, 2, sum)
  
  ## Horvitz Thompson Estimator ############
  Y_HT = sum(w_sampled * y_sampled)
  X_HT = drop(t(x_sampled) %*% w_sampled)
  
  ## Regression Estimator ############
  beta_wls = lm(y_sampled ~ 0 + x_sampled, weights = w_sampled)
  #beta_wls = lm(y_sampled2 ~ 0 + x_sampled2)
  coef_wls = coef(beta_wls)
  Y_REG = Y_HT + sum((X_t - X_HT) * coef_wls)
  
  ## Oracle Regression 
  beta_wls_Oracle = lm(y_sampled ~ 0 + x_sampled[, beta!=0], weights = w_sampled)
  coef_Oracle <- rep(0, length(beta))
  coef_Oracle[beta!=0] <- coef(beta_wls_Oracle)
  Y_Oracle = Y_HT + sum((X_t - X_HT) * coef_Oracle)

  
  ## Model Assisted Estimators ############
  x_sampled2 =  diag(sqrt(w_sampled)) %*% x_sampled
  y_sampled2 =  diag(sqrt(w_sampled)) %*% y_sampled
  
  # Ridge
  lambda_Ridge = 1
  
  fit = ncvfit(x_sampled2, y_sampled2, penalty = "lasso",
               alpha = .Machine$double.eps, lambda = lambda_Ridge)
  coef_Ridge2 = drop(solve(t(x_sampled) %*% diag(w_sampled, n) %*% x_sampled + n *lambda_Ridge * diag(p + 1), 
                           t(x_sampled) %*% diag(w_sampled, n) %*% y_sampled))
  coef_Ridge = unname(fit$beta)
  head(cbind(coef_Ridge, coef_Ridge2))
  
  Y_Ridge = Y_HT + sum((X_t - X_HT) * coef_Ridge)
  
  # Lasso
  lambda_vec = 1.2^(20:-20)
  lambda_Lasso = crossval(lambda_vec = lambda_vec, penalty = "lasso", 
                          alpha = 1)$lambda.min
  
  fit = ncvfit(x_sampled2, y_sampled2, penalty = "lasso",
               alpha = 1, lambda = lambda_Lasso)
  
  coef_Lasso = unname(fit$beta)
  coef_Lasso
  Y_Lasso = Y_HT + sum((X_t - X_HT) * coef_Lasso)
  
  # SCAD
  lambda_SCAD = crossval(lambda_vec = lambda_vec, penalty = "SCAD", 
                         alpha = 1)$lambda.min
  
  fit = ncvfit(x_sampled2, y_sampled2, penalty = "SCAD",
               alpha = 1, lambda = lambda_SCAD)
  
  coef_SCAD = unname(fit$beta)
  Y_SCAD = Y_HT + sum((X_t - X_HT) * coef_SCAD)
  
  
  ## Model Calibration Estimators #########
  # Ridge
  x_Ridge = cbind(1, x %*% coef_Ridge)
  X_Ridge_t = apply(x_Ridge, 2, sum)
  
  x_Ridge_sampled = cbind(1, x_sampled %*% coef_Ridge)
  X_Ridge_HT = drop(t(x_Ridge_sampled) %*% w_sampled)
  
  beta_Ridge = lm(y_sampled ~ 0 + x_Ridge_sampled, weights = w_sampled)
  Y_MC_Ridge = Y_HT + sum((X_Ridge_t - X_Ridge_HT) * coef(beta_Ridge))
  
  # Lasso
  x_Lasso = cbind(1, x %*% coef_Lasso)
  X_Lasso_t = apply(x_Lasso, 2, sum)
  
  x_Lasso_sampled = cbind(1, x_sampled %*% coef_Lasso)
  X_Lasso_HT = drop(t(x_Lasso_sampled) %*% w_sampled)
  
  beta_Lasso = lm(y_sampled ~ 0 + x_Lasso_sampled, weights = w_sampled)
  Y_MC_Lasso = Y_HT + sum((X_Lasso_t - X_Lasso_HT) * coef(beta_Lasso))
  
  # SCAD
  x_SCAD = cbind(1, x %*% coef_SCAD)
  X_SCAD_t = apply(x_SCAD, 2, sum)
  
  x_SCAD_sampled = cbind(1, x_sampled %*% coef_SCAD)
  X_SCAD_HT = drop(t(x_SCAD_sampled) %*% w_sampled)
  
  beta_SCAD = lm(y_sampled ~ 0 + x_SCAD_sampled, weights = w_sampled)
  Y_MC_SCAD = Y_HT + sum((X_SCAD_t - X_SCAD_HT) * coef(beta_SCAD))
  
  
  ##  Ridge Calibration estimators #########
  # Ridge
  Q = diag(1 / n / lambda_Ridge, p + 1)
  Y_Ridge_Ridge = drop(Y_HT + t(X_t - X_HT) %*% 
                         Q %*% t(x_sampled) %*% solve(x_sampled %*% Q %*% t(x_sampled) + diag(1 / w_sampled), y_sampled))
  #Q %*% t(x_sampled) %*% solve(x_sampled %*% Q %*% t(x_sampled) + diag(1 / w_sampled), y_sampled)
  #solve(t(x_sampled) %*% diag(w_sampled) %*% x_sampled + solve(Q), t(x_sampled) %*% diag(w_sampled) %*% y_sampled)
  
  # Lasso
  Q = diag(abs(coef_Lasso) / n / lambda_Lasso)
  Y_Lasso_Ridge = drop(Y_HT + t(X_t - X_HT) %*% 
                         Q %*% t(x_sampled) %*% solve(x_sampled %*% Q %*% t(x_sampled) + diag(1 / w_sampled), y_sampled))
  
  # SCAD
  denom = SCAD.derivative(abs(coef_SCAD), lambda_SCAD)
  denom = ifelse(denom == 0, 0.000000001, denom)
  
  Q = diag(abs(coef_SCAD) / denom / n)
  Y_SCAD_Ridge = drop(Y_HT + t(X_t - X_HT) %*% 
                        Q %*% t(x_sampled) %*% solve(x_sampled %*% Q %*% t(x_sampled) + diag(1 / w_sampled), y_sampled))
  
  ## Lasso Calibration estimator #########
  
  Y_res[simnum, ] <- c(Y_t, Y_HT, Y_REG,
                       Y_Ridge, Y_Lasso, Y_SCAD,
                       Y_MC_Ridge, Y_MC_Lasso, Y_MC_SCAD,
                       Y_Lasso_Ridge, Y_SCAD_Ridge, Y_Oracle)
  # c(Y_t, Y_HT, Y_REG, 
  #   Y_Ridge, Y_Lasso, Y_SCAD, 
  #   Y_MC_Ridge, Y_MC_Lasso, Y_MC_SCAD,
  #   Y_Lasso_Ridge, Y_SCAD_Ridge)
}
colnames(Y_res) <- c("TruePara", "HT", "GREG", "Ridge", 
                     "Lasso", "SCAD",
                     "Ridge_MC", "Lasso_MC",
                     "SCAD_MC", "Lasso_Ridge",
                     "SCAD_Ridge", "Oracle")
write.csv(Y_res, sprintf("Y_res_B%g_r%g_n%g.csv", B, r, n))

Bias_res = apply(Y_res[,-1] - Y_res[,1], 2, mean)
SD_res = apply(Y_res[,-1], 2, sd)
RMSE_res = sqrt(Bias_res^2 + SD_res^2)

xtable(cbind(Bias_res, SD_res, RMSE_res), caption = sprintf("n = %g, p = %g, r = %g", n, p, r), align = "lrrr")

cbind(Bias_res / Bias_res[11], SD_res / SD_res[11], RMSE_res / RMSE_res[11])
cbind(Bias_res, SD_res, RMSE_res)