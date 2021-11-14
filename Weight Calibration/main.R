#install.packages("ncvreg")
library(ncvreg)
library(MASS)
library(MXM)
library(glmnet)

n = 100
N = 10000
p = 40
r = 0.75

beta = c(1.0, rep(0, 10), rep(1.5, 10), rep(0, 10), rep(3.0, 10))

B = 1000 # 497

e = rnorm(N)

ar1_cor <- function(n, rho) {
  exponent <- abs(matrix(1:n - 1, nrow = n, ncol = n, byrow = TRUE) - 
                    (1:n - 1))
  rho^exponent
}

x = mvrnorm(n = N, rep(0, p), ar1_cor(40, 0.2))
x = pnorm(x)
x = scale(x, T, T)
x = cbind(1, x)


y = x %*% beta + e

ratio = (1 - r) / r
z = rnorm(n = N, 0, sqrt(ratio)) + e
z_sorted_idx = order(z)
len = round(N / 4)

SampleIdx = rep(0, n)
w_sampled = rep(0, n)
n_h = c(15, 20, 30, 35)
cumn_h = cumsum(n_h)
for (i in 1:4){
  Idx_z = (len * (i - 1)+ 1): (len * i)
  Idx = z_sorted_idx[Idx_z]
  from = ifelse(i == 1, 0, cumn_h[i-1])
  SampleIdx[(from + 1) : cumn_h[i]] = sample(Idx, size = n_h[i], replace = FALSE)
  w_sampled[(from + 1) : cumn_h[i]] = len / n_h[i]
}

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
Y_REG = Y_HT + sum((X_t - X_HT) * coef(beta_wls))

## Model Assisted Estimators ############
x_sampled2 =  diag(sqrt(w_sampled)) %*% x_sampled
y_sampled2 =  diag(sqrt(w_sampled)) %*% y_sampled

# Ridge
#fit_Ridge <- cv.ncvreg(x_sampled2, y_sampled2, penalty = "lasso", alpha = .Machine$double.eps)
#fit_Ridge <- cv.glmnet(x_sampled[, -1], y_sampled, weights = w_sampled, standardize = FALSE,
#                       alpha = 0, lambda = fit_Ridge$lambda *.Machine$double.eps, thresh = 1E-15)
#fit_Ridge$lambda*.Machine$double.eps
#fit_Ridge$lambda.min*.Machine$double.eps
#lambda_Ridge = fit_Ridge$lambda.min
x_sampled[1, 1] <- x_sampled[1, 1] + 0.0000001
fit_Ridge <- cv.glmnet(x_sampled, y_sampled, weights = w_sampled, 
                       standardize = FALSE, intercept = FALSE,
                       alpha = 0, thresh = 1E-25)
coef_Ridge2 = as.vector(coef.glmnet(fit_Ridge, s = fit_Ridge$lambda.min, exact = TRUE, 
                      x = x_sampled, y= y_sampled, weights= w_sampled))[-1]

lambda_Ridge = fit_Ridge$lambda.min / sd(y_sampled)

coef_Ridge = drop(solve(t(x_sampled) %*% diag(w_sampled, n) %*% x_sampled + n / sd(y_sampled) * fit_Ridge$lambda.min * diag(p + 1), 
           t(x_sampled) %*% diag(w_sampled, n) %*% y_sampled))

Y_Ridge = Y_HT + sum((X_t - X_HT) * coef_Ridge)
#unname(coef(beta_wls))

# Lasso
fit_Lasso <- cv.glmnet(x_sampled, y_sampled, weights = w_sampled, 
                       standardize = FALSE, intercept = FALSE, thresh = 1E-20)
coef_Lasso = as.vector(coef.glmnet(fit_Lasso, s = fit_Lasso$lambda.min, exact = TRUE, 
                      x = x_sampled, y= y_sampled, weights= w_sampled))[-1]
#fit_Lasso$lambda

lambda_Lasso = fit_Lasso$lambda.min / sd(y_sampled)
Y_Lasso = Y_HT + sum((X_t - X_HT) * coef_Lasso)

# SCAD

## Model Calibration Estimators #########
# Ridge
x_Ridge = cbind(1, x %*% as.vector(coef_Ridge))
X_Ridge_t = apply(x_Ridge, 2, sum)

x_Ridge_sampled = cbind(1, x_sampled %*% as.vector(coef_Ridge))
X_Ridge_HT = drop(t(x_Ridge_sampled) %*% w_sampled)

beta_Ridge = lm(y_sampled ~ 0 + x_Ridge_sampled, weights = w_sampled)
Y_MC_Ridge = Y_HT + sum((X_Ridge_t - X_Ridge_HT) * coef(beta_Ridge))

# Lasso
x_Lasso = cbind(1, x %*% as.vector(coef_Lasso))
X_Lasso_t = apply(x_Lasso, 2, sum)

x_Lasso_sampled = cbind(1, x_sampled %*% as.vector(coef_Lasso))
X_Lasso_HT = drop(t(x_Lasso_sampled) %*% w_sampled)

beta_Lasso = lm(y_sampled ~ 0 + x_Lasso_sampled, weights = w_sampled)
Y_MC_Lasso = Y_HT + sum((X_Lasso_t - X_Lasso_HT) * coef(beta_Lasso))

# SCAD

##  Ridge Calibration estimators #########
# Ridge
Q = diag(1 / n / lambda_Ridge, p + 1)
Y_Ridge_Ridge = drop(Y_HT + t(X_t - X_HT) %*% 
  Q %*% t(x_sampled) %*% solve(x_sampled %*% Q %*% t(x_sampled) + diag(1 / w_sampled), y_sampled))
Q %*% t(x_sampled) %*% solve(x_sampled %*% Q %*% t(x_sampled) + diag(1 / w_sampled), y_sampled)
solve(t(x_sampled) %*% diag(w_sampled) %*% x_sampled + solve(Q), t(x_sampled) %*% diag(w_sampled) %*% y_sampled)

Q %*% t(x_sampled) %*% solve(x_sampled %*% Q %*% t(x_sampled) + diag(1 / w_sampled), y_sampled)


# Lasso


# SCAD

## Lasso Calibration estimator #########


