#install.packages("ncvreg")
library(ncvreg)
library(MASS)
library(MXM)

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

## True parameter
Y_t = sum(y)
X_t = apply(x, 2, sum)

## Horvitz Thompson Estimator
Y_HT = sum(w_sampled * y_sampled)
X_HT = drop(t(x_sampled) %*% w_sampled)

# Regression Estimator
beta_wls = lm(y_sampled ~ 0 + x_sampled, weights = w_sampled)
Y_REG = Y_HT + sum((X_t - X_HT) * coef(beta_wls))


fit <- cv.ncvreg(x_sampled, y_sampled, penalty = "lasso")
fit <- cv.ncvreg(x_sampled, y_sampled, penalty = "lasso", alpha = .Machine$double.eps)

fit <- cv.ncvreg(x, y, penalty = "SCAD")
sfit <- summary(fit)
sfit
sfit$lambda[sfit$min]
coef(fit)
