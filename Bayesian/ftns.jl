function solvebeta(x, y, beta, n)
    beta_t1 = 0.0
    beta_t2 = 0.0
    cnt = 0
    while true
        cnt = cnt + 1
        f = sum(@. 1 / (1 + exp(-beta_t1 - x * beta[2] - y * beta[3]))) - n
        df = sum(@. 1 / (1 + exp(-beta_t1 - x * beta[2] - y * beta[3])) / (1 + exp(beta_t1 + x * beta[2] + y * beta[3])))
        beta_t2 = beta_t1 - f / df
        if abs(beta_t2 - beta_t1) < 10^(-100)
            break
        elseif cnt > 500
            #@warn("beta0 NOT convergent")
            break
        else
            beta_t1 = beta_t2
        end
    end
    return beta_t2
end
