# Test R's stat.glmnet_lambdasmax to understand its standardization behavior

library(knockoff)

# Check the source code of stat.glmnet_lambdasmax
cat("=== stat.glmnet_lambdasmax source code ===\n")
print(knockoff::stat.glmnet_lambdasmax)

cat("\n\n=== stat.glmnet_lambdadiff source code ===\n")
print(knockoff::stat.glmnet_lambdadiff)
