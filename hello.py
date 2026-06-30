import pandas as pd
import numpy as np

df = pd.read_csv('Task 3 and 4_Loan_Data.csv')
print(df.head())
print(df.info())

df = df.drop(columns=['customer_id'])

print(df.head())

print(df['fico_score'].describe())

n=5
df['fico_score'].sort_values
fico_scores = np.sort(df['fico_score'].unique())

print(fico_scores)

grouped =df.groupby('fico_score')['default'].agg(['count','sum'])


print(grouped)

n_per_score = grouped['count'].values
k_per_score = grouped['sum'].values

print(n_per_score)
print(k_per_score)

cum_n = np.cumsum(n_per_score)
cum_k = np.cumsum(k_per_score)

print(cum_n)
print(cum_k)

def bucket_score(j, i):
    n = cum_n[i] - cum_n[j-1] if j>0 else cum_n[i]
    k = cum_k[i] - cum_k[j-1] if j>0 else cum_k[i]
    if k == 0 or k == n:
        return 0

    p = k/n
    score = k * np.log(p) + (n-k)* np.log(1 - p)
    return score

    
print("about to call bucket_score")
print(bucket_score(0, len(cum_n)-1))

dp = np.full((len(fico_scores)+1, n+1), -np.inf)
split = np.full((len(fico_scores)+1, n+1), 0, dtype=int)
dp[0, 0] = 0

for i in range(1, len(fico_scores)+1):
    for k in range(1, n+1):
        for j in range(k-1, i):
            candidate = dp[j, k-1] + bucket_score(j, i-1)
            if candidate > dp[i, k]:
                dp[i, k] = candidate
                split[i, k] = j

print(dp)
print(split)
print(dp[len(fico_scores), n])

boundaries = []
i = len(fico_scores)
for k in range(n, 0, -1):
    j = split[i, k]
    boundaries.append(fico_scores[j])
    i = j


boundaries.reverse()
boundaries = [int(b) for b in boundaries]

print("total borrowers: ", cum_n[len(cum_n)-1])

print("bucket boundaries (FICO scores): ", boundaries)

print("log-likelihood score: ", dp[len(fico_scores), n])

print("optimal number of buckets: ", n)

print("default rate: ", cum_k[len(cum_n)-1] / cum_n[len(cum_n)-1])

print("default probability: ", cum_k[len(cum_n)-1] / cum_n[len(cum_n)-1] * 100,"%")