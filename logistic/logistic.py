# Based on http://blog.yhathq.com/posts/logistic-regression-and-python.html

import pandas as pd
import numpy as np
import pylab as pl
import statsmodels.api as sm

df = pd.read_csv("binary.csv")
df.describe()
df.head()
df.columns = ["admit", "gre", "gpa", "prestige"]
df.head()
pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])
df.hist()
dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')

cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
data
data.head()
data['intercept'] = 1.0
data
train_cols = data.columns[1:]
logit = sm.Logit(data['admit'], data[train_cols])
result = logit.fit()
result.summary()
np.exp(result.params)
result.predict([700, 4.0, 1, 0, 0, 1.0])

# Build X, Y from 1st file
f = open('ex2data1.txt')
lines = f.readlines()
x1 = []
x2 = []
y = []
for line in lines:
    line = line.replace("\n", "")
    vals = line.split(",")
    x1.append(float(vals[0]))
    x2.append(float(vals[1]))
    y.append(int(vals[2]))

x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)

x = np.vstack([x1, x2]).T

logit = sm.Logit(y, x)
result = logit.fit()

result.params
result.predict([45.0, 85.0])

# Build X, Y from 2nd file
f = open('ex2data2.txt')
lines = f.readlines()
x1 = []
x2 = []
y = []
for line in lines:
    line = line.replace("\n", "")
    vals = line.split(",")
    x1.append(float(vals[0]))
    x2.append(float(vals[1]))
    y.append(int(vals[2]))

x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)

x = np.vstack([x1, x2]).T

pos_mask = y == 1
neg_mask = y == 0
pos_x1 = x1[pos_mask]
neg_x1 = x1[neg_mask]
pos_x2 = x2[pos_mask]
neg_x2 = x2[neg_mask]
pl.clf()
pl.scatter(pos_x1, pos_x2, c='r')
pl.scatter(neg_x1, neg_x2, c='g')
logit = sm.Logit(y, x)
result = logit.fit()

result.params
result.predict([1.0, 1.0])

result


