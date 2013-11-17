from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Build X/Y arrays from file 1
f = open('ex1data1.txt')
lines = f.readlines()
x = []
y = []
for line in lines:
    line = line.replace("\n", "")
    vals = line.split(",")
    x.append(float(vals[0]))
    y.append(float(vals[1]))

x = np.array(x)
y = np.array(y)

# Run regression
gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)

print gradient, intercept, r_value, p_value, std_err

sc = plt.scatter(x, y)
plt.plot(x, (gradient*x + intercept))

# Now curve fit a higher polynomial to the same data
popt, pcov = curve_fit(func, x, y)
"""
The result is:
popt[0] = a , popt[1] = b and popt[2] = c of the function,
so f(x) = popt[0]*x**2 + popt[1]*x**3 + popt[2].
"""

print "a = %s , b = %s, c = %s" % (popt[0], popt[1], popt[2])
"""
Print the coefficients and plot the funcion.
"""

plt.plot(x, func(x, *popt), label="Fitted Curve")


# Build X, Y from 2nd file
f = open('ex1data2.txt')
lines = f.readlines()
x1 = []
x2 = []
y = []
for line in lines:
    line = line.replace("\n", "")
    vals = line.split(",")
    x1.append(float(vals[0]))
    x2.append(float(vals[1]))
    y.append(float(vals[2]))

x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)


# linregress doesn't do multi regression, so we use sklearn
ones = np.ones(x1.shape)
x = np.vstack([x1, x2]).T # don't need ones for

regr = linear_model.LinearRegression()
regr.fit( x, y )
regr.predict([3000, 3])
