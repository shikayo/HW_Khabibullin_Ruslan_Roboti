import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize

def func(x, a, b, c):
    y = np.polyval([a,b,c], x)
    return y

x = np.linspace(0, 5, 45)
n = np.random.normal(0,0.2,45)
x = x + n

y = func(x,np.random.random(len(x/2)),np.random.random(len(x/2)),np.random.random(len(x/2)))

alpha = optimize.curve_fit(func, xdata = x, ydata = y)[0]
print(alpha)

x_new_value = np.arange(min(x), max(x), 0.1)
y_new_value = func(x_new_value, alpha[0],alpha[1],alpha[2])

plt.figure(figsize = (10,8))
plt.plot(x,y, 'b.')
plt.plot(x_new_value,y_new_value, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()