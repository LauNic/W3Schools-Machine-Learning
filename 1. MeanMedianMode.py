import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
print("speed:", type(speed), speed)
# speed: <class 'list'> [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# Mean is the average
x = np.mean(speed)
print("mean: ", x)
# mean:  89.76923076923077

# Median is the value in the middle after sorting
med = np.median(speed)
print("median: ", med)
# median:  87.0

# Mode is the value that appears most
mod = st.mode(speed)
print("mode:", mod)
# mode: ModeResult(mode=86, count=3)

# draw logspace X-points
X = np.logspace(-3, 3, 5)
y = np.logspace(-3, 3, 5)
plt.plot(X, y)
plt.show()
