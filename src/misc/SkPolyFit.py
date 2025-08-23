""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
	 
# SkPolyFit.py:  Polynomial regression with sklearn

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

poly = PolynomialFeatures(degree=6, include_bias=False)    # Degree 6 poly
mA = np.array([1, 2, 3, 4, 5, 6, 7])                  # Atomic mass number
poly_features = poly.fit_transform(mA.reshape(-1, 1))               # Data
B = [0.0140, 1.1520, 2.8235, 1.8150, 1.3871, 0.9465, 1.2971]        # BE/N
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, B)
b_predicted = poly_reg_model.predict(poly_features)
intcp = poly_reg_model.intercept_,
print(intcp)
coefs = poly_reg_model.coef_                     # Polynomial coefficients
print(coefs) 

def predict_y_value(x):
    y = -1.91 + 1.72*x + 0.288*(x**2) - 0.182*(x**3) + 0.016* (x**4)
    return y
def pred_y_val(x):
    y = intcp + coefs[0]*x + coefs[1]*x*x + coefs[2]*x**3
    return y
    
xx = np.linspace(1,7,50)                                 # Plot polynomial
yy = predict_y_value(xx)
y4 = pred_y_val(xx)
fig, ax = plt.subplots()
ax.scatter(mA,B)                                             # Plot points
plt.xlabel('Mass Number')
plt.ylabel('Binding Energy per nucleon')
plt.plot(xx, yy, c = "red", label="3rd degree poly")          # Solid line
plt.legend()
plt.plot(xx, y4, label ="4th degree poly")
plt.legend()
plt.show()