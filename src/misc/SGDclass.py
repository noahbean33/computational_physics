""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""
	 
# SGDclass.py:  ML via Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier         # StochGradDescent
from sklearn.inspection import DecisionBoundaryDisplay       # Def region
import pandas as pd                                     # To read dataset
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline                             # Set matplot for notebook
parts = pd.read_table("part.dat",delim_whitespace=True)       # Read data
X = parts["Mass"]                                            #  X: masses
y = parts['Type']                                     #  Types (integers)
print('Class labels:', np.unique(y))                          # 4 classes
d = {'col1':x, 'col2':y}                             # 2 column X,y array
df = pd.DataFrame(d)                                  # Form 2d DataFrame
X = np.array(df)                                # DataFrame to numpy array
idx = np.arange(X.shape[0])                                   # Index 0-35
np.random.seed(13)
np.random.shuffle(idx)                              # Random index shuffle
X = X[idx]                                                # Random X order
y = y[idx]                                                # Random Y order
colors = "bryg"                                           # 4 class colors
mean = X.mean(axis=0)                                          # Calc mean
std  = X.std(axis=0)
X = (X - mean)/std                                          # Now mean = 0
print("mean std", mean,std)
lrgd = SGDClassifier(alpha=0.001, max_iter=100).fit(X,y)
print(lrgd)                              # Alpha: regularization strength
ax = plt.gca()
disp = DecisionBoundaryDisplay.from_estimator(lrgd,X, cmap = plt.cm.Paired, ax = ax, response_method = "predict", xlabel = "massMeV/c2",ylabel="Type")
plt.axis("tight")
print("lclasses", lrgd.classes_)                               # 4 classes        
for i, color in zip(lrgd.classes_, colors):         # Plot training points             
    idx = np.where(y == i)
    print("scatter", X[idx,0], X[idx,1])
    plt.scatter(X[idx,0],X[idx,1], c=color, cmap=plt.cm.Paired,
                edgecolor="black",s=20)
plt.axis("tight")
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = lrgd.coef_                         # Average weights for all steps
intercept = lrgd.intercept_

def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c,0]) - intercept[c]) / coef[c,1],
    plt.plot([xmin, xmax], [line(xmin), line(xmax)],
                          ls="--", color=color)
print(lrgd.classes_)
for i, color in zip(lrgd.classes_, colors):                 # Plot lines
    print(i,color)
    plot_hyperplane(i, color)
plt.legend()
plt.show()