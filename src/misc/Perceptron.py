""" From "COMPUTATIONAL PHYSICS: PROBLEM SOLVING with PYTHON"
    4th Ed. by RH Landau, MJ Paez, and CC Bordeianu (D)
    Copyright R Landau, Oregon State Unv, MJ Paez, Univ Antioquia, 
    C Bordeianu, Univ Bucharest, 2024. 
    Please respect copyright & acknowledge our work."""

# Perceptron.py: Creat perceptron with sklearn 
 
import pandas as pd                                     # To read dataset
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

parts = pd.read_table("C:particle.dat",delim_whitespace=True)
X = parts["Mass"]                                            # X: masses
y = parts['T']                                                 # y: Type
print('Class labels:', np.unique(y))                     # The 4 classes
d = {'col1':X, 'col2':y}                         # d: 2-D array of X & y
dfrom sklearn.model_selection import train_test_split      # Split array
X_train, X_test, y_train, y_test = train_test_split(
df, y, test_size=0.3, random_state=1, stratify=y)   # Form 2-D dataframe
from sklearn.model_selection import train_test_split       # Split array

# Shuffle data dataf=pd.DataFrame(d)
X_train, X_test, y_train, y_test = train_test_split(
df, y, test_size=0.3, random_state=1, stratify=y)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std,y_train)            # Fit data
y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots()
plt.xlabel("mass")
plt.ylabel("Type")

for i in range(36): # Plot spin (0, 1, 3/2, 1/2) vs mass
    if y[i] == 0: plt.scatter(X[i],y[i], c='red',marker='x',s=150)
    if y[i] == 1: plt.scatter(X[i],y[i], c='blue',marker="^",s=150)
    if y[i] == 3: plt.scatter(X[i],y[i], c='brown', marker =">",s=150)
    if y[i] == 2: plt.scatter(X[i],y[i], c='magenta', marker="<",s=150)
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.01):
  markers = ('s', 'x', 'o', '^', 'v')       # Markers for and color map
  colors = ('brown', 'peachPuff', 'lightgreen', 'gold', 'cyan')
  cmap = ListedColormap(colors[:len(np.unique(y))])
  x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
             np.arange(x2_min, x2_max, resolution)) # Decision surface
  Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
  Z = Z.reshape(xx1.shape) 
  plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap) # Alpha: Transp
  plt.xlim(xx1.min(), xx1.max())
  plt.ylim(xx2.min(), xx2.max())
  for idx, cl in enumerate(np.unique(y)):
       plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8,\
        c=colors[idx], marker=markers[idx], label=cl,edgecolor='black')
       if test_idx:                              # Highlight test examples
         X_test, y_test = X[test_idx, :], y[test_idx]
         plt.scatter(X_test[:, 0], X_test[:, 1],edgecolor='black',\
          alpha=1.0,linewidth=1, marker='o',s=100, label='test set')
plt.show()