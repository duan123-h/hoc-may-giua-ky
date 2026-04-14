import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import combinations
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import joblib

df = pd.read_csv('airfoil_self_noise.dat', sep='\s+', header=None)
df.columns = [
    'Frequency', 
    'Angle_of_Attack', 
    'Chord_Length', 
    'Stream_Velocity', 
    'Displacement_Thickness', 
    'Y'
]
df['Frequency'] = np.log10(df['Frequency'])

X = df.drop(["Y"], axis=1) 
y = df.get(["Y"]) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    
    random_state=45   
)

y_test = y_test.values 
y_train = y_train.values 

scaler  = joblib.load('scaler.joblib')
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
w = np.load('weights_model.npy')

def create_features(X):
    n_samples, n_features = X.shape
    one = np.ones((n_samples, 1))
    interactions = [one]
    interactions.append(X)
    interactions.append(X**2)
    interactions.append(X**3)
    interactions.append(X**4)

    for i, j in combinations(range(n_features), 2):
        new_col = X[:, i] * X[:, j]
        interactions.append(new_col)
    for i, j in combinations(range(n_features), 2):
        new_col = (X[:, i]**2) * X[:, j]
        interactions.append(new_col)
    for i, j in combinations(range(n_features), 2):
        new_col = (X[:, i]) * (X[:, j]**2)
        interactions.append(new_col)
    for i, j in combinations(range(n_features), 2):
        new_col = (X[:, i]**3) * X[:, j]
        interactions.append(new_col)
    for i, j in combinations(range(n_features), 2):
        new_col = (X[:, i]) * (X[:, j]**3)
        interactions.append(new_col)
    for i, j in combinations(range(n_features), 2):
        new_col = (X[:, i]**4) * X[:, j]
        interactions.append(new_col)
    for i, j in combinations(range(n_features), 2):
        new_col = (X[:, i]) * (X[:, j]**4)
        interactions.append(new_col)
    return np.column_stack(interactions)

def get_predict(x,w):
    p=[]
    for i in range(0,len(x)):
        xi = x[i].reshape(-1,1)
        y_pred= (w.T @ xi).item() 
        p.append(y_pred)
    return p

X_train_poly = create_features(X_train)
X_test_poly = create_features(X_test)

y_pred_final = get_predict(X_test_poly,w)
def relative_error(y_true, y_pred):
    errors = np.abs(y_pred - y_true).astype(float) / y_true
    return np.mean(errors)*100
print ('RMSE: {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred_final))))
print ('Mean relative errors: {:.1f}%'.format(relative_error(y_test, y_pred_final)))
plt.figure(figsize=(10,6))
plt.plot(y_test[:100], color='blue', label='Actual')
plt.plot(y_pred_final[:100], color='red', label='Predicted')
plt.title(f'Final model after iterations')
plt.xlabel('Sample index test')
plt.ylabel('Y')
plt.show()