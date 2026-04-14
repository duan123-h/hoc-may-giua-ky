import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import combinations
from sklearn.metrics import mean_squared_error
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

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.joblib')
print("Đã lưu bộ chuẩn hóa vào file scaler.joblib")

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

n_samples, d = X_train_poly.shape
w = np.zeros((d, 1))
theta = 1e-4
eta0 =0.00002

k = 0
graph = True

plt.figure(figsize=(10,6))
while True:
    k += 1
    eta=eta0*(1/(1 + 0.000001 * k))
    w_old = w.copy()

    for i in range(n_samples):
        xi = X_train_poly[i].reshape(-1,1)
        w = w+ eta * (y_train[i] - w.T @ xi).item() * xi
    
    if graph and k % 50 == 0:
        y_pred = get_predict(X_train_poly,w)
        plt.cla()
        plt.plot(y_train[:100], color='blue', label='Actual')
        plt.plot(y_pred[:100], color='red', label='Predicted')
        plt.title(f'Iteration {k}')
        plt.xlabel('Sample index')
        plt.ylabel('Y')
        plt.legend()
        plt.pause(0.1)

    if np.linalg.norm(w - w_old) < theta and k>5000:
        break
    if  k>100000:
        break
np.save('weights_model.npy', w)
print("Đã lưu bộ trọng số vào file weights_model.npy")

y_pred_final =get_predict(X_train_poly,w)
plt.clf()
plt.plot(y_train[:100], color='blue', label='Actual')
plt.plot(y_pred_final[:100], color='red', label='Predicted')
plt.title(f'Final model after {k} iterations')
plt.xlabel('Sample index')
plt.ylabel('Y')
plt.legend()
plt.show()
