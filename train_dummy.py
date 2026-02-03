import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

with open("models/best_random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Model saved successfully")
