import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# load feature
data = np.load("features_relief_5.npy", allow_pickle=True).item()

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"] + 1
y_test = data["y_test"] + 1

models = {
    "svm": SVC(kernel='linear', probability=True),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "lda": LinearDiscriminantAnalysis(),
    "nb": GaussianNB()
}

for name, model in models.items():

    print(f"Training {name}")

    model.fit(X_train, y_train)

    joblib.dump(model, f"{name}.pkl")

print("DONE")