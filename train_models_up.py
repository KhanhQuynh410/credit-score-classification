import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

data = np.load(
    "features_relief_5_up.npy",
    allow_pickle=True
).item()

X_train = data["X_train"]
y_train = data["y_train"] + 1

models = {

    "svm_up": SVC(
        kernel='linear',
        probability=True
    ),

    "knn_up": KNeighborsClassifier(
        n_neighbors=5
    ),

    "lda_up": LinearDiscriminantAnalysis(),

    "nb_up": GaussianNB()
}

for name, model in models.items():

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        f"{name}.pkl"
    )

print("DONE")