import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from alepython.ale import ale_plot


class one_class_prob_pred:
    def __init__(self, predictor: callable, class_no: int):
        self.predictor = predictor
        self.class_no = class_no

    def predict(self, X, **kwargs):
        return self.predictor(X,**kwargs)[:,self.class_no]


if __name__ == "__main__":
    featureNames = ["seq", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "loc"]
    yeastData = pd.read_csv("yeast.data", sep=" ", names=featureNames)
    titles = ("GradientBoost", "KNN", "Gaussian", "Random Forest", "MLP")  # add more
    models = (GradientBoostingClassifier(n_estimators=100, max_features=None, max_depth=2, random_state=5),
              KNeighborsClassifier(),
              GaussianNB(),
              RandomForestClassifier(),
              MLPClassifier())

    yeast4Classes = yeastData.loc[(yeastData["loc"] == "CYT")|( yeastData["loc"] == "NUC" )| (yeastData["loc"] == "MIT" )| (yeastData["loc"] == "ME3")]
    yeastAttrib = yeast4Classes.iloc[:, 1:9]
    yeastTarget = yeast4Classes["loc"]
    X_train, X_test, y_train, y_test = train_test_split(yeastAttrib, yeastTarget, test_size = 0.33, random_state = 42)

    for model, title in zip(models, titles):
        model.fit(X_train, y_train)
        # print(model.predict_proba(X_test.iloc[1:3]))
        # print(predictor.predict(X_test.iloc[1:2]))
        for loc in range(len(set(y_train))):
            print(model.classes_[loc])
            predictor = one_class_prob_pred(model.predict_proba, loc)
            # ale_plot(model, X_train, "alm", predictor=predictor.predict, monte_carlo=True)
            ale_plot(model, X_train, ["mit", "alm"], predictor=predictor.predict,
                     monte_carlo=True, monte_carlo_ratio= 0.3)
    plt.show()