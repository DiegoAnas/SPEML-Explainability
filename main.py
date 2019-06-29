import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

if __name__ == "__main__":
    featureNames = ["seq", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "loc"]
    yeastData = pd.read_csv("yeast.data", sep=" ", names=featureNames)
    titles = ("GradientBoost", "KNN", "Gaussian", "Random Forest")  # add more
    models = (GradientBoostingClassifier(n_estimators=100, max_features=None, max_depth=2, random_state=5),
              KNeighborsClassifier(),
              GaussianNB(),
              RandomForestClassifier())
    kFold = KFold(n_splits=2, shuffle=False, random_state=39)

    yeastAttrib = yeastData.iloc[:,1:9].values  # fix column indexes
    yeastTarget = yeastData["loc"].values
    fold = 1
    # Using sklearn PDP, only avaiblable for GraBoost
    # for train_index, test_index in kFold.split(yeastAttrib):
    #     print(f"------------"
    #           f"Fold {fold}")
    #     fold += 1
    #     # for model, title in zip(models, titles):
    #     train_data, train_target = yeastAttrib[train_index], yeastTarget[train_index]
    #     test_data, test_target = yeastAttrib[test_index], yeastTarget[test_index]
    #     clf = model.fit(train_data, train_target)
    #     prediction = clf.predict(test_data)
    #     print(classification_report(test_target, prediction))
    #     print(f"Confusion Matrix: \n {confusion_matrix(test_target, prediction)}")
    #     features = [0, 1, 2]
    #     plot_partial_dependence(clf, train_data, features, target="CYT")
    #     plt.show()

    # fig, axs = plt.subplots(len(models), kFold.n_splits)
    # for train_index, test_index in kFold.split(yeastAttrib):
    #     print(f"------------"
    #           f"Fold {fold}")
    #     modelno = 1
    #     for model, title in zip(models, titles):
    #         train_data, train_target = yeastAttrib[train_index], yeastTarget[train_index]
    #         test_data, test_target = yeastAttrib[test_index], yeastTarget[test_index]
    #         clf = model.fit(train_data, train_target)
    #
    #         prediction = clf.predict(test_data)
    #         print(f"{title}")
    #         print(classification_report(test_target, prediction))
    #         print(f"Confusion Matrix: \n {confusion_matrix(test_target, prediction)}")
    #
    #         ax = axs[modelno - 1, fold - 1]
    #         interpreter = Interpretation(test_data, feature_names=featureNames[1:9])
    #         model_no_proba = InMemoryModel(model.predict, examples=test_data, unique_values=model.classes_)
    #         interpreter.feature_importance.plot_feature_importance(model_no_proba, ascending=False, ax=ax)
    #         ax.set_title(f"{title} on fold {fold}")
    #         print("\n")
    #         modelno += 1
    #     fold += 1
    # plt.tight_layout()
    # plt.show()

    yeast4Classes = yeastData.loc[(yeastData["loc"] == "CYT")|( yeastData["loc"] == "NUC" )| (yeastData["loc"] == "MIT" )| (yeastData["loc"] == "ME3")]
    yeastAttrib = yeast4Classes.iloc[:, 1:9].values  # fix column indexes
    yeastTarget = yeast4Classes["loc"].values
    plt.subplot(1,2,1)
    ax = sns.violinplot(data=yeast4Classes.iloc[:, [1, 2, 3, 4, 7, 8]], orient="v")
    plt.subplot(1, 2, 2)
    ax = sns.violinplot(data=yeastData.iloc[:, [1, 2, 3, 4, 7, 8]], orient="v")
    plt.show()
    fold = 1
    fig, axs = plt.subplots(len(models), kFold.n_splits)
    for train_index, test_index in kFold.split(yeastAttrib):
        print(f"------------"
              f"Fold {fold}")
        modelno = 1
        for model, title in zip(models, titles):
            train_data, train_target = yeastAttrib[train_index], yeastTarget[train_index]
            test_data, test_target = yeastAttrib[test_index], yeastTarget[test_index]
            clf = model.fit(train_data, train_target)

            prediction = clf.predict(test_data)
            print(f"{title}")
            print(classification_report(test_target, prediction))
            print(f"Confusion Matrix: \n {confusion_matrix(test_target, prediction)}")

            ax = axs[modelno - 1, fold - 1]
            interpreter = Interpretation(test_data, feature_names=featureNames[1:9])
            model_no_proba = InMemoryModel(model.predict, examples=test_data, unique_values=model.classes_)
            interpreter.feature_importance.plot_feature_importance(model_no_proba, ascending=False, ax=ax)
            ax.set_title(f"{title} on fold {fold}")
            print("\n")
            modelno += 1
        fold += 1
    plt.tight_layout()
    plt.show()
    exit(0)