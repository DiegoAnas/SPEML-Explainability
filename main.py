import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

if __name__ == "__main__":
    featureNames = ["seq", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "loc"]
    yeastData = pd.read_csv("yeast.data", sep=" ", names=featureNames)
    # plt.boxplot(yeastData['mcg'], vert=False)
    # plt.ylabel('mcg')
    # plt.boxplot(yeastData['nuc'], vert=False)
    # ax = sns.stripplot(data=yeastData, jitter=0.2, size=2.5, orient="h")
    # ax = sns.boxplot(data = yeastData["nuc"], orient="h", notch=True)
    # plt.subplot(2, 1, 1)
    # ax = sns.violinplot(data=yeastData.iloc[:,[1,2,3,4,7,8]], orient="v")
    # plt.title("Violin plots of Yeast dataset attributes")
    # plt.subplot(2, 1, 2)
    # plt.title("Class distribution of Yeast dataset")
    # ax = sns.countplot(yeastData["loc"])
    # plt.tight_layout()
    # plt.show()

    # models = (GradientBoostingClassifier(n_estimators=100, max_features=None, max_depth=2, random_state=5),
    #           GradientBoostingClassifier())
    model = GradientBoostingClassifier(n_estimators=100, max_features=None, max_depth=2, random_state=5)
    # first works better
    # GradientBoostingClassifier(n_estimators=20, learning_rate = 0.1, max_features=NONE=all, max_depth = 2, random_state = 0)
    # titles = ("GradientBoost1",
    #           "GradientBoost2")
    # kFold = KFold(n_splits=3, shuffle=False, random_state=39)
    kFold = KFold(n_splits=2, shuffle=False, random_state=39)
    yeastAttrib = yeastData.iloc[:,1:9].values  # fix column indexes
    yeastTarget = yeastData["loc"].values
    print(f"yeast shape {yeastAttrib.shape} target {yeastTarget.shape}")
    fold = 1
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
    for train_index, test_index in kFold.split(yeastAttrib):
        print(f"------------"
              f"Fold {fold}")
        fold += 1
        # for model, title in zip(models, titles):
        train_data, train_target = yeastAttrib[train_index], yeastTarget[train_index]
        test_data, test_target = yeastAttrib[test_index], yeastTarget[test_index]
        clf = model.fit(train_data, train_target)
        prediction = clf.predict(test_data)
        print(classification_report(test_target, prediction))
        print(f"Confusion Matrix: \n {confusion_matrix(test_target, prediction)}")

        interpreter = Interpretation(test_data, feature_names=featureNames[1:9])
        model_no_proba = InMemoryModel(model.predict, examples=test_data, unique_values=model.classes_)
        ax = interpreter.feature_importance.plot_feature_importance(model_no_proba, ascending=False)
        plt.show()
    exit(0)