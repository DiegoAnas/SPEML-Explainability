import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from skater.core.global_interpretation.tree_surrogate import TreeSurrogate
#from skater.util.dataops import show_in_notebook ##??

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
    #     train_data, train_target = yeastAttrib[train_index], yeastTarget[train_index]
    #     test_data, test_target = yeastAttrib[test_index], yeastTarget[test_index]
    #     for model, title in zip(models, titles):
    #         clf = model.fit(train_data, train_target)
    #         prediction = clf.predict(test_data)
    #         print(f"{title}")
    #         print(classification_report(test_target, prediction))
    #         print(f"Confusion Matrix: \n {confusion_matrix(test_target, prediction)}")
    #
    #         ax = axs[modelno - 1, fold - 1]
    #         interpreter = Interpretation(test_data, feature_names=featureNames[1:9])
    #         # model_no_proba = InMemoryModel(model.predict, examples=test_data, unique_values=model.classes_)
    #         model_mem = InMemoryModel(model.predict_proba, examples=test_data)
    #         interpreter.feature_importance.plot_feature_importance(model_mem, ascending=False, ax=ax)
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
    #         # model_no_proba = InMemoryModel(model.predict, examples=test_data, unique_values=model.classes_)
    #         model_mem = InMemoryModel(model.predict_proba, examples=test_data)
    #         interpreter.feature_importance.plot_feature_importance(model_mem, ascending=False, ax=ax)
    #         ax.set_title(f"{title} on fold {fold}")
    #         print("\n")
    #         modelno += 1
    #     fold += 1
    # plt.tight_layout()

    for train_index, test_index in kFold.split(yeastAttrib):
        print(f"------------"
              f"Fold {fold}")
        modelno = 1
        train_data, train_target = yeastAttrib[train_index], yeastTarget[train_index]
        test_data, test_target = yeastAttrib[test_index], yeastTarget[test_index]
        for model, title in zip(models, titles):
            clf = model.fit(train_data, train_target)
            prediction = clf.predict(test_data)
            print(f"{title}")
            print(classification_report(test_target, prediction))
            print(f"Confusion Matrix: \n {confusion_matrix(test_target, prediction)}")

            ax = axs[modelno - 1, fold - 1]
            interpreter = Interpretation(test_data, feature_names=featureNames[1:9])
            # model_no_proba = InMemoryModel(model.predict, examples=test_data, unique_values=model.classes_)
            pyint_model = InMemoryModel(model.predict_proba, examples=test_data,
                                        target_names=["CYT", "ME3", "MIT", "NUC"])
            interpreter.feature_importance.plot_feature_importance(pyint_model, ascending=False, ax=ax,
                                                                   progressbar=False)
            ax.set_title(f"{title} on fold {fold}")
            print("\n")

            ## To avoid clutter I only produce plots for gradient boosting and one fold only
            if (fold == 2 and modelno == 1):
                # Plot PDPs of variable "alm" since it is the most important feature, for 3 of the 4 models
                ## Not for Gaussian Naive bayes tho, explain that
                # for other variables just change the name
                # for other models just change the number
                # interpreter.partial_dependence.plot_partial_dependence(["alm"],
                #                                                        pyint_model, grid_resolution=30,
                #                                                        with_variance=True)
                # # PDP interaction between two variables, for each class
                # interpreter.partial_dependence.plot_partial_dependence([("nuc", "mit")], pyint_model,
                #                                                        grid_resolution=10)
                surrogate_explainer = interpreter.tree_surrogate(oracle=pyint_model, seed=5)
                surrogate_explainer.fit(train_data, train_target, use_oracle=True, prune='post', scorer_type='default')
                surrogate_explainer.plot_global_decisions(file_name='simple_tree_class.png', fig_size=(8, 8))
                #show_in_notebook('simple_tree_pre.png', width=400, height=300)
                # This initialization, although showcased on the docs, does not work
                # surrogate_explainer = interpreter.tree_surrogate(estimator_type_='classifier',
                #                                                 feature_names=featureNames[1:9],
                #                                                 class_names=["CYT", "ME3", "MIT", "NUC"], seed=5)
                # y_hat_train = model.predict(train_data)
                # y_hat = models['gb'].predict(test_data)
                # print(f"""Surrogate score:
                #       {surrogate_explainer.learn(train_data, y_hat_train, oracle_y=train_target, cv=True)}""")
        # couldnt figure how to put it into one subplot, since it plots directly
            modelno += 1
        fold += 1
    plt.show()
    exit(0)