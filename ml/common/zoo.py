"""The candidate classifiers benchmarked for the clinical (tabular) modules."""
from __future__ import annotations

from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def tabular_models(rs: int = 42) -> dict:
    def lr():
        return LogisticRegression(max_iter=3000)

    def rf():
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                      random_state=rs)

    def gb():
        return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                          subsample=0.9, random_state=rs)

    def xgb():
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", random_state=rs,
                             n_jobs=4, verbosity=0)

    def svm():
        return SVC(kernel="rbf", C=1.0, probability=True, random_state=rs)

    return {
        "Logistic Regression": lr(),
        "SVM": svm(),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=rs),
        "Random Forest": rf(),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                            random_state=rs),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, algorithm="SAMME",
                                       random_state=rs),
        "Gradient Boosting": gb(),
        "XGBoost": xgb(),
        "Voting Ensemble": VotingClassifier(
            estimators=[("lr", lr()), ("rf", rf()), ("gb", gb()), ("xgb", xgb()), ("svm", svm())],
            voting="soft"),
        "Stacking Ensemble": StackingClassifier(
            estimators=[("lr", lr()), ("rf", rf()), ("gb", gb()), ("xgb", xgb())],
            final_estimator=LogisticRegression(max_iter=3000), cv=5),
    }
