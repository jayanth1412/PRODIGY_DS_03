import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import pandas as pd
import os


def plot_confusion_matrix(cm):

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d'
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")

    plt.show()


def plot_decision_tree(model, feature_names):

    plt.figure(figsize=(30, 12))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        fontsize=8
    )

    plt.title("Decision Tree Classifier")

    plt.savefig("outputs/decision_tree.png")
    plt.show()


def plot_feature_importance(model, feature_names):

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance
    )

    plt.title("Feature Importance")

    plt.savefig("outputs/feature_importance.png")
    plt.show()