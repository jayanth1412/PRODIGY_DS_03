from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib


def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/decision_tree_model.pkl")

    return model, X_test, y_test