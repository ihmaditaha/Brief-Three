import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_classification_models():
    return {
        "Logistic Regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000)),
                ]
            ),
            {
                'model__solver': ['lbfgs'],
                'model__l1_ratio': [0],
                'model__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
            }
,
        ),
        "Decision Tree": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", DecisionTreeClassifier(random_state=42)),
                ]
            ),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [3, 5, 10, 15, 20],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", RandomForestClassifier(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [3, 10, 20],
                "model__min_samples_split": [2, 5],
                "model__max_features": ["sqrt", "log2", None],
            },
        ),
        "SVM": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(random_state=42)),
                ]
            ),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", "auto", 0.01, 0.1],
                "model__kernel": ["rbf", "linear"],
            },
        ),
    }


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def tune_and_evaluate(X_train, y_train, X_test, y_test, experiment_name="Classification_experiment"):
    mlflow.set_experiment(experiment_name)

    models = get_classification_models()
    best_overall_model = None
    best_overall_score = 0.0
    best_model_name = ""
    results = []

    for name, (pipeline, param_grid) in models.items():
        with mlflow.start_run(run_name=name):
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                scoring="f1_weighted",
                error_score="raise",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.predict(X_test)
            score = grid_search.best_score_
            c_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_f1_weighted", score)
            mlflow.log_metric("test_accuracy", report["accuracy"])
            mlflow.sklearn.log_model(grid_search.best_estimator_, name="model",skops_trusted_types=["sklearn.tree._tree.Tree"])

            results.append(
                {
                    "name": name,
                    "best_params": grid_search.best_params_,
                    "score": score,
                    "confusion_matrix": c_matrix,
                    "classification_report": report,
                }
            )

            if score > best_overall_score:
                best_overall_score = score
                best_overall_model = grid_search.best_estimator_
                best_model_name = name

    return best_overall_model, best_model_name, best_overall_score, results