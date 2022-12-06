from sklearn import metrics
from pprint import pprint

from ID3Solver import ID3Solver


def get_metrics(y_true, y_pred, printing=True):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.recall_score(y_true, y_pred)

    if printing:
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1_score)

    return [accuracy, precision, recall, f1_score]


def determine_the_best_model(target_label, X_train, y_train, X_val, y_val, X_test, y_test, depth_range=7,
                             main_metric="f1_score", print_model=False):
    best_model = None
    best_metric_score = 0

    print(f"\n************************** Validation set ******************************\n")

    for depth in range(1, depth_range + 1):
        print("  ")
        print(f"Depth {depth}")
        print(f"--------------------------------------------------------------------------------")
        model = ID3Solver(depth)
        model.fit(X_train, y_train, target_label)
        y_pred = model.predict(X_val)

        if main_metric == "accuracy":
            metric_type = 0
        elif main_metric == "precision":
            metric_type = 1
        elif main_metric == "recall":
            metric_type = 2
        elif main_metric == "f1_score":
            metric_type = 3
        else:
            raise "Invalid metric"

        metric_score = get_metrics(y_val, y_pred)[metric_type]

        if metric_score > best_metric_score:
            best_metric_score = metric_score
            best_model = model

    print(f"\n************************** Best model ******************************\n")

    if print_model:
        print("Best model: ")
        pprint(best_model.tree)

    print("Best depth: ", best_model.depth)
    print(f"Best {main_metric}: ", best_metric_score)

    print(f"\n************************** Test set ******************************\n")

    y_pred_test = best_model.predict(X_test)
    get_metrics(y_test, y_pred_test)[metric_type]