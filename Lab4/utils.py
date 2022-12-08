import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from pprint import pprint

from ID3Solver import ID3Solver


def get_metrics(y_true, y_pred, printing=True):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)

    if printing:
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1_score)

    return [accuracy, precision, recall, f1_score]


def plot_history(train_history, validation_history, metric):
    plt.figure(figsize=(7, 5))
    plt.plot(train_history, linewidth=2, label="Training set")
    plt.xticks(np.arange(0, len(train_history)), np.arange(1, len(train_history) + 1))
    plt.plot(validation_history, linewidth=2, label="Validation set")
    plt.xticks(np.arange(0, len(validation_history)), np.arange(1, len(validation_history) + 1))
    plt.xlabel('depth', fontsize=12)
    plt.ylabel(f'{metric}', fontsize=12)
    plt.legend(loc="upper left", fontsize=10)
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    cf_matrix = metrics.confusion_matrix(y_test, y_pred)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, center=True, annot=labels, fmt="", cmap='viridis')
    plt.title("Confusion matrix for test set")
    plt.show()


def determine_the_best_model(X_train, y_train, X_val, y_val, X_test, y_test, depth_range=11, main_metric="f1_score", print_model=False, plot_results=True, plot_conf_matrix=True):
    best_model = None
    best_metric_score = 0
    train_history = []
    validation_history = []

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

    for depth in range(1, depth_range + 1):
        print(f"\nDepth {depth}")
        print(f"--------------------------------------------------------------------------------")

        model = ID3Solver(depth)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        print(f"\nTraining Set")
        metric_score_train = get_metrics(y_train, y_pred_train)[metric_type]

        print(f"\nValidation Set")
        metric_score_val = get_metrics(y_val, y_pred_val)[metric_type]

        train_history.append(metric_score_train)
        validation_history.append(metric_score_val)

        if metric_score_val > best_metric_score:
            best_metric_score = metric_score_val
            best_model = model

    if plot_results:
        plot_history(train_history, validation_history, main_metric)


    print(f"\nBest Model")
    print(f"--------------------------------------------------------------------------------")

    print(f"\nValidation Set")

    if print_model:
        print("Best model: ")
        pprint(best_model.tree)

    print("Best depth: ", best_model.depth)
    print(f"Best {main_metric}: ", best_metric_score)


    print(f"\nTest Set")

    y_pred_test = best_model.predict(X_test)
    get_metrics(y_test, y_pred_test)[metric_type]

    if plot_conf_matrix:
        plot_confusion_matrix(y_test, y_pred_test)

    return best_model
