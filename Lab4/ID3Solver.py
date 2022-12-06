import numpy as np


class ID3Solver():
    def __init__(self, depth=float('inf')):
        self.target_label = None
        self.tree = None
        self.depth = depth

    def get_parameters(self):
        return {
            "depth": self.depth
        }

    def fit(self, X, y, target_label):
        self.target_label = target_label
        X[self.target_label] = y
        X.dropna(inplace=True)
        self.tree = self.create_tree(X, self.depth)

    def get_entropy(self, feature_data):
        feature_elements, feature_counts = np.unique(feature_data, return_counts=True)
        entropy = np.sum(
            [(-feature_counts[i] / np.sum(feature_counts)) * np.log2(feature_counts[i] / np.sum(feature_counts)) for i
             in range(len(feature_elements))])
        return entropy

    def get_info_gain(self, data, decision_feature):
        total_entropy = self.get_entropy(data[self.target_label])
        feature_values, feature_counts = np.unique(data[decision_feature], return_counts=True)

        for i in range(len(feature_values)):
            probability = feature_counts[i] / np.sum(feature_counts)
            feature_data = data[data[decision_feature] == feature_values[i]][self.target_label]
            total_entropy -= self.get_entropy(feature_data) * probability

        return total_entropy

    def get_decision_feature(self, data):
        features = data.columns.drop(self.target_label)
        info_gain_values = [self.get_info_gain(data, feature) for feature in features]
        decision_feature = features[np.argmax(info_gain_values)]
        return decision_feature

    def create_tree(self, data, depth):
        if data.shape[0] == 0:
            return {}

        if depth == 0:
            return data[self.target_label].value_counts().idxmax()

        if len(np.unique(data[self.target_label])) == 1:
            return np.unique(data[self.target_label])[0]

        decision_feature = self.get_decision_feature(data)

        tree = {decision_feature: {}}

        for feature_value in np.unique(data[decision_feature]):
            subset = data[data[decision_feature] == feature_value]
            subset.pop(decision_feature)
            subtree = self.create_tree(subset, depth - 1)
            tree[decision_feature][feature_value] = subtree

        return tree

    def predict_item(self, tree, item):
        if isinstance(tree, dict):
            decision_feature = next(iter(tree))
            value = item[decision_feature]

            if value in tree[decision_feature]:
                return self.predict_item(tree[decision_feature][value], item)
            else:
                return False

        else:
            return tree

    def predict(self, X):
        prediction = []
        for i in range(X.shape[0]):
            prediction.append(self.predict_item(self.tree, X.iloc[i]))
        return prediction

