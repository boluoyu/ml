import numpy as np

from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score


class Evaluator:
    name = 'evaluator'
    precision_intervals = [d for d in np.arange(0.0, 1.05, 0.05)]

    def evaluate(self, class_map, y_pred, y_true):
        evaluation_metrics = []

        for category, category_ix in class_map.items():
            evaluation_metrics.append(self.get_category_metrics(category, category_ix, y_true=y_true, y_pred=y_pred))

    def get_category_metrics(self, category, category_ix, y_true, y_pred):
        category_predicted_labels = [1 if np.argmax(yi_pred) == category_ix else 0 for yi_pred in y_pred] # 1 if category was predicted, 0 if not
        category_y_true = [1 if np.argmax(yi_true) == category_ix else 0 for yi_true in y_true]

        f1score = self.compute_f1score(y_true=category_y_true, y_pred=category_predicted_labels)
        roc_auc = self.compute_roc_auc(y_true=category_y_true, y_pred=category_predicted_labels)
        precision_recall_threshold_choices = self.compute_precision_recall_threshold_choices(
            y_true=category_y_true,
            y_pred=category_predicted_labels
        )

        return dict(
            category=category,
            fscore=f1score,
            roc_auc=roc_auc,
            precision_recall_threshold_choices=precision_recall_threshold_choices
        )

    def compute_roc_auc(self, y_true, y_pred):
        roc_auc = roc_auc_score(y_true, y_pred)
        return roc_auc

    def compute_f1score(self, y_true, y_pred):
        fscore = f1_score(y_true=y_true, y_pred=y_pred)
        return fscore

    def compute_precision_recall_threshold_choices(self, y_true, y_pred):
        precision_recall_curve_threshold_values = []

        pr_curve = precision_recall_curve(y_true, y_pred)

        for target_precision in self.precision_intervals:
            precision, recall, threshold = self.get_point_at_desired_precision(pr_curve, target_precision)
            precision_recall_curve_threshold_values.append(
                dict(precision=precision, recall=recall, threshold=threshold))

        return precision_recall_curve_threshold_values

    def get_point_at_desired_precision(self, precision_recall_curve, desired_precision=0.8):
        for precision, recall, threshold in zip(*precision_recall_curve):
            if precision >= desired_precision:
                return precision, recall, threshold
        return -1, -1, -1
