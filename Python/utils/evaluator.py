import numpy as np

from utils.metrics import Dice, Recall, Precision, IoU, Rmse


class Evaluator(object):
    def __init__(self, metric_list):
        self.metrics = None
        self.stored_metric = {}

        # to avoid recomputation of intersection or union
        # they will be computed once at the beginning of
        # the evalutation if needed and then shared between metrics
        self.intersection_required = False
        self.union_required = False
        self.n_pixel_pred_required = False
        self.n_pixel_gt_required = False

        for metric in metric_list:
            self.add_metric(metric.lower())
            print(metric)

    def add_metric(self, metric):
        if self.metrics is None:
            self.metrics = []

        if metric == 'rmse':
            metric = Rmse()
            self.metrics.append(metric)
        if metric == 'dice':
            metric = Dice()
            self.metrics.append(metric)
            self.intersection_required = True
            self.n_pixel_gt_required = True
            self.n_pixel_pred_required = True

        if metric == 'recall':
            metric = Recall()
            self.metrics.append(metric)
            self.intersection_required = True
            self.n_pixel_gt_required = True

        if metric == 'precision':
            metric = Precision()
            self.metrics.append(metric)
            self.intersection_required = True
            self.n_pixel_pred_required = True

        if metric == 'iou':
            metric = IoU()
            self.metrics.append(metric)
            self.intersection_required = True
            self.union_required = True
        metric_name = metric.get_name()
        self.stored_metric[metric_name] = []

    def evaluate(self, pred, ground_truth):
        # we compute the values shared between metrics
        self.process(pred, ground_truth)
        evaluation_results = {}
        for metric in self.metrics:
            metric_value = metric.compute(evaluator=self)
            self.stored_metric[metric.name].append(metric_value)
            evaluation_results[metric.name] = metric_value
        return evaluation_results

    def process(self, pred, gt):
        self.pred, self.gt = pred, gt
        bin_gt = np.clip(gt, 0, 1).astype('int')
        bin_pred = np.clip(pred, 0, 1).astype('int')
        if len(bin_gt.shape) == 2:
            bin_gt = np.expand_dims(bin_gt, axis=0)
            bin_pred = np.expand_dims(bin_pred, axis=0)
        dim = tuple(range(1, len(bin_pred.shape)))

        if self.intersection_required:
            self.intersection = np.sum(bin_gt * bin_pred, axis=dim)

        if self.union_required:
            self.union = np.sum(np.clip(bin_pred + bin_gt, 0, 1), axis=dim)

        if self.n_pixel_pred_required:
            self.n_pixel_pred = np.sum(bin_pred, axis=dim)

        if self.n_pixel_gt_required:
            self.n_pixel_gt = np.sum(bin_gt, axis=dim)

        self.bin_pred, self.bin_gt, self.dim = bin_pred, bin_gt, dim
