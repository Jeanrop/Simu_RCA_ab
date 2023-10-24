import numpy as np


def binarize(pred, gt):
    gt = np.clip(gt, 0, 1).astype('int')
    pred = np.clip(pred, 0, 1).astype('int')
    if len(gt.shape) == 2:
        gt = np.expand_dims(gt, axis=0)
        pred = np.expand_dims(pred, axis=0)
    dim = tuple(range(1, len(pred.shape)))
    return pred, gt, dim


class Metric():
    def __init__(self, name='undefined'):
        self.name = name

    def compute(self, pred=None, gt=None, evaluator=None, reduce=True):
        if evaluator:
            metric = self.compute_on_evaluator(evaluator)
        else:
            metric = self.compute_on_couple(pred, gt)

        if reduce:
            metric = np.mean(metric)

        return metric

    def get_name(self):
        return self.name

    def compute_on_evaluator(self, evaluator):
        raise NotImplementedError

    def compute_on_couple(self, pred, gt):
        raise NotImplementedError


class Dice(Metric):
    def __init__(self):
        super().__init__(name='Dice')

    def compute_on_evaluator(self, evaluator):
        dice = 2 * evaluator.intersection
        dice = dice / (evaluator.n_pixel_gt + evaluator.n_pixel_pred)
        return dice

    def compute_on_couple(self, pred, gt):
        pred, gt, dim = binarize(pred, gt)
        intersection = np.sum(gt * pred, axis=dim)
        n_pixel_pred = np.sum(pred, axis=dim)
        n_pixel_gt = np.sum(gt, axis=dim)
        dice = 2 * intersection / (n_pixel_gt + n_pixel_pred)
        return dice


class Recall(Metric):
    def __init__(self):
        super().__init__(name='Recall')

    def compute_on_evaluator(self, evaluator):
        recall = evaluator.intersection / evaluator.n_pixel_gt
        return recall

    def compute_on_couple(self, pred, gt):
        pred, gt, dim = binarize(pred, gt)
        intersection = np.sum(gt * pred, axis=dim)
        n_pixel_gt = np.sum(gt, axis=dim)
        recall = intersection / n_pixel_gt
        return recall


class Precision(Metric):
    def __init__(self):
        super().__init__(name='Precision')

    def compute_on_evaluator(self, evaluator):
        precision = evaluator.intersection / evaluator.n_pixel_pred
        return precision

    def compute_on_couple(self, pred, gt):
        pred, gt, dim = binarize(pred, gt)
        intersection = np.sum(gt * pred, axis=dim)
        n_pixel_pred = np.sum(pred, axis=dim)
        precision = intersection / n_pixel_pred
        return precision


class IoU(Metric):
    def __init__(self):
        super().__init__(name='IoU')

    def compute_on_evaluator(self, evaluator):
        iou = evaluator.intersection / evaluator.union
        return iou

    def compute_on_couple(self, pred, gt):
        pred, gt, dim = binarize(pred, gt)
        intersection = np.sum(gt * pred, axis=dim)
        union = np.sum(np.clip(pred + gt, 0, 1), axis=dim)
        return intersection / union


class Rmse(Metric):
    def __init__(self):
        super().__init__(name='Rmse_log10')

    def compute_on_evaluator(self, evaluator):
        return self.compute_on_couple(evaluator.pred, evaluator.gt)

    def compute_on_couple(self, pred, gt):
        if len(gt.shape) == 2:
            gt = np.expand_dims(gt, axis=0)
            pred = np.expand_dims(pred, axis=0)
        dim = tuple(range(1, len(pred.shape)))
        pred = np.log10(1+pred)
        gt = np.log10(1+gt)
        pred = pred / pred.max(axis=dim, keepdims=True)
        gt = gt / gt.max(axis=dim, keepdims=True)
        rmse = np.sqrt(np.mean((pred - gt) ** 2, axis=dim))
        return rmse


class Variance(Metric):
    def __init__(self):
        super().__init__(name='Variance')

    def compute_on_evaluator(self, evaluator):
        return self.compute_on_couple(evaluator.pred)

    def compute_on_couple(self, pred, gt=None):
        if len(pred.shape) == 2:
            pred = np.expand_dims(pred, axis=0)
        dim = tuple(range(1, len(pred.shape)))
        mean_pred = np.mean(pred, axis=dim)
        rmse = (pred - mean_pred) ** 2
        return rmse
