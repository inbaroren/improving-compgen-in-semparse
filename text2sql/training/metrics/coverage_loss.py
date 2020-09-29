import numpy as np


def calculate_coverage_loss(data, mask):
    # shape: steps, input
    attention = np.stack([step_info['question_attention'].detach().cpu() for step_info in data])
    if mask is None:
        mask = np.ones_like(attention)
    else:
        # mask is a list that marks what steps produced a terminal. convert to an array
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=1)

    coverage_vectors = attention.cumsum(0)
    coverage_vectors = coverage_vectors[:-1,:]
    attention = attention[1:,:]
    mask = mask[1:,:]

    # the loss is the sum over the minimum weight given to an input token by the coverage vector
    # (sum of attention vector to current step) or the current attention vector
    coverage_loss = np.sum(np.minimum(coverage_vectors, attention) * mask) / np.sum(mask)

    return coverage_loss


class CoverageAttentionLossMetric:
    """
    Checks batch-equality of two token lists and computes an accuracy metric based on that.
    Also, checks if the same number of clauses of each type (SELECT, FROM, WHERE, END) as in target
    appears in the prediction.
    """
    def __init__(self) -> None:
        self._total_coverage_loss = 0.
        self._total_count = 0.

    def __call__(self,
                 debug_info,
                 mask=None):

        self._total_coverage_loss += calculate_coverage_loss(debug_info, mask)
        self._total_count += 1

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy,
        The "clause accuracy" - is a clause that appears in the gold tokens appear in the prediction:
        each of select, from, where, end token (";").
        """
        coverage_loss = 0 if self._total_count == 0. else self._total_coverage_loss / self._total_count
        if reset:
            self.reset()

        return {'coverage_loss': coverage_loss}

    def reset(self):
        self._total_count = 0.
        self._total_coverage_loss = 0.
