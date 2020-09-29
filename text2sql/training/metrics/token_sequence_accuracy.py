from typing import List, Dict
from overrides import overrides

from allennlp.training.metrics import Metric
from text2sql.data.dataset_readers.dataset_utils.text2sql_utils import retokenize_gold


@Metric.register("my_token_sequence_accuracy")
class TokenSequenceAccuracy(Metric):
    """
    Simple sequence accuracy based on tokens, as opposed to tensors.
    """

    def __init__(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.

    @overrides
    def reset(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.

    @overrides
    def __call__(self,
                 predictions: List[List[str]],
                 gold_targets: List[List[str]]) -> None:
        self._total_counts += len(predictions)
        for predicted_tokens, gold_tokens in zip(predictions, gold_targets):
            # TODO: find out how come preds are different then targets :|
            # if '\'' in predicted_tokens and '"' in gold_tokens:
            #     predicted_tokens = [tok if tok != '\'' else '"' for tok in predicted_tokens]
            # if re.findall(r" [A-Za-z0-9_]+\.[A-Za-z0-9_]+ ", " ".join(gold_tokens)) and \
            #    not re.findall(r" [A-Za-z0-9_]+\.[A-Za-z0-9_]+ ", " ".join(predicted_tokens)):
            #     gold_tokens = retokenize_gold(gold_tokens)
            if predicted_tokens == gold_tokens:
                self._correct_counts += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self._total_counts == 0:
            accuracy = 0.
        else:
            accuracy = self._correct_counts / self._total_counts

        if reset:
            self.reset()

        return {"seq_acc": accuracy}
