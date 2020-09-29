from text2sql.data.preprocess.sql_templates import sql_schema_sanitize
from text2sql.data.dataset_readers.dataset_utils.text2sql_utils import read_schema_dict


from typing import List, Dict
from overrides import overrides
import os

from allennlp.training.metrics import Metric

DATA_BASE_PATH = Path(os.getcwd()) / "data" / "sql data"
DATA_PATTERN = '*/*.json'


def get_glob_templ(gold_tokens, schema):
    """
    Create the string like in the data json file and convert to global template
    Assumptions:
        1. columns form:
            1) TAB_NAMEalias# . COL_NAME
        2. tables form:
            1) TAB_NAME AS TAB_NAMEalias#
    """
    sql = " ".join(gold_tokens)
    return sql_schema_sanitize(sql, schema)


@Metric.register("glob_templ_acc")
class GlobalTemplAccuracy(Metric):
    """
    Simple sequence accuracy based on tokens, as opposed to tensors.
    Anonymize the KB tokens (names of tables and columns, variables)
    """

    def __init__(self, schema_path):
        self._correct_counts = 0.0
        self._total_counts = 0.0
        self._schema = read_schema_dict(schema_path)

    @overrides
    def reset(self) -> None:
        self._correct_counts = 0.0
        self._total_counts = 0.0

    @overrides
    def __call__(self,
                 predictions: List[List[str]],
                 gold_targets: List[List[str]]) -> None:
        self._total_counts += len(predictions)
        assert gold_targets[0][0] == "SELECT"
        for predicted_tokens, gold_tokens in zip(predictions, gold_targets):
            _, _, pred_templ = get_glob_templ(predicted_tokens, self._schema)
            _, _, gold_templ = get_glob_templ(gold_tokens, self._schema)
            if pred_templ == gold_templ:
                self._correct_counts += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        try:
            accuracy = self._correct_counts / self._total_counts
        except:
            accuracy = 0.0

        if reset:
            self.reset()

        return {'schema_free_acc': accuracy}
