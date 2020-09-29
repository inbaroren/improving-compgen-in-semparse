from typing import List
from overrides import overrides
import re
import os
from typing import Dict
from allennlp.training.metrics import Metric
from text2sql.data.dataset_readers.dataset_utils.text2sql_utils import read_schema_dict

DATA_BASE_PATH = Path(os.getcwd()) / "data" / "sql data"


def get_unaliased_consts(toks, schema):
    """
    Assumptions:
        1. columns form:
            1) TAB_NAME . COL_NAME
        2. tables form:
            1) TAB_NAME
    """
    kb_consts = set()
    prev_tok = toks[0]
    next_tok = toks[2]
    for tok in toks:
        if tok in schema:
            kb_consts.add(tok)
        elif prev_tok in schema and next_tok in schema[prev_tok] and tok == ".":
            kb_consts.add(f"{prev_tok} . {next_tok}")
    return kb_consts


def get_consts(toks, schema, supervision='gold'):
    """
    Assumptions:
        1. columns form:
            1) TAB_NAMEalias# . COL_NAME
            2) TAB_NAMEalias# . DERIVED_FIELDalias#
        2. tables form:
            1) TAB_NAME AS TAB_NAMEalias#
            2) subquery AS DERIVED_TALBEalias#

    Parameters:
    -----------
    toks: tokens of tokenized sql query, List[str]
    schema: dictionary of tables as keys and columns as values, Dict[str, List[str]]
    supervision: is the query input a prediction of a model, in ['gold', 'pred']
    """
    sql = " ".join(toks)
    kb_consts = set()

    # first add schema native entities
    for tab, cols in schema.items():
        for col in cols:
            pattern = " (%s[\s]{0,1}alias[0-9])[\s]{0,1}\.[\s]{0,1}%s " % (tab, col)
            for m in re.finditer(pattern, sql):
                normalized_col = re.sub(r"([^\s])\.([^\s])", r"\g<1> . \g<2>", m.group(0)[1:-1])
                normalized_col = re.sub(r"([^\s]+)[\s]alias([0-9])", r"\g<1>alias\g<2>", normalized_col)
                kb_consts.add(normalized_col)  # make sure it is always the same!

                normalized_tab = re.sub(r"([^\s]+)[\s]alias([0-9])", r"\g<1>alias\g<2>", m.group(1))
                kb_consts.add(normalized_tab)
        # find other tables
        pattern = " %s AS (%s[\s]{0,1}alias[0-9]) " % (tab, tab)
        for m in re.finditer(pattern, sql):
            normalized_tab = re.sub(r"([^\s]+)[\s]alias([0-9])", r"\g<1>alias\g<2>", m.group(1))
            kb_consts.add(normalized_tab)

    # now add all kind of derived fields and tables
    for m in re.finditer(r" ([A-Z_]+[\s]{0,1}alias[0-9])[\s]{0,1}\.[\s]{0,1}DERIVED_FIELD[\s]{0,1}alias[0-9] ", sql):
        normalized_col = re.sub(r"([^\s])\.([^\s])", r"\g<1> . \g<2>", m.group(0)[1:-1])
        normalized_col = re.sub(r"([^\s]+)[\s]alias([0-9])", r"\g<1>alias\g<2>", normalized_col)
        kb_consts.add(normalized_col)  # make sure it is always the same!

        normalized_tab = re.sub(r"([^\s]+)[\s]alias([0-9])", r"\g<1>alias\g<2>", m.group(1))
        kb_consts.add(normalized_tab)
    for m in re.finditer(r"\) AS (DERIVED_TABLE[\s]{0,1}alias[0-9]) ", sql):
        normalized_tab = re.sub(r"([^\s]+)[\s]alias([0-9])", r"\g<1>alias\g<2>", m.group(1))
        kb_consts.add(normalized_tab)

    return kb_consts


@Metric.register("kb_acc")
class KnowledgeBaseConstsAccuracy(Metric):
    """
    Simple sequence accuracy based on tokens, as opposed to tensors.
    """

    def __init__(self, dataset='advising', aliased=True, schema_path=None) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.
        if not schema_path:
            self._schema = read_schema_dict(os.path.join(DATA_BASE_PATH, f"{dataset}-schema.csv"))
        else:
            self._schema = read_schema_dict(schema_path)
        self._aliased = aliased

    @overrides
    def reset(self) -> None:
        self._correct_counts = 0.  # corrects by global template
        self._total_counts = 0.

    @overrides
    def __call__(self,
                 predictions: List[List[str]],
                 gold_targets: List[List[str]]) -> None:
        assert gold_targets[0][0] == "SELECT"
        self._total_counts += len(predictions)
        for predicted_tokens, gold_tokens in zip(predictions, gold_targets):
            if self._aliased:
                predicted_kb_consts = get_consts(predicted_tokens, self._schema, 'pred')
                gold_kb_consts = get_consts(gold_tokens, self._schema, 'gold')
            else:
                predicted_kb_consts = get_unaliased_consts(predicted_tokens, self._schema)
                gold_kb_consts = get_unaliased_consts(gold_tokens, self._schema)
            if predicted_kb_consts == gold_kb_consts:
                self._correct_counts += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        kb_acc = 0. if self._total_counts == 0 else self._correct_counts / self._total_counts

        if reset:
            self.reset()

        return {'kb_acc': kb_acc}
