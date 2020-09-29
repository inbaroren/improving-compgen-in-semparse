from typing import Dict, List, Tuple
import logging
import json
import glob
import os
import sqlite3
import random

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SpanField, ListField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

import text2sql.data.dataset_readers.dataset_utils.text2sql_utils as text2sql_utils
from allennlp.data.dataset_readers.dataset_utils import text2sql_utils as tu
from text2sql.data.preprocess.sql_templates import sql_schema_sanitize
from text2sql.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer, StandardTokenizer
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("seq2seq_spans")
class Seq2SeqSpansDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    ``SimpleSeq2Seq`` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of ``read`` is a list of ``Instance`` s with the fields:
        source_tokens: ``TextField`` and
        target_tokens: ``TextField``

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    Parameters
    ----------
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 schema_path: str,
                 database_path: str = None,
                 use_all_sql: bool = False,
                 use_all_queries: bool = True,
                 remove_unneeded_aliases: bool = False,
                 use_prelinked_entities: bool = True,
                 cross_validation_split_to_exclude: int = None,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 lazy: bool = False,
                 random_seed:int = 0,
                 schema_free_supervision=False) -> None:
        super().__init__(lazy)
        self._random_seed = random_seed
        # becuase the spans were preproceessed, it is essential to enforce the same tokenization
        self._source_tokenizer = WhitespaceTokenizer()
        self._target_tokenizer = StandardTokenizer()
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token

        self._cross_validation_split_to_exclude = str(cross_validation_split_to_exclude)
        self._use_all_sql = use_all_sql
        self._use_all_queries = use_all_queries
        self._remove_unneeded_aliases = remove_unneeded_aliases
        self._use_prelinked_entities = use_prelinked_entities

        if database_path is not None:
            database_path = cached_path(database_path)
            connection = sqlite3.connect(database_path)
            self._cursor = connection.cursor()
        else:
            self._cursor = None

        self._schema_path = schema_path
        self._schema_free_supervision = schema_free_supervision

    @overrides
    def _read(self, file_path: str):
        """
        This dataset reader consumes the data from
        https://github.com/jkkummerfeld/text2sql-data/tree/master/data
        formatted using ``scripts/reformat_text2sql_data.py``.

        Parameters
        ----------
        file_path : ``str``, required.
            For this dataset reader, file_path can either be a path to a file `or` a
            path to a directory containing json files. The reason for this is because
            some of the text2sql datasets require cross validation, which means they are split
            up into many small files, for which you only want to exclude one.
        """
        files = [p for p in glob.glob(file_path)
                 if self._cross_validation_split_to_exclude not in os.path.basename(p)]

        for path in files:
            split_data = []
            with open(cached_path(path), "r") as data_file:
                logger.info("Reading instances from lines in file at: %s", path)
                data = json.load(data_file)

            for text, sql, spans in text2sql_utils.process_sql_data_standard(data,
                                                                      use_linked=self._use_prelinked_entities,
                                                                      use_all_sql=self._use_all_sql,
                                                                      use_all_queries=self._use_all_queries,
                                                                      output_spans=True):
                instance = self.text_to_instance(text, sql, spans)
                if instance is not None:
                    split_data.append(instance)
            # randomize and output
            # random.Random(self._random_seed).shuffle(split_data)
            for instance in split_data:
                yield instance

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None, spans: List[Tuple[int, int]] = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        spans_field: List[Field] = []
        spans = self._fix_spans_coverage(spans, len(tokenized_source))
        for start, end in spans:
            spans_field.append(SpanField(start, end, source_field))
        span_list_field: ListField = ListField(spans_field)

        if target_string is not None:
            if self._schema_free_supervision:
                _, _, target_string = sql_schema_sanitize(target_string, text2sql_utils.read_schema_dict(self._schema_path))
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            if self._remove_unneeded_aliases:
                new_target = tu.clean_unneeded_aliases([token.text for token in tokenized_target])
                tokenized_target = [Token(t) for t in new_target]
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)
            return Instance({"source_tokens": source_field, "spans": span_list_field, "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field, "spans": span_list_field})

    def _fix_spans_coverage(self, spans: List[Tuple[int, int]], source_length: int):
        """
        Given a list of spans, fixes them to be inclusive, shifts them to adapt the sequence with START_SYMBOL,
        and adds all the size 1 spans
        :param spans: spans over source_tokenized
        :param source_length: the length of source_tokenized
        :return: List[Tuple[int, int]], spans.union(all size 1 spans)
        """
        source_start_index = 0
        source_end_index = source_length-1
        # add +1 to the start indices since a START_SYMBOL was added
        # end indices are now inclusive
        if self._source_add_start_token:
            new_spans: List[Tuple[int, int]] = []
            for s, e in spans:
                new_spans.append((s + 1, e))
            source_start_index += 1
            source_end_index -= 1
        else:
            new_spans = spans
        spans_set = set(new_spans)
        for i in range(source_start_index, source_end_index+1):
            # inclusive spans
            spans_set.add((i, i))
        return spans_set


if __name__ == '__main__':
    # test redear
    c = Seq2SeqSpansDatasetReader('target',
                                  use_all_sql=False,
                                  use_all_queries=True,
                                  use_prelinked_entities=True)
    for dataset in ['advising']:
        for split_type in ['schema_free_split', 'new_question_split', 'schema_full_split']:
            for split in ['final_new_no_join_dev', 'final_new_no_join_test']:
                data = c.read(f'/datainbaro2/text2sql/parsers_models/allennlp_text2sql/data/sql data/{dataset}/{split_type}/{split}.json')

