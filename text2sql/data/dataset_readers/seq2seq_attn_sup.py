from typing import Dict, List
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
from allennlp.data.fields import TextField, IndexField, ListField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

import text2sql.data.dataset_readers.dataset_utils.text2sql_utils as text2sql_utils

from text2sql.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer, StandardTokenizer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("attn_sup_seq2seq")
class AttnSupSeq2SeqDatasetReader(DatasetReader):
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
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 lazy: bool = False,
                 random_seed:int = 0) -> None:
        super().__init__(lazy)
        self._random_seed = random_seed
        self._source_tokenizer = WhitespaceTokenizer()
        self._alignment_tokenizer = self._source_tokenizer
        self._target_tokenizer = StandardTokenizer()

        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._alignment_token_indexer = self._source_token_indexers
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

            # print("text2sql utils preprocess: ",
            #       len([x for x in tu.process_sql_data(data, use_all_sql=False, use_all_queries=False)]))
            for text, sql, alignment in text2sql_utils.process_sql_data_attn_sup(data,
                                                                      use_linked=self._use_prelinked_entities,
                                                                      use_all_sql=self._use_all_sql,
                                                                      use_all_queries=self._use_all_queries):
                instance = self.text_to_instance(text, sql, alignment)
                if instance is not None:
                    split_data.append(instance)
            # randomize and output
            # random.Random(self._random_seed).shuffle(split_data)
            for instance in split_data:
                yield instance

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None, alignment: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            if self._remove_unneeded_aliases:
                new_target = tu.clean_unneeded_aliases([token.text for token in tokenized_target])
                tokenized_target = [Token(t) for t in new_target]
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            alignment_index_fields: List[IndexField] = []
            tokenized_alignment = self._source_tokenizer.tokenize(alignment)
            tmp_source_tokenized_strings = [t.text for t in tokenized_source]
            for aligned_token in tokenized_alignment:
                try:
                    aligned_token_index = int(tmp_source_tokenized_strings.index(aligned_token.text))
                except ValueError as e:
                    # Since START_TOKEN is added, no step should be aligned to it so it can be used as
                    # a special "no alignment" index
                    aligned_token_index = 0
                alignment_index_fields.append(IndexField(aligned_token_index, source_field))
            if not alignment_index_fields:
                # if there was no alignemnet (it was None or ""), add dummy alignments
                for _ in range(len(tokenized_target)-2):
                    alignment_index_fields.append(IndexField(0, source_field))
            alignment_field = ListField(alignment_index_fields)

            return Instance({"source_tokens": source_field, "target_tokens": target_field, "alignment_sequence": alignment_field})
        else:
            return Instance({'source_tokens': source_field})


if __name__ == '__main__':
    for use_all_sql in [False]:
        for use_all_queries in [True]:
            c = AttnSupSeq2SeqDatasetReader('target',
                                            use_all_sql=use_all_sql,
                                            use_all_queries=use_all_queries,
                                            use_prelinked_entities=True)
            for dataset in ['scholar']:
                for split_type in ['schema_full_split']:
                    for split in ['aligned_train']:
                        data = c.read(f'../../../data/sql data/{dataset}/{split_type}/{split}.json')


