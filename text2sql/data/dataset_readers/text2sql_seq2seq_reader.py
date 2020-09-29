from typing import Dict
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
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

import text2sql.data.dataset_readers.dataset_utils.text2sql_utils as text2sql_utils
from text2sql.data.preprocess.sql_templates import sql_schema_sanitize
from text2sql.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("text2sql_seq2seq_reader")
class Seq2SeqDatasetReader(DatasetReader):
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
        self._source_tokenizer = source_tokenizer or WhitespaceTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
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
            for text, sql in text2sql_utils.process_sql_data_standard(data,
                                                                      use_linked=self._use_prelinked_entities,
                                                                      use_all_sql=self._use_all_sql,
                                                                      use_all_queries=self._use_all_queries):
                instance = self.text_to_instance(text, sql)
                if instance is not None:
                    split_data.append(instance)
            # randomize and output
            # random.Random(self._random_seed).shuffle(split_data)
            for instance in split_data:
                yield instance

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)
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
            return Instance({"source_tokens": source_field, "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field})
