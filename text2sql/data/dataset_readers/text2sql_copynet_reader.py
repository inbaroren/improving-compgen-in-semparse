from typing import Dict, List
import logging
import json
import glob
import os
import sqlite3
import numpy as np
import random

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, NamespaceSwappingField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
# from allennlp.data.dataset_readers.dataset_utils import text2sql_utils
from allennlp.data.dataset_readers.semantic_parsing.template_text2sql import TemplateText2SqlDatasetReader

import text2sql.data.dataset_readers.dataset_utils.text2sql_utils as text2sql_utils
# from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import read_dataset_schema
from allennlp.common.util import START_SYMBOL, END_SYMBOL


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("text2sql_copynet_reader")
class CopyNetText2SqlDatasetReader(DatasetReader):
    def __init__(self,
                 target_namespace: str,
                 schema_path: str = None,
                 database_path: str = None,
                 use_all_sql: bool = True,
                 remove_unneeded_aliases: bool = True,
                 use_prelinked_entities: bool = True,
                 cross_validation_split_to_exclude: int = None,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 aug_data_path: str = None,
                 aug_ratio: float = 0.0,
                 random_seed: int = 0) -> None:
        super().__init__(lazy)
        self._use_all_sql = use_all_sql
        self._remove_unneeded_aliases = remove_unneeded_aliases
        self._use_prelinked_entities = use_prelinked_entities

        self._cross_validation_split_to_exclude = str(cross_validation_split_to_exclude)

        if database_path is not None:
            database_path = cached_path(database_path)
            connection = sqlite3.connect(database_path)
            self._cursor = connection.cursor()
        else:
            self._cursor = None

        self._schema_path = schema_path

        self._target_namespace = target_namespace
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        if "tokens" not in self._source_token_indexers or \
                not isinstance(self._source_token_indexers["tokens"], SingleIdTokenIndexer):
            raise ConfigurationError("CopyNetDatasetReader expects 'source_token_indexers' to contain "
                                     "a 'single_id' token indexer called 'tokens'.")
        self._target_token_indexers: Dict[str, TokenIndexer] = {
            "tokens": SingleIdTokenIndexer(namespace=self._target_namespace)
        }
        self._aug_data = []
        if aug_data_path:
            with open(cached_path(aug_data_path), "r") as data_file:
                aug_data = json.load(data_file)
            self._aug_data = [entry for entry in aug_data if entry['valid']]
        self._aug_ratio = aug_ratio
        self._random_seed = random_seed

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
            logger.info("Reading instances from lines in file at: %s", path)
            with open(cached_path(path), "r") as data_file:
                data = json.load(data_file)
            for text, sql in text2sql_utils.process_sql_data_standard(data, self._use_prelinked_entities):
                instance = self.text_to_instance(text, sql)
                if instance is not None:
                    split_data.append(instance)
            # if needs to augment data
            if self._aug_ratio > 0:
                random.Random(self._random_seed).shuffle(self._aug_data)
                aug_num = int(self._aug_ratio * len(self._aug_data)) + 1
                aug_data = [self.text_to_instance(" ".join(entry["inp"]), " ".join(entry["out"]))
                                  for entry in self._aug_data[:aug_num]]
                split_data.extend(aug_data)
            # randomize and output
            random.Random(self._random_seed).shuffle(split_data)
            for instance in split_data:
                yield instance

    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)))
        return out

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
        """
        Turn raw source string and target string into an ``Instance``.

        Parameters
        ----------
        source_string : ``str``, required
        target_string : ``str``, optional (default = None)

        Returns
        -------
        Instance
            See the above for a description of the fields that the instance will contain.
        """
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(tokenized_source[1:-1], self._target_namespace)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source[1:-1]]}
        fields_dict = {
                "source_tokens": source_field,
                "source_to_target": source_to_target_field,
        }

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
            source_and_target_token_ids = self._tokens_to_ids(tokenized_source[1:-1] +
                                                              tokenized_target)
            source_token_ids = source_and_target_token_ids[:len(tokenized_source)-2]
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))
            target_token_ids = source_and_target_token_ids[len(tokenized_source)-2:]
            fields_dict["target_token_ids"] = ArrayField(np.array(target_token_ids))
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source[1:-1])
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))

        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)
