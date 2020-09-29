from typing import Dict, List, Tuple
import logging
import json
import glob
import os
import sqlite3
import dill
from pathlib import Path
import collections

from overrides import overrides

from allennlp.common import JsonDict
from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, Field, ProductionRuleField, ListField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer, PretrainedBertIndexer
from text2sql.semparse.worlds.grmr_attn_sup_world import AttnSupGrammarBasedWorld
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import read_dataset_schema
import text2sql.data.dataset_readers.dataset_utils.text2sql_utils as local_text2sql_utils

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("grammar_based_attn_sup")
class GrammarBasedAttnSupText2SqlDatasetReader(DatasetReader):
    """
    Reads text2sql data with the FastAlign token-to-token alignment
    """
    def __init__(self,
                 schema_path: str,
                 database_file: str = None,
                 use_all_sql: bool = False,
                 use_all_queries: bool = True,
                 remove_unneeded_aliases: bool = False,
                 use_prelinked_entities: bool = True,
                 use_untyped_entities: bool = True,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 cross_validation_split_to_exclude: int = None,
                 lazy: bool = False,
                 load_cache: bool = True,
                 save_cache: bool = True,
                 loading_limit: int = -1) -> None:
        """
        :param schema_path: str path to csv file that describes the database schema
        :param database_file: str optional, used to load values from the database as terminal rules. unused in case of anonymization
        :param use_all_sql: bool if false, for each english query only one example is read, using the first SQL program.
        :param use_all_queries: bool if false, read only one example out of all the examples with the same SQL
        :param remove_unneeded_aliases: bool not supported, default False
        :param use_prelinked_entities: bool not supported, default True
        :param use_untyped_entities: bool not supported, default True
        :param token_indexers: Dict[str, TokenIndexer], optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
        :param cross_validation_split_to_exclude: int, optional (default = None)
        Some of the text2sql datasets are very small, so you may need to do cross validation.
        Here, you can specify a integer corresponding to a split_{int}.json file not to include
        in the training set.
        :param load_cache: bool if true, loads dataset from cahce
        :param save_cache: bool if true, saves dataset to cache
        :param loading_limit: int if larger than -1, read only loading_limit examples (for debug)
        """
        super().__init__(lazy)

        self._load_cache = load_cache
        self._save_cache = save_cache
        self._loading_limit = loading_limit

        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._use_all_sql = use_all_sql
        self._remove_unneeded_aliases = remove_unneeded_aliases
        self._use_prelinked_entities = use_prelinked_entities
        self._use_all_queries = use_all_queries

        if not self._use_prelinked_entities:
            raise ConfigurationError("The grammar based text2sql dataset reader "
                                     "currently requires the use of entity pre-linking.")

        self._cross_validation_split_to_exclude = str(cross_validation_split_to_exclude)

        if database_file:
            try:
                database_file = cached_path(database_file)
                connection = sqlite3.connect(database_file)
                self._cursor = connection.cursor()
            except FileNotFoundError as e:
                self._cursor = None
        else:
            self._cursor = None

        self._schema_path = schema_path
        self._schema = read_dataset_schema(self._schema_path)
        self._world = AttnSupGrammarBasedWorld(schema_path,
                                               self._cursor,
                                               use_prelinked_entities=use_prelinked_entities,
                                               use_untyped_entities=use_untyped_entities)

    @overrides
    def _read(self, file_path: str):
        """
        This dataset reader consumes the data from
        https://github.com/jkkummerfeld/text2sql-data/tree/master/data
        formatted using ``scripts/reformat_text2sql_data.py``
        and updated with pre trained alignments from fast align as the additional fields

        Parameters
        ----------
        file_path : ``str``, required.
            For this dataset reader, file_path can either be a path to a file `or` a
            path to a directory containing json files. The reason for this is because
            some of the text2sql datasets require cross validation, which means they are split
            up into many small files, for which you only want to exclude one.
        """
        # For example, file scholar/schema_full_split/aligned_final_dev.json will be saved in
        # scholar/schema_full_split/attnsupgrammar_cache_aligned_final_dev
        file_path = Path(file_path)
        if 'elmo' in self._token_indexers.keys():
            cache_dir = os.path.join(file_path.parent, f'attnsupgrmr_elmo_cache_{file_path.stem}')
        elif 'bert' in self._token_indexers.keys():
            cache_dir = os.path.join(file_path.parent, f'attnsupgrmr_bert_cache_{file_path.stem}')
        else:
            cache_dir = os.path.join(file_path.parent, f'attnsupgrmr_cache_{file_path.stem}')
        if self._load_cache:
            logger.info(f'Trying to load cache from {cache_dir}')
            if not os.path.isdir(cache_dir):
                logger.info(f'Can\'t load cache, cache {cache_dir} doesn\'t exits')
                self._load_cache = False
        if self._save_cache:
            os.makedirs(cache_dir, exist_ok=True)

        files = [p for p in glob.glob(str(file_path))
                 if self._cross_validation_split_to_exclude not in os.path.basename(p)]
        cnt = 0 # used to limit the number of loaded instances
        for path in files:
            with open(cached_path(path), "r") as data_file:
                data = json.load(data_file)

            total_cnt = -1 # used to name the cache files
            for sql_data in local_text2sql_utils.process_sql_data_attn_sup_grmr(data,
                                                                                use_all_sql=self._use_all_sql,
                                                                                remove_unneeded_aliases=self._remove_unneeded_aliases,
                                                                                schema=self._schema,
                                                                                use_all_queries=self._use_all_queries):
                # Handle caching - only caching instances that are not None
                # (any non parsable sql query will result in None)
                total_cnt += 1
                cache_filename = f'instance-{total_cnt}.pt'
                cache_filepath = os.path.join(cache_dir, cache_filename)
                if self._loading_limit == cnt:
                    break
                if self._load_cache:
                    try:
                        instance = dill.load(open(cache_filepath, 'rb'))
                        cnt += 1
                        yield instance
                    except Exception as e:
                        # could not load from cache - keep loading without cache
                        pass
                else:
                    linked_entities = sql_data.sql_variables if self._use_prelinked_entities else None
                    instance = self.text_to_instance(query=sql_data.text_with_variables,
                                                     derived_cols=sql_data.derived_cols,
                                                     derived_tables=sql_data.derived_tables,
                                                     prelinked_entities=linked_entities,
                                                     sql=sql_data.sql,
                                                     alignment=sql_data.alignment_with_variables)
                    if instance is not None:
                        cnt += 1
                        if self._save_cache:
                            dill.dump(instance, open(cache_filepath, 'wb'))
                        yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         query: List[str],
                         derived_cols: List[Tuple[str, str]],
                         derived_tables: List[str],
                         prelinked_entities: Dict[str, Dict[str, str]] = None,
                         sql: List[str] = None,
                         alignment: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokens = TextField([Token(t) for t in query], self._token_indexers)
        fields["tokens"] = tokens

        if sql is not None:
            action_sequence, all_actions = self._world.get_action_sequence_and_all_actions(query=sql,
                                                                                           derived_cols=derived_cols,
                                                                                           derived_tables=derived_tables,
                                                                                           prelinked_entities=prelinked_entities)
            if action_sequence is None:
                return None

            if alignment is not None:
                # Modify the alignment according to the action sequence
                alignment = AttnSupGrammarBasedWorld.modify_alignment(action_sequence=action_sequence,
                                                                      alignment=alignment)
            else:
                # having a list of NO_ALIGN is basically equivalent to mask all the alignment
                alignment = ['NO_ALIGN'] * len(action_sequence)

        index_fields: List[Field] = []
        production_rule_fields: List[Field] = []

        for production_rule in all_actions:
            nonterminal, _ = production_rule.split(' ->')
            production_rule = ' '.join(production_rule.split(' '))
            field = ProductionRuleField(production_rule,
                                        self._world.is_global_rule(nonterminal),
                                        nonterminal=nonterminal)
            production_rule_fields.append(field)

        valid_actions_field = ListField(production_rule_fields)
        fields["valid_actions"] = valid_actions_field

        action_map = {action.rule: i # type: ignore
                      for i, action in enumerate(valid_actions_field.field_list)}

        for production_rule in action_sequence:
            index_fields.append(IndexField(action_map[production_rule], valid_actions_field))
        if not action_sequence:
            index_fields = [IndexField(-1, valid_actions_field)]
        # if not action_sequence and re.findall(r"COUNT \( \* \) (?:<|>|<>|=) 0", " ".join(sql)):
        #     index_fields = [IndexField(-2, valid_actions_field)]

        action_sequence_field = ListField(index_fields)
        fields["action_sequence"] = action_sequence_field

        alignment_index_fields: List[IndexField] = []
        tmp_tokens_as_strings = [t.text for t in tokens]
        for aligned_token in alignment:
            try:
                aligned_token_index = int(tmp_tokens_as_strings.index(aligned_token))
                alignment_index_fields.append(IndexField(aligned_token_index, tokens))
            except ValueError as e:
                # a special "no alignment" index
                alignment_index_fields.append(IndexField(-1, tokens.empty_field()))
        fields["alignment_sequence"] = ListField(alignment_index_fields)

        return Instance(fields)

    def read_json_dict(self, json_dict: JsonDict) -> Instance:
        """
        Expectied keys:
        question: string
        sql: string
        variables: a dictionary with the variables as keys, and original entities as values - {'author0': 'jane doe'}
        """
        text_vars_str = json_dict['variables'].replace('\'','"')
        text_vars = json.loads(text_vars_str)
        sql_vars = [{'name': k, 'example': v, 'type': k[:-1]} for k,v in text_vars.items()]
        data = [{'sentences': [{'text': json_dict['question'], 'question-split': 'question', 'variables': text_vars}],
                'sql': [json_dict['sql']],
                'variables': sql_vars}]

        for sql_data in local_text2sql_utils.process_sql_data(data,
                                                              use_all_sql=self._use_all_sql,
                                                              remove_unneeded_aliases=self._remove_unneeded_aliases,
                                                              schema=self._schema,
                                                              use_all_queries=self._use_all_queries):
            linked_entities = sql_data.sql_variables if self._use_prelinked_entities else None
            instance = self.text_to_instance(query=sql_data.text_with_variables,
                                             derived_cols=sql_data.derived_cols,
                                             derived_tables=sql_data.derived_tables,
                                             prelinked_entities=linked_entities,
                                             sql=sql_data.sql)
            return instance
