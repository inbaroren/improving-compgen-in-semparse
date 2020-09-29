"""
Utility functions for reading the standardised text2sql datasets presented in
`"Improving Text to SQL Evaluation Methodology" <https://arxiv.org/abs/1806.09029>`_
"""

import numpy as np
from typing import List, Dict, NamedTuple, Iterable, Tuple, Set
from collections import defaultdict

from allennlp.common import JsonDict
import json
import collections
import os
import copy
import re
from text2sql.data.preprocess.text2sql_canonicalizer import process_sentence as preprocess_text

COUNTER = 0


class SqlData(NamedTuple):
    """
    A utility class for reading in text2sql data.

    Parameters
    ----------
    text : ``List[str]``
        The tokens in the text of the query.
    text_with_variables : ``List[str]``
        The tokens in the text of the query with variables
        mapped to table names/abstract variables.
    variable_tags : ``List[str]``
        Labels for each word in ``text`` which correspond to
        which variable in the sql the token is linked to. "O"
        is used to denote no tag.
    sql : ``List[str]``
        The tokens in the SQL query which corresponds to the text.
    text_variables : ``Dict[str, str]``
        A dictionary of variables associated with the text, e.g. {"city_name0": "san fransisco"}
    sql_variables : ``Dict[str, Dict[str, str]]``
        A dictionary of variables and column references associated with the sql query.
    """
    text: List[str]
    text_with_variables: List[str]
    variable_tags: List[str]
    sql: List[str]
    text_variables: Dict[str, str]
    sql_variables: Dict[str, Dict[str, str]]
    derived_tables: List[str]
    derived_cols: List[Tuple[str, str]]
    alignment_with_variables: List[str] = None
    spans: List[Tuple[int, int]] = None


class TableColumn(NamedTuple):
    name: str
    column_type: str
    is_primary_key: bool


def column_has_string_type(column: TableColumn) -> bool:
    if "varchar" in column.column_type:
        return True
    elif column.column_type == "text":
        return True
    elif column.column_type == "longtext":
        return True

    return False


def column_has_numeric_type(column: TableColumn) -> bool:
    if "int" in column.column_type:
        return True
    elif "float" in column.column_type:
        return True
    elif "double" in column.column_type:
        return True
    return False


def replace_variables(sentence: List[str],
                      sentence_variables: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """
    Replaces abstract variables in text with their concrete counterparts.
    """
    tokens = []
    tags = []
    for token in sentence:
        if token not in sentence_variables:
            tokens.append(token)
            tags.append("O")
        else:
            for word in sentence_variables[token].split():
                tokens.append(word)
                tags.append(token)
    return tokens, tags


def split_table_and_column_names(table: str) -> Iterable[str]:
    partitioned = [x for x in table.partition(".") if x != '']
    # Avoid splitting decimal strings.
    if partitioned[0].isnumeric() and partitioned[-1].isnumeric():
        return [table]
    return partitioned


def clean_and_split_sql(sql: str) -> List[str]:
    """
    Cleans up and unifies a SQL query. This involves unifying quoted strings
    and splitting brackets which aren't formatted consistently in the data.
    """
    sql_tokens: List[str] = []
    for token in sql.strip().split():
        token = token.replace('"', "'").replace("%", "")
        if token.endswith("(") and len(token) > 1:
            sql_tokens.extend(split_table_and_column_names(token[:-1]))
            sql_tokens.extend(split_table_and_column_names(token[-1]))
        else:
            sql_tokens.extend(split_table_and_column_names(token))
    return sql_tokens


def fix_specific_examples(sql):
    wrong_sql_queries = {}
    wrong_sql_queries['SELECT COUNT( RIVERalias0.RIVER_NAME ) FROM RIVER AS RIVERalias0 WHERE RIVERalias0.LENGTH > ALL ( SELECT RIVERalias1.LENGTH FROM RIVER AS RIVERalias1 WHERE RIVERalias1.RIVER_NAME = \"river_name0\" ) AND RIVERalias0.TRAVERSE = \"state_name0\" ;'] = 'SELECT COUNT( RIVERalias0.RIVER_NAME ) FROM RIVER AS RIVERalias0 WHERE RIVERalias0.LENGTH > ( SELECT MAX( RIVERalias1.LENGTH ) FROM RIVER AS RIVERalias1 WHERE RIVERalias1.RIVER_NAME = \"river_name0\" ) AND RIVERalias0.TRAVERSE = \"state_name0\" ;'
    wrong_sql_queries['SELECT STATEalias0.STATE_NAME FROM STATE AS STATEalias0 LEFT OUTER JOIN BORDER_INFO AS BORDER_INFOalias0 ON STATEalias0.STATE_NAME = BORDER_INFOalias0.STATE_NAME WHERE STATEalias0.STATE_NAME <> \"state_name0\" AND STATEalias0.STATE_NAME <> \"state_name1\" GROUP BY STATEalias0.STATE_NAME HAVING COUNT( BORDER_INFOalias0.BORDER ) = ( SELECT MIN( DERIVED_TABLEalias0.DERIVED_FIELDalias0 ) FROM ( SELECT COUNT( BORDER_INFOalias1.BORDER ) AS DERIVED_FIELDalias0 , STATEalias1.STATE_NAME FROM STATE AS STATEalias1 LEFT OUTER JOIN BORDER_INFO AS BORDER_INFOalias1 ON STATEalias1.STATE_NAME = BORDER_INFOalias1.STATE_NAME WHERE STATEalias1.STATE_NAME <> \"state_name0\" AND STATEalias1.STATE_NAME <> \"state_name1\" GROUP BY STATEalias1.STATE_NAME ) AS DERIVED_TABLEalias0 ) ;'] = 'SELECT STATEalias0.STATE_NAME FROM STATE AS STATEalias0 , BORDER_INFO AS BORDER_INFOalias0 WHERE STATEalias0.STATE_NAME = BORDER_INFOalias0.STATE_NAME AND STATEalias0.STATE_NAME <> "state_name0" AND STATEalias0.STATE_NAME <> "state_name1" GROUP BY STATEalias0.STATE_NAME HAVING COUNT( BORDER_INFOalias0.BORDER ) = ( SELECT MIN( DERIVED_TABLEalias0.DERIVED_FIELDalias0 ) FROM ( SELECT COUNT( BORDER_INFOalias1.BORDER ) AS DERIVED_FIELDalias0 , STATEalias1.STATE_NAME FROM STATE AS STATEalias1 , BORDER_INFO AS BORDER_INFOalias1 WHERE STATEalias1.STATE_NAME = BORDER_INFOalias1.STATE_NAME AND STATEalias1.STATE_NAME <> "state_name0" AND STATEalias1.STATE_NAME <> "state_name1" GROUP BY STATEalias1.STATE_NAME ) AS DERIVED_TABLEalias0 ) ;'
    wrong_sql_queries['SELECT STATEalias0.STATE_NAME FROM STATE AS STATEalias0 LEFT OUTER JOIN BORDER_INFO AS BORDER_INFOalias0 ON STATEalias0.STATE_NAME = BORDER_INFOalias0.STATE_NAME GROUP BY STATEalias0.STATE_NAME HAVING COUNT( BORDER_INFOalias0.BORDER ) = ( SELECT MIN( DERIVED_TABLEalias0.DERIVED_FIELDalias0 ) FROM ( SELECT COUNT( BORDER_INFOalias1.BORDER ) AS DERIVED_FIELDalias0 , STATEalias1.STATE_NAME FROM STATE AS STATEalias1 LEFT OUTER JOIN BORDER_INFO AS BORDER_INFOalias1 ON STATEalias1.STATE_NAME = BORDER_INFOalias1.STATE_NAME GROUP BY STATEalias1.STATE_NAME ) AS DERIVED_TABLEalias0 ) ;'] = 'SELECT STATEalias0.STATE_NAME FROM STATE AS STATEalias0 , BORDER_INFO AS BORDER_INFOalias0 WHERE STATEalias0.STATE_NAME = BORDER_INFOalias0.STATE_NAME GROUP BY STATEalias0.STATE_NAME HAVING COUNT( BORDER_INFOalias0.BORDER ) = ( SELECT MIN( DERIVED_TABLEalias0.DERIVED_FIELDalias0 ) FROM ( SELECT COUNT( BORDER_INFOalias1.BORDER ) AS DERIVED_FIELDalias0 , STATEalias1.STATE_NAME FROM STATE AS STATEalias1 , BORDER_INFO AS BORDER_INFOalias1 WHERE STATEalias1.STATE_NAME = BORDER_INFOalias1.STATE_NAME GROUP BY STATEalias1.STATE_NAME ) AS DERIVED_TABLEalias0 ) ;'
    wrong_sql_queries['SELECT DISTINCT RIVERalias0.LENGTH FROM RIVER AS RIVERalias0 WHERE RIVERalias0.RIVER_NAME = ( SELECT RIVER_NAME FROM ( SELECT COUNT( 1 ) AS DERIVED_FIELDalias0 , RIVERalias1.RIVER_NAME FROM RIVER AS RIVERalias1 GROUP BY RIVERalias1.RIVER_NAME ) AS DERIVED_TABLEalias0 WHERE DERIVED_TABLEalias0.DERIVED_FIELDalias0 = ( SELECT MAX( DERIVED_TABLEalias1.DERIVED_FIELDalias1 ) FROM ( SELECT COUNT( 1 ) AS DERIVED_FIELDalias1 , RIVERalias2.RIVER_NAME FROM RIVER AS RIVERalias2 GROUP BY RIVERalias2.RIVER_NAME ) AS DERIVED_TABLEalias1 ) ) ;'] = 'SELECT DISTINCT RIVERalias0.LENGTH FROM RIVER AS RIVERalias0 WHERE RIVERalias0.RIVER_NAME = ( SELECT DERIVED_TABLEalias0.RIVER_NAME FROM ( SELECT COUNT( 1 ) AS DERIVED_FIELDalias0 , RIVERalias1.RIVER_NAME FROM RIVER AS RIVERalias1 GROUP BY RIVERalias1.RIVER_NAME ) AS DERIVED_TABLEalias0 WHERE DERIVED_TABLEalias0.DERIVED_FIELDalias0 = ( SELECT MAX( DERIVED_TABLEalias1.DERIVED_FIELDalias1 ) FROM ( SELECT COUNT( 1 ) AS DERIVED_FIELDalias1 , RIVERalias2.RIVER_NAME FROM RIVER AS RIVERalias2 GROUP BY RIVERalias2.RIVER_NAME ) AS DERIVED_TABLEalias1 ) ) ;'

    sql = wrong_sql_queries.get(sql, sql)
    return sql


def clean_and_split_sql_v2(sql: str) -> List[str]:
    """
    Cleans up and unifies a SQL query. This involves unifying quoted strings
    and splitting brackets which aren't formatted consistently in the data.
    """
    sql_tokens: List[str] = []
    # fixes that were seen in training data
    if re.findall(r"SELECT DISTINCT\s*\(\s*[A-Z_]+alias[0-9]\.[A-Z_]+\s*\)", sql):
        sql = re.sub(r"SELECT DISTINCT\s*\(\s*([A-Z_]+alias[0-9]\.[A-Z_]+)\s*\)", r"SELECT DISTINCT \g<1>", sql)
    sql = fix_specific_examples(sql)
    # tokenize
    for token in sql.strip().split():
        token = token.replace('"', "'").replace("%", "").replace('(', ' ( ').replace(",", " , ")
        sql_tokens.extend(token.strip().split())

    return sql_tokens


def disambiguate_col_names(sql_tokens: List[str]) -> \
        Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Replaces all aliased tables with "TABLE_PLACEHOLDER AS TABLEalias#"
    Returns all the aliases for sub-queries and fields
    :param sql_tokens: tokens of the sql query, where all columns appear as "table_name . col_name"
    :return: sql_tokens
    """
    sql = ' '.join(sql_tokens)
    sql = re.sub(r" [A-Z_]+ AS ([A-Z_]+alias[0-9]) ", " TABLE_PLACEHOLDER AS \g<1> ", sql)
    new_tokens = sql.split()

    derived_tabs = re.findall(r" DERIVED_TABLEalias[0-9] ", sql)
    derived_tabs = list(set([der_tab.strip() for der_tab in derived_tabs]))

    derived_columns = re.findall(r" [A-Z_]+alias[0-9]\.DERIVED_FIELDalias[0-9]", sql) # derived_field
    derived_columns.extend(re.findall(r" DERIVED_TABLEalias[0-9]\.[A-Z_]+ ", sql)) # all columns of derived table
    derived_cols = list(set([(col.strip().split('.')[0], col.strip().split('.')[1]) for col in derived_columns]))

    return new_tokens, derived_tabs, derived_cols


def replace_variables_sql(sql: List[str],
                          text_variables: Dict[str, str],
                          sql_variables: Dict[str, Dict[str, str]]) -> List[str]:
    # assert text_variables.keys() == sql_variables.keys(), "Variables in question and SQL query are differnet!"
    tokens = []
    tags = []
    for token in sql:
        if token[1:-1] in text_variables and token[0] == token[-1] == "\'":
            # string value
            word = text_variables[token[1:-1]]
            tokens.append('\''+word+'\'')
            tags.append(token)
        elif token in text_variables:
            # number value
            word = text_variables[token]
            tokens.append(word)
            tags.append(token)
        else:
            tokens.append(token)
            tags.append("O")
    return tokens


def resolve_primary_keys_in_schema(sql_tokens: List[str],
                                   schema: Dict[str, List[TableColumn]]) -> List[str]:
    """
    Some examples in the text2sql datasets use ID as a column reference to the
    column of a table which has a primary key. This causes problems if you are trying
    to constrain a grammar to only produce the column names directly, because you don't
    know what ID refers to. So instead of dealing with that, we just replace it.
    """
    primary_keys_for_tables = {name: max(columns, key=lambda x: x.is_primary_key).name
                               for name, columns in schema.items()}
    resolved_tokens = []
    for i, token in enumerate(sql_tokens):
        if i > 2:
            table_name = sql_tokens[i - 2]
            if token == "ID" and table_name in primary_keys_for_tables.keys():
                token = primary_keys_for_tables[table_name]
        resolved_tokens.append(token)
    return resolved_tokens


def resolve_primary_keys_in_schema_aliased(sql_tokens: List[str],
                                           schema: Dict[str, List[TableColumn]]) -> List[str]:
    """
    Some examples in the text2sql datasets use ID as a column reference to the
    column of a table which has a primary key. This causes problems if you are trying
    to constrain a grammar to only produce the column names directly, because you don't
    know what ID refers to. So instead of dealing with that, we just replace it.
    """
    primary_keys_for_tables = {name: max(columns, key=lambda x: x.is_primary_key).name
                               for name, columns in schema.items()}
    resolved_tokens = []
    for i, token in enumerate(sql_tokens):
        if i > 2 and len(sql_tokens[i - 2]) > 6:
            # all tokens that are table names has "alias#" suffix that has to be removed
            table_name = sql_tokens[i - 2][:-6]
            if token == "ID" and table_name in primary_keys_for_tables.keys():
                token = primary_keys_for_tables[table_name]
        resolved_tokens.append(token)
    return resolved_tokens


def clean_unneeded_aliases(sql_tokens: List[str]) -> List[str]:
    unneeded_aliases = {}
    previous_token = sql_tokens[0]
    for (token, next_token) in zip(sql_tokens[1:-1], sql_tokens[2:]):
        if token == "AS" and previous_token is not None:
            # Check to see if the table name without the alias
            # is the same.
            table_name = next_token[:-6]
            if table_name == previous_token:
                # If so, store the mapping as a replacement.
                unneeded_aliases[next_token] = previous_token

        previous_token = token

    dealiased_tokens: List[str] = []
    for token in sql_tokens:
        new_token = unneeded_aliases.get(token, None)

        if new_token is not None and dealiased_tokens[-1] == "AS":
            dealiased_tokens.pop()
            continue
        elif new_token is None:
            new_token = token

        dealiased_tokens.append(new_token)

    return dealiased_tokens


class SqlScope:
    def __init__(self, level):
        self._level = level
        self._descendants = []
        self._unneeded_aliases = {}
        self._previous_token = None

    def update_aliases(self, key, value):
        self._unneeded_aliases[key] = value

    def get_alias_value(self, key):
        return self._unneeded_aliases.get(key, None)

    def update_decendants(self, scope):
        self._descendants.append(scope)


def clean_first_aliases(sql_tokens: List[str]) -> List[str]:
    """
    An alias is needed only for derived tables, and for self-join
    Aliases applied on for the section their defined in (for example, an alias to the table FLIGHTS in a
    subquery isn't applicable to other mentions of the table in outer scopes)
    """
    hard_example = "SELECT DISTINCT RIVERalias0.RIVER_NAME FROM RIVER AS RIVERalias0 WHERE RIVERalias0.LENGTH IN ( SELECT MAX( DERIVED_TABLEalias1.DERIVED_FIELDalias0 ) FROM ( SELECT MAX( RIVERalias1.LENGTH ) AS DERIVED_FIELDalias0 , RIVERalias1.TRAVERSE FROM RIVER AS RIVERalias1 WHERE RIVERalias1.TRAVERSE IN ( SELECT BORDER_INFOalias0.STATE_NAME FROM BORDER_INFO AS BORDER_INFOalias0 WHERE BORDER_INFOalias0.BORDER IN ( SELECT BORDER_INFOalias1.BORDER FROM BORDER_INFO AS BORDER_INFOalias1 GROUP BY BORDER_INFOalias1.BORDER HAVING COUNT( 1 ) = ( SELECT MAX( DERIVED_TABLEalias0.DERIVED_FIELDalias1 ) FROM ( SELECT BORDER_INFOalias2.BORDER , COUNT( 1 ) AS DERIVED_FIELDalias1 FROM BORDER_INFO AS BORDER_INFOalias2 GROUP BY BORDER_INFOalias2.BORDER ) AS DERIVED_TABLEalias0 ) ) ) GROUP BY RIVERalias1.TRAVERSE ) AS DERIVED_TABLEalias1 ) ;"
    scopes = {}
    unneeded_aliases = {}
    previous_token = sql_tokens[0]
    new_scope_tokens = ['(', 'SELECT']

    for (token, next_token) in zip(sql_tokens[1:-1], sql_tokens[2:]):
        if token == "AS" and previous_token is not None:
            # Check to see if the table name without the alias
            # is the same.
            table_name = next_token[:-6]
            if table_name == previous_token and next_token[-1] == "0":
                # If table first appears in the query, alias is redundant...
                # If so, store the mapping as a replacement.
                unneeded_aliases[next_token] = previous_token

        previous_token = token

    dealiased_tokens: List[str] = []
    for token in sql_tokens:
        new_token = unneeded_aliases.get(token, None)

        if new_token is not None and dealiased_tokens[-1] == "AS":
            dealiased_tokens.pop()
            continue
        elif new_token is None:
            new_token = token

        dealiased_tokens.append(new_token)

    return dealiased_tokens


def read_dataset_schema(schema_path: str) -> Dict[str, List[TableColumn]]:
    """
    Reads a schema from the text2sql data, returning a dictionary
    mapping table names to their columns and respective types.
    This handles columns in an arbitrary order and also allows
    either ``{Table, Field}`` or ``{Table, Field} Name`` as headers,
    because both appear in the data. It also uppercases table and
    column names if they are not already uppercase.

    Parameters
    ----------
    schema_path : ``str``, required.
        The path to the csv schema.

    Returns
    -------
    A dictionary mapping table names to typed columns.
    """
    schema: Dict[str, List[TableColumn]] = defaultdict(list)
    for i, line in enumerate(open(schema_path, "r")):
        if i == 0:
            header = [x.strip() for x in line.split(",")]
        elif line[0] == "-":
            continue
        else:
            data = {key: value for key, value in zip(header, [x.strip() for x in line.split(",")])}

            table = data.get("Table Name", None) or data.get("Table")
            column = data.get("Field Name", None) or data.get("Field")
            is_primary_key = data.get("Primary Key") == "y"
            schema[table.upper()].append(TableColumn(column.upper(), data["Type"], is_primary_key))

    return {**schema}


def read_schema_dict(schema_path):
    """
    Reads the schema csv file for a dataset and returns a dictionary
    Dict[str,List[str]] - for each table name (upper case), its columns names (upper case)
    """
    schema = defaultdict(list)
    for i, line in enumerate(open(schema_path, "r")):
        if i == 0:
            header = [x.strip() for x in line.split(",")]
        elif line[0] == "-":
            continue
        else:
            data = {key: value for key, value in zip(header, [x.strip() for x in line.split(",")])}

            table = data.get("Table Name", None) or data.get("Table")
            column = data.get("Field Name", None) or data.get("Field")
            schema[table.upper()].append(column.upper())
    return schema


def process_sql_data_standard(data, use_linked, use_all_sql, use_all_queries, output_spans=False) -> Tuple[str, str]:
    """
    Reads pairs of (sentence, sql) from data. gives different results than "process_sql_data" since the
    "seen_sentences" set is initialized once for all the sentences in data (and not for every entry...)
    """
    text2sql_pairs = set() # set of tuples (question, sql query)
    linked_text2sql = set() # set of tuples (question with variables, sql with variables)
    for entry in data:
        all_sql = [entry["sql"][0]] if not use_all_sql else entry["sql"]
        for sql in all_sql:
            seen_texts = set()  # set of all seen questions
            seen_linked_texts = set()  # set of all seen texts with variables (=linked entities)
            for utt in entry["sentences"]:
                if utt['question-split'] == 'exclude':
                    continue
                built_sql = sql
                text = utt["text"]
                built_text = text
                for k, v in utt["variables"].items():
                    built_sql = built_sql.replace(k, v)
                    built_text = built_text.replace(k, v)
                built_text = preprocess_text(built_text)
                text = preprocess_text(text)
                # fix `` to " back
                built_text = built_text.replace('``', '\"')
                text = text.replace('``', '\"')
                # convert from List[Tuple[int, int]] to string
                spans = ' '.join([f"{s[0]}-{s[1]}" for s in utt['constituency_parser_spans']])
                if use_all_queries:
                    text2sql_pairs.add((built_text, built_sql))
                    linked_text2sql.add((text, sql, spans))
                else:
                    if built_text not in seen_texts:  # don't add two texts with same sql
                        seen_texts.add(built_text)
                        text2sql_pairs.add((built_text, built_sql))
                    if text not in seen_linked_texts:  # don't add two texts with different sql
                        seen_linked_texts.add(text)
                        linked_text2sql.add((text, sql))

    if not use_linked:
        for pair in text2sql_pairs:
            yield pair
    else:
        for pair in linked_text2sql:
            if output_spans:
                # convert back to List[Tuple[int, int]]
                spans = [(int(span.split('-')[0]), int(span.split('-')[1])) for span in pair[2].split()]
                yield pair[0], pair[1], spans
            else:
                yield pair[0], pair[1]


def process_sql_data_attn_sup(data, use_linked, use_all_sql, use_all_queries) -> Tuple[str, str]:
    """
    Reads pairs of (sentence, sql) from data. gives different results than "process_sql_data" since the
    "seen_sentences" set is initialized once for all the sentences in data (and not for every entry...)
    """
    text2sql_pairs = set() # set of tuples (question, sql query)
    linked_text2sql = set() # set of tuples (question with variables, sql with variables)
    for entry in data:
        all_sql = [entry["sql"][0]] if not use_all_sql else entry["sql"]
        for sql in all_sql:
            seen_texts = set()  # set of all seen questions
            seen_linked_texts = set()  # set of all seen texts with variables (=linked entities)
            for utt in entry["sentences"]:
                if utt['question-split'] == 'exclude':
                    continue
                built_sql = sql
                text = utt["text"]
                alignment = utt.get("alignment", "")
                built_text = text
                built_alignment = alignment
                for k, v in utt["variables"].items():
                    built_sql = built_sql.replace(k, v)
                    built_text = built_text.replace(k, v)
                    built_alignment = built_alignment.replace(k, v)
                built_text = preprocess_text(built_text)
                text = preprocess_text(text)
                # fix `` to " back
                built_text = built_text.replace('``', '\"')
                text = text.replace('``', '\"')
                if use_all_queries:
                    text2sql_pairs.add((built_text, built_sql, built_alignment))
                    linked_text2sql.add((text, sql, alignment))
                else:
                    if built_text not in seen_texts:  # don't add two texts with same sql
                        seen_texts.add(built_text)
                        text2sql_pairs.add((built_text, built_sql))
                    if text not in seen_linked_texts:  # don't add two texts with different sql
                        seen_linked_texts.add(text)
                        linked_text2sql.add((text, sql))

    if not use_linked:
        for pair in text2sql_pairs:
            yield pair
    else:
        for pair in linked_text2sql:
            yield pair


def split_data(base_path, data_file_name):
    splits = ['train', 'dev', 'test']
    split_data = {'query': collections.defaultdict(list),
                  'question': collections.defaultdict(list)}
    with open(os.path.join(base_path, data_file_name)) as f:
        data = json.load(f)
    for entry in data:
        split_data['query'][entry['query-split']].append(entry)
        split_utts = {'train': [], 'test': [], 'dev': []}
        for utt in entry["sentences"]:
            split_utts[utt['question-split']].append(utt)
        for split_name in splits:
            new_entry = copy.deepcopy(entry)
            new_entry['sentences'] = copy.deepcopy(split_utts[split_name])
            split_data['question'][split_name].append(new_entry)
    assert len(splits) == len([k for k in split_data['question']]), "Not all splits found for question split"
    assert len(splits) == len([k for k in split_data['query']]), "Not all splits found for query split"

    for split_type in ['query', 'question']:
        for split_name in splits:
            new_data_path = os.path.join(base_path, f"{split_type}_split")
            os.makedirs(new_data_path, exist_ok=True)
            with open(os.path.join(new_data_path, f'{split_name}.json'), 'w+') as f:
                 json.dump(split_data[split_type][split_name], f)


def process_sql_data(data: List[JsonDict],
                     use_all_sql: bool = False,
                     use_all_queries: bool = True,
                     remove_unneeded_aliases: bool = False,
                     schema: Dict[str, List[TableColumn]] = None,
                     load_spans: bool = False) -> Iterable[SqlData]:
    """
    A utility function for reading in text2sql data. The blob is
    the result of loading the json from a file produced by the script
    ``scripts/reformat_text2sql_data.py``.

    Parameters
    ----------
    data : ``JsonDict``
    use_all_sql : ``bool``, optional (default = False)
        Whether to use all of the sql queries which have identical semantics,
        or whether to just use the first one.
    use_all_queries : ``bool``, (default = False)
        Whether or not to enforce query sentence uniqueness. If false,
        duplicated queries will occur in the dataset as separate instances,
        as for a given SQL query, not only are there multiple queries with
        the same template, but there are also duplicate queries.
    remove_unneeded_aliases : ``bool``, (default = False)
        The text2sql data by default creates alias names for `all` tables,
        regardless of whether the table is derived or if it is identical to
        the original (e.g SELECT TABLEalias0.COLUMN FROM TABLE AS TABLEalias0).
        This is not necessary and makes the action sequence and grammar manipulation
        much harder in a grammar based decoder. Note that this does not
        remove aliases which are legitimately required, such as when a new
        table is formed by performing operations on the original table.
    schema : ``Dict[str, List[TableColumn]]``, optional, (default = None)
        A schema to resolve primary keys against. Converts 'ID' column names
        to their actual name with respect to the Primary Key for the table
        in the schema.
    """
    for example in data:
        seen_sentences: Set[str] = set()
        for sent_info in example['sentences']:
            if sent_info['question-split'] == 'exclude':
                continue
            spans = None
            if load_spans:
                # the spans are over the text_with_variables! text could have different indices
                # convert from List[Tuple[int, int]] to string
                spans = sent_info.get('constituency_parser_spans', None)
            # Loop over the different sql statements with "equivalent" semantics
            for sql in example["sql"]:
                text_with_variables = preprocess_text(sent_info['text'].strip()).replace('``', '"').split()
                text_vars = sent_info['variables']
                query_tokens, tags = replace_variables(text_with_variables, text_vars)
                if not use_all_queries:
                    key = " ".join(query_tokens)
                    if key in seen_sentences:
                        continue
                    else:
                        seen_sentences.add(key)

                sql_tokens = clean_and_split_sql_v2(sql)
                if remove_unneeded_aliases:
                    sql_tokens = clean_unneeded_aliases(sql_tokens)
                if schema is not None:
                    sql_tokens = resolve_primary_keys_in_schema(sql_tokens, schema)

                derived_tabs, derived_cols = [], []
                if not remove_unneeded_aliases:
                    sql_tokens, derived_tabs, derived_cols = disambiguate_col_names(sql_tokens)
                sql_variables = {}
                for variable in example['variables']:
                    sql_variables[variable['name']] = {'text': variable['example'], 'type': variable['type']}

                sql_data = SqlData(text=query_tokens,
                                   text_with_variables=text_with_variables,
                                   variable_tags=tags,
                                   sql=sql_tokens,
                                   text_variables=text_vars,
                                   sql_variables=sql_variables,
                                   derived_tables=derived_tabs,
                                   derived_cols=derived_cols,
                                   spans=spans)
                yield sql_data

                # Some questions might have multiple equivalent SQL statements.
                # By default, we just use the first one. TODO(Mark): Use the shortest?
                if not use_all_sql:
                    break


def process_sql_data_attn_sup_grmr(data: List[JsonDict],
                     use_all_sql: bool = False,
                     use_all_queries: bool = True,
                     remove_unneeded_aliases: bool = False,
                     schema: Dict[str, List[TableColumn]] = None) -> Iterable[SqlData]:
    """
    A utility function for reading in text2sql data. The blob is
    the result of loading the json from a file produced by the script
    ``scripts/reformat_text2sql_data.py``.

    Parameters
    ----------
    data : ``JsonDict``
    use_all_sql : ``bool``, optional (default = False)
        Whether to use all of the sql queries which have identical semantics,
        or whether to just use the first one.
    use_all_queries : ``bool``, (default = False)
        Whether or not to enforce query sentence uniqueness. If false,
        duplicated queries will occur in the dataset as separate instances,
        as for a given SQL query, not only are there multiple queries with
        the same template, but there are also duplicate queries.
    remove_unneeded_aliases : ``bool``, (default = False)
        The text2sql data by default creates alias names for `all` tables,
        regardless of whether the table is derived or if it is identical to
        the original (e.g SELECT TABLEalias0.COLUMN FROM TABLE AS TABLEalias0).
        This is not necessary and makes the action sequence and grammar manipulation
        much harder in a grammar based decoder. Note that this does not
        remove aliases which are legitimately required, such as when a new
        table is formed by performing operations on the original table.
    schema : ``Dict[str, List[TableColumn]]``, optional, (default = None)
        A schema to resolve primary keys against. Converts 'ID' column names
        to their actual name with respect to the Primary Key for the table
        in the schema.
    """
    for example in data:
        seen_sentences: Set[str] = set()
        for sent_info in example['sentences']:
            if sent_info['question-split'] == 'exclude':
                continue
            # Loop over the different sql statements with "equivalent" semantics
            for sql in example["sql"]:
                text_with_variables = sent_info['text'].strip().split()
                text_vars = sent_info['variables']
                try:
                    alignment_with_variables = sent_info["alignment"].strip().split()
                except:
                    alignment_with_variables = None

                query_tokens, tags = replace_variables(text_with_variables, text_vars)
                if not use_all_queries:
                    key = " ".join(query_tokens)
                    if key in seen_sentences:
                        continue
                    else:
                        seen_sentences.add(key)

                sql_tokens = clean_and_split_sql_v2(sql)
                if remove_unneeded_aliases:
                    sql_tokens = clean_unneeded_aliases(sql_tokens)
                if schema is not None:
                    sql_tokens = resolve_primary_keys_in_schema(sql_tokens, schema)

                derived_tabs, derived_cols = [], []
                if not remove_unneeded_aliases:
                    sql_tokens, derived_tabs, derived_cols = disambiguate_col_names(sql_tokens)
                sql_variables = {}
                for variable in example['variables']:
                    sql_variables[variable['name']] = {'text': variable['example'], 'type': variable['type']}

                sql_data = SqlData(text=query_tokens,
                                   text_with_variables=text_with_variables,
                                   variable_tags=tags,
                                   sql=sql_tokens,
                                   text_variables=text_vars,
                                   sql_variables=sql_variables,
                                   derived_tables=derived_tabs,
                                   derived_cols=derived_cols,
                                   alignment_with_variables=alignment_with_variables)
                yield sql_data

                # Some questions might have multiple equivalent SQL statements.
                # By default, we just use the first one. TODO(Mark): Use the shortest?
                if not use_all_sql:
                    break


def retokenize_gold(gold_tokens):
    sql = " ".join(gold_tokens)
    sql = re.sub(r"([A-Za-z0-9_]+alias[0-9])\.([A-Za-z0-9_]+|\* )", "\g<1> . \g<2>", sql)

    return sql.split()


if __name__ == '__main__':
    split_data('/media/disk1/inbaro/data/tmp_semparse/geography', 'geo.json')
    # split_data('/media/disk1/inbaro/data/tmp_semparse/scholar', 'scholar.json')
    # data_1 = split_data('/media/disk1/inbaro/data/tmp_semparse', 'atis.json')
    # split_data('/media/disk1/inbaro/data/tmp_semparse', 'advising.json')
    # split_data('/media/disk1/inbaro/data/tmp_semparse', 'atis.json')
    # path = '/media/disk1/inbaro/data/tmp_semparse/atis/query_split/dev.json'
    # with open(path, "r") as data_file:
    #     data = json.load(data_file)
    # first_set = set()
    # second_set = set()
    # for text, sql in process_sql_data_standard(data, True):
    #     first_set.add(text)
    # for text, sql in process_sql_data_test(data, True):
    #     second_set.add(text)
    # print(len(second_set.difference(first_set)))
    # print("++", len(second_set), len(first_set))
    # print(len(first_set.difference(second_set)))




