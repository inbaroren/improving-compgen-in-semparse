import json
from pathlib import Path
import re
import collections
import numpy as np
from random import shuffle
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from text2sql.data.dataset_readers.dataset_utils.text2sql_utils import read_schema_dict
from text2sql.data.dataset_readers.dataset_utils.text2sql_utils import clean_unneeded_aliases, \
    clean_and_split_sql, \
    resolve_primary_keys_in_schema

# in the schema I got columns and tables. I should get all the columns and tables appearing in a query and then create
# a dictionary replacing them with tokens COL# and TAB#

SQL = [
    "SELECT CITYalias0.CITY_NAME FROM CITY AS CITYalias0 WHERE CITYalias0.POPULATION = "\
    "( SELECT MAX( CITYalias1.POPULATION ) FROM CITY AS CITYalias1 "\
    "WHERE CITYalias1.STATE_NAME = \"state_name0\" ) "\
    "AND CITYalias0.STATE_NAME = \"state_name0\" ;",
    "SELECT RIVERalias0.RIVER_NAME FROM RIVER AS RIVERalias0 WHERE RIVERalias0.TRAVERSE IN " \
    "( SELECT CITYalias0.STATE_NAME FROM CITY AS CITYalias0 WHERE CITYalias0.POPULATION = " \
    "( SELECT MAX( CITYalias1.POPULATION ) FROM CITY AS CITYalias1 ) ) ;",
    "SELECT STATEalias0.CAPITAL FROM CITY AS CITYalias0 , STATE AS STATEalias0 WHERE CITYalias0.POPULATION <= 150000 "
    "AND STATEalias0.CAPITAL = CITYalias0.CITY_NAME ;",
    "SELECT DISTINCT COURSEalias0.DEPARTMENT , COURSEalias0.NAME , COURSEalias0.NUMBER , PROGRAM_COURSEalias0.WORKLOAD ,"
    " PROGRAM_COURSEalias0.WORKLOAD FROM COURSE AS COURSEalias0 , PROGRAM_COURSE AS PROGRAM_COURSEalias0 "
    "WHERE PROGRAM_COURSEalias0.CATEGORY LIKE \"%requirement0%\" AND PROGRAM_COURSEalias0.COURSE_ID = "
    "COURSEalias0.COURSE_ID AND PROGRAM_COURSEalias0.WORKLOAD = ( SELECT MIN( PROGRAM_COURSEalias1.WORKLOAD ) "
    "FROM PROGRAM_COURSE AS PROGRAM_COURSEalias1 WHERE PROGRAM_COURSEalias1.CATEGORY LIKE \"%requirement0%\" ) ;"
]
LINKED_ENTS = [
    [{'name': 'state_name0', 'type': 'state_name'}],
    [],
    [],
    [{'name': 'requirement0'}]
]
SCHEMA = [
    {'CITY': ['POPULATION', 'STATE_NAME', 'CITY_NAME']},
    {'RIVER': ['RIVER_NAME','TRAVERSE'], 'CITY': ['STATE_NAME', 'POPULATION']},
    {'STATE': ['CAPITAL'], 'CITY': ['POPULATION', 'CITY_NAME']},
    {'COURSE': ['DEPARTMENT', 'COURSE_ID', 'NUMBER', 'NAME'], 'PROGRAM_COURSE': ['CATEGORY', 'WORKLOAD', 'CATEGORY']}
]
EXPECTED = [
    (
        {'TAB1': 'CITY'},
        {'TAB1': {'COL1': 'CITY_NAME', 'COL2': 'POPULATION', 'COL3': 'STATE_NAME'}},
        "SELECT TAB1alias0.COL1 FROM TAB1 AS TAB1alias0 "
        "WHERE TAB1alias0.COL2 = ( SELECT MAX( TAB1alias1.COL2 ) "
        "FROM TAB1 AS TAB1alias1 WHERE TAB1alias1.COL3 = \"value\" ) "
        "AND TAB1alias0.COL3 = \"value\" ;"
    ),
    (
        {'TAB1': 'RIVER', 'TAB2': 'CITY'},
        {'TAB1': {'COL1': 'RIVER_NAME','COL2': 'TRAVERSE'},
         'TAB2': {'COL1': 'STATE_NAME', 'COL2': 'POPULATION'}},
        "SELECT TAB1alias0.COL1 FROM TAB1 AS TAB1alias0 WHERE TAB1alias0.COL2 IN "
        "( SELECT TAB2alias0.COL1 FROM TAB2 AS TAB2alias0 WHERE TAB2alias0.COL2 = "
        "( SELECT MAX( TAB2alias1.COL2 ) FROM TAB2 AS TAB2alias1 ) ) ;"
    ),
    (
        {'TAB1': 'STATE', 'TAB2': 'CITY'},
        {'TAB2': {'COL1': 'POPULATION', 'COL2': 'CITY_NAME'}, 'TAB1': {'COL1': 'CAPITAL'}},
        "SELECT TAB1alias0.COL1 FROM TAB2 AS TAB2alias0 , TAB1 AS TAB1alias0 WHERE TAB2alias0.COL1 <= 150000 "
        "AND TAB1alias0.COL1 = TAB2alias0.COL2 ;"
    ),
    (
        {'TAB1': 'COURSE', 'TAB2': 'PROGRAM_COURSE'},
        {'TAB1': {'COL1': 'DEPARTMENT', 'COL2': 'NAME', 'COL3': 'NUMBER', 'COL4': 'COURSE_ID'},
         'TAB2': {'COL1': 'WORKLOAD', 'COL2': 'CATEGORY', 'COL3': 'COURSE_ID'}},
        "SELECT DISTINCT TAB1alias0.COL1 , TAB1alias0.COL2 , TAB1alias0.COL3 , TAB2alias0.COL1 ,"
        " TAB2alias0.COL1 FROM TAB1 AS TAB1alias0 , TAB2 AS TAB2alias0 "
        "WHERE TAB2alias0.COL2 LIKE \"%value%\" AND TAB2alias0.COL3 = "
        "TAB1alias0.COL4 AND TAB2alias0.COL1 = ( SELECT MIN( TAB2alias1.COL1 ) "
        "FROM TAB2 AS TAB2alias1 WHERE TAB2alias1.COL2 LIKE \"%value%\" ) ;"
    )
]

D_SQL = SQL
D_SCHEMA = SCHEMA
D_EXPECTED = [
    (
        {'TAB1': 'CITY'},
        {'TAB1': {'COL1': 'CITY_NAME', 'COL2': 'POPULATION', 'COL3': 'STATE_NAME'}},
        "SELECT TAB1 . COL1 FROM TAB1 "
        "WHERE TAB1 . COL2 = ( SELECT MAX ( TAB1 . COL2 ) "
        "FROM TAB1 WHERE TAB1 . COL3 = 'state_name0' ) "
        "AND TAB1 . COL3 = 'state_name0' ;"
    ),
    (
        {'TAB1': 'RIVER', 'TAB2': 'CITY'},
        {'TAB1': {'COL1': 'RIVER_NAME','COL2': 'TRAVERSE'},
         'TAB2': {'COL1': 'STATE_NAME', 'COL2': 'POPULATION'}},
        "SELECT TAB1 . COL1 FROM TAB1 WHERE TAB1 . COL2 IN "
        "( SELECT TAB2 . COL1 FROM TAB2 WHERE TAB2 . COL2 = "
        "( SELECT MAX ( TAB2 . COL2 ) FROM TAB2 ) ) ;"
    ),
    (
        {'TAB1': 'STATE', 'TAB2': 'CITY'},
        {'TAB2': {'COL1': 'POPULATION', 'COL2': 'CITY_NAME'}, 'TAB1': {'COL1': 'CAPITAL'}},
        "SELECT TAB1 . COL1 FROM TAB2 , TAB1 WHERE TAB2 . COL1 <= 150000 "
        "AND TAB1 . COL1 = TAB2 . COL2 ;"
    ),
    (
        {'TAB1': 'COURSE', 'TAB2': 'PROGRAM_COURSE'},
        {'TAB1': {'COL1': 'DEPARTMENT', 'COL2': 'NAME', 'COL3': 'NUMBER', 'COL4': 'COURSE_ID'},
         'TAB2': {'COL1': 'WORKLOAD', 'COL2': 'CATEGORY', 'COL3': 'COURSE_ID'}},
        "SELECT DISTINCT TAB1 . COL1 , TAB1 . COL2 , TAB1 . COL3 , TAB2 . COL1 ,"
        " TAB2 . COL1 FROM TAB1 , TAB2 "
        "WHERE TAB2 . COL2 LIKE 'requirement0' AND TAB2 . COL3 = "
        "TAB1 . COL4 AND TAB2 . COL1 = ( SELECT MIN ( TAB2 . COL1 ) "
        "FROM TAB2 WHERE TAB2 . COL2 LIKE 'requirement0' ) ;"
    )
]


def prep_dealiased_sql(aliased_sql, schema):
    """
    Mimics tokenization of text2sql_parser (grammar based decoder)
    :param aliased_sql: str
    :return: str
    """
    sql_tokens = clean_and_split_sql(aliased_sql)
    sql_tokens = clean_unneeded_aliases(sql_tokens)
    # sql_tokens = resolve_primary_keys_in_schema(sql_tokens, schema)
    return " ".join(sql_tokens)


def dealiased_sql_schema_sanitize(sql: str, schema: Dict[str, List[str]]) -> Tuple[Dict, Dict, str]:
    tab_map = {}  # map table name to table global alias, i.e {CITY: TAB1}
    col_map = {}  # map column name to column global alias, i.e {CITY: {POPULATION: COL1}}
    new_sql = sql[:]
    for tab in schema.keys():
        if re.findall(f" {tab} ", new_sql):
            # get table global alias or create it if it doesn't exist
            tab_global_alias = tab_map.get(tab, f"TAB{len(tab_map) + 1}")
            tab_map[tab] = tab_global_alias
            # replace table name with table global alias
            new_sql = new_sql.replace(f" {tab} ", f" {tab_global_alias} ")
    for tab_alias in tab_map.values():
        for col_usage in re.findall(f" ({tab_alias} \. [^,\s]+) ", new_sql):
            _, col = col_usage.split(". ")
            # create table dict in col_map if it doesn't exists
            col_map[tab_alias] = col_map.get(tab_alias, {})
            # get column global alias or create it if it doesn't exist
            col_global_alias = col_map[tab_alias].get(col, f"COL{len(col_map[tab_alias]) + 1}")
            col_map[tab_alias][col] = col_global_alias
            # replace table alias with table global alias
            new_sql = new_sql.replace(col_usage, col_usage.replace(col, col_global_alias))
    return {v: k for k, v in tab_map.items()}, {k: {iv: ik for ik, iv in v.items()} for k, v in col_map.items()}, new_sql


def sql_schema_sanitize(sql: str, schema: Dict[str, List[str]], sql_vars: List[Dict[str, str]]= None) -> Tuple[Dict, Dict, str]:
    """
    Returns the sql query with all KB specific info is anonymized, and a dictionary with the anonymized details
    Assumptions:
        1. First token is SELECT
        2. Last token is ;
        3. Each column is phrased as TABLE_NAMEalias#.COLUMN_NAME
        4. Every source in the from clause is aliased with "AS", e.g "TABLE1 AS TABLE1alias#"
        5. sql is anonymized, i.e all values are replaced with a constant name: value_type#
    """
    tab_map = {}        # map table name to table global alias, i.e {CITY: TAB1}
    col_map = {}        # map column name to column global alias, i.e {CITY: {POPULATION: COL1}}
    new_sql = sql[:]
    # search for table alias
    for tab in schema.keys():
        for tab_alias_declaration in re.findall(f" ({tab}\sAS\s[^,\s]+) ", new_sql):
            _, _, alias = tab_alias_declaration.split(" ")
            alias = alias.replace(tab, "")  # example: alias0
            # get table global alias or create it if it doesn't exist
            tab_global_alias = tab_map.get(tab, f"TAB{len(tab_map) + 1}")
            tab_map[tab] = tab_global_alias
            # replace table name with table global alias
            new_sql = new_sql.replace(f" {tab} AS", f" {tab_global_alias} AS")
            # replace table alias with table global alias
            new_sql = new_sql.replace(f" {tab}{alias}", f" {tab_global_alias}{alias}")
    # search for columns
    for tab_alias in tab_map.values():
        for col_usage in re.findall(f" ({tab_alias}alias[0-9]\s*\.\s*[A-Z_]+) ", new_sql):
            _, col = col_usage.split(".")
            col = col.strip()
            # create table dict in col_map if it doesn't exists
            col_map[tab_alias] = col_map.get(tab_alias, {})
            # get column global alias or create it if it doesn't exist
            col_global_alias = col_map[tab_alias].get(col, f"COL{len(col_map[tab_alias]) + 1}")
            col_map[tab_alias][col] = col_global_alias
            # replace table alias with table global alias
            new_sql = new_sql.replace(col_usage, col_usage.replace(col, col_global_alias))

    if sql_vars is not None:
        for var in sql_vars:
            # find if it is a string or a number by finding if it is in " or not
            new_sql = re.sub(f"{var['name']}", "value", new_sql)
    else:
        for m_var in re.finditer(r" [\s\'\"%]([a-z_]+[0-9])[\s\'\"%] ", new_sql):
            if m_var.group(1).startswith('alias'):
                continue
            new_sql = re.sub(f"{m_var.group(1)}", "value", new_sql)

    return {v: k for k, v in tab_map.items()}, {k: {iv: ik for ik, iv in v.items()} for k, v in col_map.items()}, new_sql


def test():
    assert len(SQL) == len(SCHEMA) == len(EXPECTED) == len(LINKED_ENTS)
    for i in range(len(SQL)):
        print(f"##### Sample {i + 1} #####")
        res = sql_schema_sanitize(sql=SQL[i], schema=SCHEMA[i], sql_vars=LINKED_ENTS[i])
        for j in range(len(res)):
            assert EXPECTED[i][j] == res[j], f"\nResult  :{res[j]}\nExpected:{EXPECTED[i][j]}"
        print("Done")


def test_dealised():
    assert len(D_SQL) == len(D_SCHEMA) == len(D_EXPECTED)
    for i in range(len(D_SQL)):
        print(f"##### Sample {i + 1} #####")
        sql = prep_dealiased_sql(D_SQL[i], D_SCHEMA[i])
        res = dealiased_sql_schema_sanitize(sql, D_SCHEMA[i])
        for j in range(len(res)):
            assert D_EXPECTED[i][j] == res[j], f"\nResult  :{res[j]}\nExpected:{D_EXPECTED[i][j]}"
        print("Done")

