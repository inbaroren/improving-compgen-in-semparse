"""
A context represents the SQL context in which an utterance appears
for the any of the text2sql datasets, with the grammar and the valid actions.
For anonymized data (all strings and numbers are replaced with etypes variables)
we use untyped grammar.
"""
from typing import List, Dict, Tuple
from sqlite3 import Cursor


from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import TableColumn
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import column_has_numeric_type
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import column_has_string_type

GRAMMAR_DICTIONARY = {}
GRAMMAR_DICTIONARY["statement"] = ['(query ws ";")', '(query ws)']
GRAMMAR_DICTIONARY["query"] = ['(ws select_core ws groupby_clause ws orderby_clause ws limit)',
                               '(ws select_core ws groupby_clause ws orderby_clause)',
                               '(ws select_core ws groupby_clause ws limit)',
                               '(ws select_core ws orderby_clause ws limit)',
                               '(ws select_core ws groupby_clause)',
                               '(ws select_core ws orderby_clause)',
                               '(ws select_core)']

GRAMMAR_DICTIONARY["select_core"] = ['(select_with_distinct ws select_results ws from_clause ws "WHERE" ws where_clause)',
                                     '(select_with_distinct ws select_results ws from_clause)']
                                     # , '(select_with_distinct ws select_results ws where_clause)']
                                     # , '(select_with_distinct ws select_results)']
GRAMMAR_DICTIONARY["select_with_distinct"] = ['(ws "SELECT" ws "DISTINCT")', '(ws "SELECT")']
GRAMMAR_DICTIONARY["select_results"] = ['(ws select_result ws "," ws select_results)', '(ws select_result)']
GRAMMAR_DICTIONARY["select_result"] = ['"*"', '(table_name ws ".*")', '(table_name ws ". *")', 'col_ref',
                                       '(function ws "AS" wsp col_alias)', 'function', '(col_ref ws "AS" wsp col_alias)']

GRAMMAR_DICTIONARY["from_clause"] = ['ws "FROM" ws source']
GRAMMAR_DICTIONARY["source"] = ['(ws single_source ws "," ws source)', '(ws single_source)']
GRAMMAR_DICTIONARY["single_source"] = ['source_table', 'source_subq']
GRAMMAR_DICTIONARY["source_table"] = ['("TABLE_PLACEHOLDER" ws "AS" wsp table_name)']
GRAMMAR_DICTIONARY["source_subq"] = ['("(" ws query ws ")" ws "AS" ws subq_alias)', '("(" ws query ws ")")']
GRAMMAR_DICTIONARY["limit"] = ['("LIMIT" ws non_literal_number)', '("LIMIT" ws value)']

GRAMMAR_DICTIONARY["where_clause"] = ['(ws expr ws where_conj)',
                                      '(ws expr ws where_or)',
                                      '(ws "(" ws where_clause ws ")" ws where_or)',
                                      '(ws "(" ws where_clause ws ")" ws where_conj)',
                                      '(ws "(" ws where_clause ws ")")',
                                      '(ws expr)']
# GRAMMAR_DICTIONARY["where_clause"] = ['(ws "WHERE" wsp expr ws where_conj)', '(ws "WHERE" wsp expr)']
GRAMMAR_DICTIONARY["where_conj"] = ['(ws "AND" wsp where_clause)']
GRAMMAR_DICTIONARY["where_or"] = ['(ws "OR" wsp where_clause)']

GRAMMAR_DICTIONARY["groupby_clause"] = ['(ws "GROUP" ws "BY" ws group_clause ws "HAVING" ws expr)',
                                        '(ws "GROUP" ws "BY" ws group_clause)']
GRAMMAR_DICTIONARY["group_clause"] = ['(ws expr ws "," ws group_clause)', '(ws expr)']

GRAMMAR_DICTIONARY["orderby_clause"] = ['ws "ORDER" ws "BY" ws order_clause']
GRAMMAR_DICTIONARY["order_clause"] = ['(ordering_term ws "," ws order_clause)', 'ordering_term']
GRAMMAR_DICTIONARY["ordering_term"] = ['(ws expr ws ordering)', '(ws expr)', '(ws col_alias ws ordering)']
GRAMMAR_DICTIONARY["ordering"] = ['(ws "ASC")', '(ws "DESC")']

GRAMMAR_DICTIONARY["table_name"] = []
GRAMMAR_DICTIONARY['column_name'] = []
GRAMMAR_DICTIONARY["col_ref"] = ['(ws subq_alias ws "." ws col_alias)', '(ws table_name ws "." ws col_alias)', '(ws subq_alias ws "." ws column_name)']

GRAMMAR_DICTIONARY['col_alias'] = ['"DERIVED_FIELDalias0"', '"DERIVED_FIELDalias1"', '"DERIVED_FIELDalias2"', '"DERIVED_FIELDalias3"', '"DERIVED_FIELDalias4"', '"DERIVED_FIELDalias5"', '"DERIVED_FIELDalias6"']

GRAMMAR_DICTIONARY['subq_alias'] = ['"DERIVED_TABLEalias0"', '"DERIVED_TABLEalias1"', '"DERIVED_TABLEalias2"', '"DERIVED_TABLEalias3"', '"DERIVED_TABLEalias4"', '"DERIVED_TABLEalias5"', '"DERIVED_TABLEalias6"']

GRAMMAR_DICTIONARY["ws"] = [r'~"\s*"i']
GRAMMAR_DICTIONARY['wsp'] = [r'~"\s+"i']

GRAMMAR_DICTIONARY["expr"] = ['in_expr',
                              # Like expressions.
                              '(value wsp "LIKE" wsp string)',
                              '(value wsp "NOT" ws "LIKE" wsp value)',
                              # Between expressions.
                              '(value ws "BETWEEN" wsp value ws "AND" wsp value)',
                              '(value ws "NOT" ws "BETWEEN" wsp value ws "AND" wsp value)',
                              # Binary expressions.
                              '(value ws binaryop wsp expr)',
                              # Unary expressions.
                              '(unaryop ws expr)',
                              # Two types of null check expressions.
                              '(col_ref ws "IS" ws "NOT" ws "NULL")',
                              '(col_ref ws "IS" ws "NULL")',
                              'source_subq',
                              'value']
GRAMMAR_DICTIONARY["in_expr"] = ['(value wsp "NOT" wsp "IN" wsp string_set)',
                                 '(value wsp "IN" wsp string_set)',
                                 '(value wsp "NOT" wsp "IN" wsp expr)',
                                 '(value wsp "IN" wsp expr)']

GRAMMAR_DICTIONARY["value"] = ['parenval', '"YEAR(CURDATE())"', '"YEAR ( CURDATE ( ))"', 'number', 'boolean',
                               'function', 'col_ref', 'string']
GRAMMAR_DICTIONARY["parenval"] = ['"(" ws expr ws ")"']
GRAMMAR_DICTIONARY["function"] = ['(fname ws "(" ws "DISTINCT" ws arg_list_or_star ws ")")',
                                  '(fname ws "(" ws arg_list_or_star ws ")")',
                                  '"YEAR(CURDATE())"', '"YEAR ( CURDATE ( ))"']

GRAMMAR_DICTIONARY["arg_list_or_star"] = ['arg_list', '"*"', '"1"']
GRAMMAR_DICTIONARY["arg_list"] = ['(expr ws "," ws arg_list)', 'expr']
 # TODO(MARK): Massive hack, remove and modify the grammar accordingly
GRAMMAR_DICTIONARY["non_literal_number"] = ['"1"', '"2"', '"3"', '"4"']
# GRAMMAR_DICTIONARY["number"] = [r'~"\d*\.?\d+"i', "'3'", "'4'"]
GRAMMAR_DICTIONARY["number"] = ['(ws "value")']
GRAMMAR_DICTIONARY["string_set"] = ['ws "(" ws string_set_vals ws ")"']
GRAMMAR_DICTIONARY["string_set_vals"] = ['(string ws "," ws string_set_vals)', 'string']
# GRAMMAR_DICTIONARY["string"] = ['~"\'.*?\'"i']
GRAMMAR_DICTIONARY["string"] = ['("\'" ws "value" ws "\'")']
GRAMMAR_DICTIONARY["fname"] = ['"COUNT"', '"SUM"', '"MAX"', '"MIN"', '"AVG"', '"ALL"']
GRAMMAR_DICTIONARY["boolean"] = ['"true"', '"false"']

GRAMMAR_DICTIONARY["binaryop"] = ['"+"', '"-"', '"*"', '"/"', '"="', '"<>"',
                                  '">="', '"<="', '">"', '"<"', '"LIKE"'
                                  ]
GRAMMAR_DICTIONARY["unaryop"] = ['"+"', '"-"', '"not"', '"NOT"']


GLOBAL_DATASET_VALUES: Dict[str, List[str]] = {
        # These are used to check values are present, or numbers of authors.
        "scholar": ["0", "1", "2", '\'journalname0\'', '\'year0\''],
        "atis": ['\'meal_code1\'', '\'restriction_code0\''],
        # 0 is used for "sea level", 750 is a "major" lake, and 150000 is a "major" city.
        "geography": ["0", "750", "150000", '\'country_name0\''],
        # This defines what an "above average" restaurant is.
        "restaurants": ["2.5"],
        # default values for morning times and semester
        "advising": ["'N / A'", "'CS-LSA'", "'10:00:00'", "'08:00:00'", "'12:00:00'", "'17:00:00'"]
}

GLOBAL_DATASET_RULES: Dict[str, List[tuple]] = {
    "advising": [('select_results', '(ws function ws binaryop ws non_literal_number)'),
                 ('non_literal_number', '"0"'),
                 ('non_literal_number', '"5"'),
                 ('non_literal_number', '"100"'),
                 ('value', 'string_function'),
                 ('string_function', '(ws string_fname ws "(" ws col_ref ws ")")'),
                 ('string_fname', '"LOWER"'),
                 ('string_fname', '"UPPER"'),
                 ('where_clause', '(ws source_subq ws binaryop ws non_literal_number)')]
}

DATASET_MAX_ALIAS_NUM: Dict[str, int] = {
    "scholar": 3,
    "geography": 6,
    "advising": 7,
    "atis": 7
}

OOD_ENTITIES: Dict[str, List[str]] = {
    "scholar": [],
    "geography": [],
    "advising": [],
    "atis": []
}


def test_all_sql_tokens_in_grammar(grammar_dictionary: Dict[str, List[str]],
                                   sql: List[str]) -> bool:
    all_terminal_expressions = [inner for outer in grammar_dictionary.values() for inner in outer]
    return set(all_terminal_expressions).issuperset(set(sql))


def update_grammar_with_tables(grammar_dictionary: Dict[str, List[str]],
                               schema: Dict[str, List[TableColumn]],
                               dataset_name: str
                               ) -> None:
    max_alias_index = DATASET_MAX_ALIAS_NUM[dataset_name]
    table_names = sorted([f'"{table}alias{i}"' for table in
                          list(schema.keys()) for i in range(max_alias_index)], reverse=True)
    grammar_dictionary['table_name'] += table_names

    all_columns = set()
    all_free_columns = set()
    for table, col_list in schema.items():

        all_free_columns.update([
            f'"{column.name.upper()}"' for column in col_list if column.name != '*'
        ])

        all_columns.update(
            [f'"{table.upper()}alias{i}.{column.name.upper()}"'
             for column in col_list
             for i in range(max_alias_index)
             if column.name != '*'])

    sorted_free_columns = sorted([column for column in all_free_columns], reverse=True)
    grammar_dictionary['column_name'] += sorted_free_columns

    sorted_columns = sorted([column for column in all_columns], reverse=True)
    grammar_dictionary['col_ref'] += sorted_columns


def update_grammar_with_table_values(grammar_dictionary: Dict[str, List[str]],
                                     schema: Dict[str, List[TableColumn]],
                                     cursor: Cursor) -> None:

    for table_name, columns in schema.items():
        for column in columns:
            cursor.execute(f'SELECT DISTINCT {table_name}.{column.name} FROM {table_name}')
            results = [x[0] for x in cursor.fetchall()]
            if column_has_string_type(column):
                productions = sorted([f'"{str(result)}"' for result in results], reverse=True)
                grammar_dictionary["string"].extend(productions)
            elif column_has_numeric_type(column):
                productions = sorted([f'"{str(result)}"' for result in results], reverse=True)
                grammar_dictionary["number"].extend(productions)


def update_grammar_with_global_values(grammar_dictionary: Dict[str, List[str]], dataset_name: str):
    values = GLOBAL_DATASET_VALUES.get(dataset_name, [])
    values_for_grammar = [f'"{str(value)}"' for value in values]
    grammar_dictionary["value"] = values_for_grammar + grammar_dictionary["value"]

    rules = GLOBAL_DATASET_RULES.get(dataset_name, [])
    for rule_non_terminal, rule_value in rules:
        grammar_dictionary[rule_non_terminal] = [rule_value] + grammar_dictionary.get(rule_non_terminal, [])


def update_grammar_to_be_variable_free(grammar_dictionary: Dict[str, List[str]]):
    """
    SQL is a predominately variable free language in terms of simple usage, in the
    sense that most queries do not create references to variables which are not
    already static tables in a dataset. However, it is possible to do this via
    derived tables. If we don't require this functionality, we can tighten the
    grammar, because we don't need to support aliased tables.
    """

    # Tables in variable free grammars cannot be aliased, so we
    # remove this functionality from the grammar.
    grammar_dictionary["select_result"] = ['"*"', '(table_name ws ".*")', 'expr']

    # Similarly, collapse the definition of a source table
    # to not contain aliases and modify references to subqueries.
    grammar_dictionary["single_source"] = ['table_name', '("(" ws query ws ")")']
    del grammar_dictionary["source_subq"]
    del grammar_dictionary["source_table"]

    grammar_dictionary["expr"] = ['in_expr',
                                  '(value wsp "LIKE" wsp string)',
                                  '(value ws "BETWEEN" wsp value ws "AND" wsp value)',
                                  '(value ws binaryop wsp expr)',
                                  '(unaryop ws expr)',
                                  '(col_ref ws "IS" ws "NOT" ws "NULL")',
                                  '(col_ref ws "IS" ws "NULL")',
                                  # This used to be source_subq - now
                                  # we don't need aliases, we can colapse it to queries.
                                  '("(" ws query ws ")")',
                                  'value']

    # Finally, remove the ability to reference an arbitrary name,
    # because now we don't have aliased tables, we don't need
    # to recognise new variables.
    del grammar_dictionary["name"]


def update_grammar_with_untyped_entities(grammar_dictionary: Dict[str, List[str]]) -> None:
    """
    Variables can be treated as numbers or strings if their type can be inferred -
    however, that can be difficult, so instead, we can just treat them all as values
    and be a bit looser on the typing we allow in our grammar. Here we just remove
    all references to number and string from the grammar, replacing them with value.
    """
    grammar_dictionary["string_set_vals"] = ['(value ws "," ws string_set_vals)', 'value']
    grammar_dictionary["value"].remove('string')
    grammar_dictionary["value"].remove('number')
    # grammar_dictionary["limit"] = ['("LIMIT" ws "1")', '("LIMIT" ws value)']
    grammar_dictionary["expr"][1] = '(value wsp "LIKE" wsp value)'
    del grammar_dictionary["string"]
    del grammar_dictionary["number"]


def update_grammar_values_with_variables(grammar_dictionary: Dict[str, List[str]],
                                         prelinked_entities: Dict[str, Dict[str, str]],
                                         dataset_name: str = "") -> None:

    for variable, _ in prelinked_entities.items():
        grammar_dictionary["value"] = [f'"\'{variable}\'"'] + [f'"{variable}"'] + grammar_dictionary["value"]


def update_grammar_with_derived_tabs_and_cols(grammar_dictionary: Dict[str, List[str]],
                                              derived_tabs: List[str],
                                              derived_cols: List[Tuple[str, str]]) -> None:

    # for derived_tab in derived_tabs:
    #     grammar_dictionary['subq_alias'] = [f'"{derived_tab}"'] + grammar_dictionary["subq_alias"]
    # all_columns = set()
    # all_col_aliases = set()
    # for tab, col in derived_cols:
    #     # if tab == 'DERIVED_TABLEalias0' and col == 'LENGTH':
    #     #     # Some shortcuts cause parsing error at test
    #     #     continue
    #     all_columns.update([f'"{tab}.{col}"'])
    #     sorted_columns = sorted([column for column in all_columns], reverse=True)
    #     grammar_dictionary['col_ref'] += sorted_columns
    #     if col.startswith("DERIVED"):
    #         all_col_aliases.add(f'"{col}"')
    # sorted_aliases = sorted(list(all_col_aliases))
    # grammar_dictionary['col_alias'] += sorted_aliases
    return None


def update_grammar_numbers_and_strings_with_variables(grammar_dictionary: Dict[str, List[str]], # pylint: disable=invalid-name
                                                      prelinked_entities: Dict[str, Dict[str, str]],
                                                      columns: Dict[str, TableColumn]) -> None:
    for variable, info in prelinked_entities.items():
        variable_column = info["type"].upper()
        matched_column = columns.get(variable_column, None)

        if matched_column is not None:
            # Try to infer the variable's type by matching it to a column in
            # the database. If we can't, we just add it as a value.
            if column_has_numeric_type(matched_column):
                grammar_dictionary["number"] = [f'"{variable}"'] + grammar_dictionary["number"]
            elif column_has_string_type(matched_column):
                grammar_dictionary["string"] = [f'"\'{variable}\'"'] + grammar_dictionary["string"]
            else:
                grammar_dictionary["value"] = [f'"\'{variable}\'"'] + [f'"{variable}"'] + grammar_dictionary["value"]
        # Otherwise, try to infer by looking at the actual value:
        else:
            try:
                # This is what happens if you try and do type inference
                # in a grammar which parses _strings_ in _Python_.
                # We're just seeing if the python interpreter can convert
                # to to a float - if it can, we assume it's a number.
                float(info["text"])
                is_numeric = True
            except ValueError:
                is_numeric = False
            if is_numeric:
                grammar_dictionary["number"] = [f'"{variable}"'] + grammar_dictionary["number"]
            elif info["text"].replace(" ", "").isalpha():
                grammar_dictionary["string"] = [f'"\'{variable}\'"'] + grammar_dictionary["string"]
            else:
                grammar_dictionary["value"] = [f'"\'{variable}\'"'] + [f'"{variable}"'] + grammar_dictionary["value"]
