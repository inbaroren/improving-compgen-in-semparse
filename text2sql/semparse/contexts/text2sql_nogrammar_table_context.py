"""
A ``Text2SqlTableContext`` represents the SQL context in which an utterance appears
for any of the text2sql datasets, with the grammar and the valid actions.
This is a DUMMY GRAMMAR to produce action sequence equivalent to token sequence prediction.
"""
from typing import List, Dict
from sqlite3 import Cursor


from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import TableColumn
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import column_has_numeric_type
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import column_has_string_type

GRAMMAR_DICTIONARY = {}

# S-> terminal S | terminal
# terminal-> SQL VOCABULARY + SCHEMA SPECIFIC VOCABULARY

# statement is the start non-termnial!!! :0
GRAMMAR_DICTIONARY["statement"] = ['(query ws ";")', '(query ws)']
GRAMMAR_DICTIONARY["query"] = ['(ws terminal ws terminal ws terminal ws terminal ws terminal ws terminal ws terminal ws terminal ws terminal query)',
                               '(ws terminal ws terminal ws terminal query)',
                               '(ws terminal query)',
                               '(ws terminal)']
GRAMMAR_DICTIONARY["terminal"] = []  # update with decoder vocabulary

GRAMMAR_DICTIONARY["terminal"].extend(['"SELECT"', '"DISTINCT"', '"FROM"', '"WHERE"', '"GROUP"', '"ORDER"', '","',
                                       '"("', '")"', '"AS"', '"LIMIT"', '"TOP"', '"BY"', '"HAVING"', '"ASC"', '"DESC"',
                                       '"."', '"LIKE"', '"BETWEEN"', '"IN"', '"IS"', '"NULL"', '"YEAR(CURDATE())"',
                                       '"*"', '"ANY"'])
GRAMMAR_DICTIONARY["terminal"].extend(['"COUNT"', '"SUM"', '"MAX"', '"MIN"', '"AVG"', '"ALL"'])
GRAMMAR_DICTIONARY["terminal"].extend(['"true"', '"false"'])

GRAMMAR_DICTIONARY["terminal"].extend(['"+"', '"-"', '"*"', '"/"', '"="', '"<>"', '">="', '"<="', '">"', '"<"', '"AND"',
                                       '"OR"', '"LIKE"'])
GRAMMAR_DICTIONARY["terminal"].extend(['"+"', '"-"', '"not"', '"NOT"'])

GRAMMAR_DICTIONARY["ws"] = [r'~"\s*"i']

GLOBAL_DATASET_VALUES: Dict[str, List[str]] = {
        # These are used to check values are present, or numbers of authors.
        "scholar": ["0", "1", "2"],
        # 0 is used for "sea level", 750 is a "major" lake, and 150000 is a "major" city.
        "geography": ["0", "750", "150000"],
        # This defines what an "above average" restaurant is.
        "restaurants": ["2.5"]
}


def update_grammar_with_tables(grammar_dictionary: Dict[str, List[str]],
                               schema: Dict[str, List[TableColumn]]) -> None:
    table_names = sorted([f'"{table}"' for table in
                          list(schema.keys())], reverse=True)
    grammar_dictionary['terminal'].extend(table_names)

    all_columns = set()
    for table in schema.values():
        all_columns.update([column.name for column in table])
    sorted_columns = sorted([f'"{column}"' for column in all_columns], reverse=True)
    grammar_dictionary['terminal'].extend(sorted_columns)
    grammar_dictionary['terminal'] = sorted(grammar_dictionary['terminal'], reverse=True)


def update_grammar_with_table_values(grammar_dictionary: Dict[str, List[str]],
                                     schema: Dict[str, List[TableColumn]],
                                     cursor: Cursor) -> None:

    for table_name, columns in schema.items():
        for column in columns:
            cursor.execute(f'SELECT DISTINCT {table_name}.{column.name} FROM {table_name}')
            results = [x[0] for x in cursor.fetchall()]
            if column_has_string_type(column):
                productions = sorted([f'"{str(result)}"' for result in results], reverse=True)
                grammar_dictionary["terminal"].extend(productions)
            elif column_has_numeric_type(column):
                productions = sorted([f'"{str(result)}"' for result in results], reverse=True)
                grammar_dictionary["terminal"].extend(productions)
    grammar_dictionary['terminal'] = sorted(grammar_dictionary['terminal'], reverse=True)


def update_grammar_with_global_values(grammar_dictionary: Dict[str, List[str]], dataset_name: str):

    values = GLOBAL_DATASET_VALUES.get(dataset_name, [])
    values_for_grammar = [f'"{str(value)}"' for value in values]
    grammar_dictionary["terminal"].extend(values_for_grammar)
    grammar_dictionary['terminal'] = sorted(grammar_dictionary['terminal'], reverse=True)


def update_grammar_values_with_variables(grammar_dictionary: Dict[str, List[str]],
                                         prelinked_entities: Dict[str, Dict[str, str]]) -> None:

    for variable, _ in prelinked_entities.items():
        grammar_dictionary["terminal"].extend([f'"\'{variable}\'"', f'"{variable}"'])
    grammar_dictionary['terminal'] = sorted(grammar_dictionary['terminal'], reverse=True)


def update_grammar_numbers_and_strings_with_variables(grammar_dictionary: Dict[str, List[str]], # pylint: disable=invalid-name
                                                      prelinked_entities: Dict[str, Dict[str, str]],
                                                      columns: Dict[str, TableColumn]) -> None:
    for variable, info in prelinked_entities.items():
        variable_column = info["type"].upper()
        matched_column = columns.get(variable_column, None)
        new_variables = [f'"\'{variable}\'"']

        if matched_column is not None:
            if not column_has_string_type(matched_column):
                new_variables.append(f'"{variable}"')
        grammar_dictionary["terminal"].extend(new_variables)

    grammar_dictionary['terminal'] = sorted(grammar_dictionary['terminal'], reverse=True)


def update_grammar_with_tokens(grammar_dictionary: Dict[str, List[str]],
                               sql_query_tokens: List[str]) -> None:
    for sql_token in sql_query_tokens:
        if sql_token not in grammar_dictionary['terminal']:
            # TODO: change to a set until all updates are called /one update
            grammar_dictionary['terminal'].append(f'"{str(sql_token)}"')

    cur_terminals = list(set(grammar_dictionary['terminal']))
    grammar_dictionary['terminal'] = sorted(cur_terminals, reverse=True)


