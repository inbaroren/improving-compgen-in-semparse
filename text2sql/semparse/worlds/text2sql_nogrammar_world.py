from typing import List, Tuple, Dict
from copy import deepcopy
from sqlite3 import Cursor
import os

from parsimonious import Grammar
from parsimonious.exceptions import ParseError

from allennlp.common.checks import ConfigurationError
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor
from allennlp.semparse.contexts.sql_context_utils import format_grammar_string, initialize_valid_actions
from allennlp.data.dataset_readers.dataset_utils.text2sql_utils import read_dataset_schema
from text2sql.semparse.contexts.text2sql_nogrammar_table_context import GRAMMAR_DICTIONARY
from text2sql.semparse.contexts.text2sql_nogrammar_table_context import update_grammar_with_table_values
from text2sql.semparse.contexts.text2sql_nogrammar_table_context import update_grammar_with_tables
from text2sql.semparse.contexts.text2sql_nogrammar_table_context import update_grammar_with_global_values, update_grammar_with_tokens
from text2sql.semparse.contexts.text2sql_nogrammar_table_context import update_grammar_values_with_variables
from text2sql.semparse.contexts.text2sql_nogrammar_table_context import update_grammar_numbers_and_strings_with_variables


class Text2SqlNoGrammarWorld:
    """
    World representation for any of the Text2Sql datasets.

    Parameters
    ----------
    schema_path: ``str``
        A path to a schema file which we read into a dictionary
        representing the SQL tables in the dataset, the keys are the
        names of the tables that map to lists of the table's column names.
    cursor : ``Cursor``, optional (default = None)
        An optional cursor for a database, which is used to add
        database values to the grammar.
    use_prelinked_entities : ``bool``, (default = True)
        Whether or not to use the pre-linked entities from the text2sql data.
        We take this parameter here because it effects whether we need to add
        table values to the grammar.
    variable_free : ``bool``, optional (default = True)
        Denotes whether the data being parsed by the grammar is variable free.
        If it is, the grammar is modified to be less expressive by removing
        elements which are not necessary if the data is variable free.
    use_untyped_entities : ``bool``, optional (default = False)
        Whether or not to try to infer the types of prelinked variables.
        If not, they are added as untyped values to the grammar instead.
    """
    def __init__(self,
                 schema_path: str,
                 cursor: Cursor = None,
                 use_prelinked_entities: bool = True) -> None:
        self.cursor = cursor
        self.schema = read_dataset_schema(schema_path)
        self.columns = {column.name: column for table in self.schema.values() for column in table}
        self.dataset_name = os.path.basename(schema_path).split("-")[0]
        self.use_prelinked_entities = use_prelinked_entities

        # NOTE: This base dictionary should not be modified.
        self.base_grammar_dictionary = self._initialize_grammar_dictionary(deepcopy(GRAMMAR_DICTIONARY))

    def get_action_sequence_and_all_actions(self,
                                            query: List[str] = None,
                                            prelinked_entities: Dict[str, Dict[str, str]] = None) -> Tuple[List[str], List[str]]: # pylint: disable=line-too-long
        grammar_with_context = deepcopy(self.base_grammar_dictionary)

        if not self.use_prelinked_entities and prelinked_entities is not None:
            raise ConfigurationError("The Text2SqlNoGrammarWorld was specified to not use prelinked "
                                     "entities, but prelinked entities were passed.")
        prelinked_entities = prelinked_entities or {}

        update_grammar_numbers_and_strings_with_variables(grammar_with_context,
                                                              prelinked_entities,
                                                              self.columns)
        update_grammar_with_tokens(grammar_with_context,
                                    query)

        grammar = Grammar(format_grammar_string(grammar_with_context))

        valid_actions = initialize_valid_actions(grammar)
        all_actions = set()
        for action_list in valid_actions.values():
            all_actions.update(action_list)
        sorted_actions = sorted(all_actions)

        sql_visitor = SqlVisitor(grammar)
        try:
            action_sequence = sql_visitor.parse(" ".join(query)) if query else []
        except ParseError as e:
            print("\nParse Error - details:\n", e.pos, '\n', e.expr, '\n', e.text)
            action_sequence = None
        except RecursionError as er:
            print("\nParse recursion error - details:\n", " ".join(query), '\n', grammar_with_context['terminal'])
            action_sequence = None

        return action_sequence, sorted_actions

    def _initialize_grammar_dictionary(self, grammar_dictionary: Dict[str, List[str]]) -> Dict[str, List[str]]:
        # Add all the table and column names to the grammar.
        update_grammar_with_tables(grammar_dictionary, self.schema)

        if self.cursor is not None and not self.use_prelinked_entities:
            # Now if we have strings in the table, we need to be able to
            # produce them, so we find all of the strings in the tables here
            # and create production rules from them. We only do this if
            # we haven't pre-linked entities, because if we have, we don't
            # need to be able to generate the values - just the placeholder
            # symbols which link to them.
            update_grammar_with_table_values(grammar_dictionary, self.schema, self.cursor)

        # Finally, update the grammar with global, non-variable values
        # found in the dataset, if present.
        update_grammar_with_global_values(grammar_dictionary, self.dataset_name)

        return grammar_dictionary

    def is_global_rule(self, production_rule: str) -> bool:
        if self.use_prelinked_entities:
            # we are checking -4 as is not a global rule if we
            # see the 0 in the a rule like 'value -> ["\'city_name0\'"]'
            if "value" in production_rule and production_rule[-4].isnumeric():
                return False
        return True
