import copy
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SqlStatelet:
    """
    This class implements the SQL generation rules that are context based
    """
    def __init__(self,
                 possible_actions,
                 enabled):
        self.possible_actions = [a[0] for a in possible_actions]
        self.action_history = []
        self.tables_used = set()
        self.tables_used_by_columns = set()
        self.derived_tables = set()
        self.derived_columns = set()
        self.used_agg = False
        self.current_stack = []
        self.subqueries_stack = []
        self.enabled = enabled

    def take_action(self, production_rule: str) -> 'SqlStatelet':
        if not self.enabled:
            return self

        new_sql_state = copy.deepcopy(self)

        lhs, rhs = production_rule.split(' -> ')
        rhs_tokens = rhs.strip('[]').split(', ')
        if lhs == 'table_name':
            if self.current_stack[-1][0] in ('select_result', 'col_ref'):
                new_sql_state.tables_used_by_columns.add(rhs_tokens[0].strip('"'))
            elif self.current_stack[-1][0] == 'source_table':
                new_sql_state.tables_used.add(rhs_tokens[0].strip('"'))

        elif lhs == 'col_ref' and len(rhs_tokens) == 1:
            new_sql_state.tables_used_by_columns.add(rhs_tokens[0].strip('"').split('.')[0])

        elif lhs == 'subq_alias':
            if self.current_stack[-1][0] =='source_subq':
                new_sql_state.subqueries_stack[-1].tables_used.add(rhs_tokens[0].strip('"'))
                new_sql_state.subqueries_stack[-1].derived_tables.add(rhs_tokens[0].strip('"'))
            elif self.current_stack[-1][0] == 'col_ref':
                new_sql_state.tables_used_by_columns.add(rhs_tokens[0].strip('"'))
                new_sql_state.derived_tables.add(rhs_tokens[0].strip('"'))

        elif lhs == 'col_alias' and self.current_stack[-1][0] == 'select_result':
            new_sql_state.derived_columns.add(rhs_tokens[0].strip('"'))

        elif lhs == 'fname':
            new_sql_state.used_agg = True

        elif lhs == "source_subq":
            # new subquery new rules!
            new_sql_state.subqueries_stack.append(copy.deepcopy(new_sql_state))
            new_sql_state.tables_used = set()
            new_sql_state.tables_used_by_columns = set()
            new_sql_state.derived_tables = set()
            new_sql_state.derived_columns = set()
            new_sql_state.used_agg = False

        new_sql_state.action_history.append(production_rule)

        new_sql_state.current_stack.append([lhs, []])

        for token in rhs_tokens:
            is_terminal = token[0] == '"' and token[-1] == '"'
            if not is_terminal:
                new_sql_state.current_stack[-1][1].append(token)

        while len(new_sql_state.current_stack[-1][1]) == 0:
            # unroll the stack until there's an unfinished nonterminal in the current or previous subqueries / sections
            finished_item = new_sql_state.current_stack[-1][0]
            del new_sql_state.current_stack[-1]
            if finished_item == 'statement':
                break
            if new_sql_state.current_stack[-1][1][0] == finished_item:
                new_sql_state.current_stack[-1][1] = new_sql_state.current_stack[-1][1][1:]

            if finished_item == 'source_subq':
                # initialize the previous sql statelet in the subqueries stack and delete
                new_sql_state.tables_used = new_sql_state.subqueries_stack[-1].tables_used
                new_sql_state.tables_used_by_columns = new_sql_state.subqueries_stack[-1].tables_used_by_columns
                new_sql_state.derived_columns = new_sql_state.subqueries_stack[-1].derived_columns
                new_sql_state.derived_tables = new_sql_state.subqueries_stack[-1].derived_tables
                new_sql_state.used_agg = new_sql_state.subqueries_stack[-1].used_agg
                del new_sql_state.subqueries_stack[-1]

        return new_sql_state

    def get_valid_actions(self, valid_actions: dict):
        if not self.enabled:
            return valid_actions

        valid_actions_ids = []
        for key, items in valid_actions.items():
            valid_actions_ids += [(key, rule_id) for rule_id in valid_actions[key][2]]
        valid_actions_rules = [self.possible_actions[rule_id] for rule_type, rule_id in valid_actions_ids]

        actions_to_remove = {k: set() for k in valid_actions.keys()}

        current_clause = self._get_current_open_clause()

        # handle limitations relevant for "post-from" clauses
        if current_clause in ['where_clause', 'orderby_clause', 'groupby_clause']:

            tmp_tables_used = self.tables_used.copy()
            if not len(self.subqueries_stack) == 0:
                tmp_tables_used.update(self.subqueries_stack[-1].tables_used)

            tmp_derived_tables = self.derived_tables.copy()
            if not len(self.subqueries_stack) == 0:
                tmp_derived_tables.update(self.subqueries_stack[-1].derived_tables)

            tmp_derived_columns = self.derived_columns.copy()
            if not len(self.subqueries_stack) == 0:
                tmp_derived_columns.update(self.subqueries_stack[-1].derived_columns)

            # limit to already chosen in "from clause" (self.used_tables, self.derived_tables, self.derived_columns)
            for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                rule_type, rule_id = rule_id
                lhs, rhs = rule.split(' -> ')
                rhs_values = rhs.strip('[]').split(', ')
                # limit col_ref -> ["TABalias#.COL"] to tables in used_tables
                # and limit in case there are no derived tables or columns
                # in a string_set / value_set in in_expr it is possible to access all tables defined and selected all
                # the way up in the stack.. so the limitation doesn't apply.
                if lhs == 'col_ref':
                    if len(rhs_values) == 1:
                        rule_table = rhs_values[0].strip('"').split('.')[0]
                        if rule_table not in tmp_tables_used:
                            actions_to_remove[rule_type].add(rule_id)
                    else:
                        if rhs_values[0] == 'subq_alias' and len(tmp_derived_tables) == 0:
                            actions_to_remove[rule_type].add(rule_id)
                        elif rhs_values[0] == 'table_name' and len(tmp_derived_columns) == 0:
                            actions_to_remove[rule_type].add(rule_id)
                elif lhs =='table_name':
                    rule_table = rhs_values[0].strip('"')
                    if rule_table not in tmp_tables_used:
                        actions_to_remove[rule_type].add(rule_id)
                elif lhs == 'subq_alias':
                    rule_table = rhs_values[0].strip('"')
                    if rule_table not in tmp_derived_tables:
                        actions_to_remove[rule_type].add(rule_id)
                # if no columns were created and aliased, don't use col_alias in ordering_term
                elif lhs == 'ordering_term':
                    if len(tmp_derived_columns) == 0 \
                            and 'col_alias' in rhs_values:
                        actions_to_remove[rule_type].add(rule_id)


        # handle limitation for column selection and from clause
        elif current_clause in ['select_core']:
            for rule_id, rule in zip(valid_actions_ids, valid_actions_rules):
                rule_type, rule_id = rule_id
                lhs, rhs = rule.split(' -> ')
                rhs_values = rhs.strip('[]').split(', ')
                # selected columns from more tables than selected, must join
                # (there are two options: source -> [single_source]/[single_source, source],
                # must choose the second!)
                if lhs == 'source' \
                    and len(self.tables_used_by_columns.difference(self.tables_used)) > 1 \
                        and 'source' not in rhs_values:
                    actions_to_remove[rule_type].add(rule_id)
                # this is the last single source, if there are any unused but selected by columns tables,
                # we must choose single_source->source_table. else if there are unused but selected by columns
                # subqueries, choose single_source->source_subq. else, do whatever!
                elif lhs == 'single_source' and len(self.current_stack[-1][1]) == 1:
                    # the state of the stack is ['source',['single_source']]
                    if len(self.tables_used_by_columns - self.tables_used) == 1:
                        if list(self.tables_used_by_columns - self.tables_used)[0].startswith('DERIVED'):
                            # the unused table is a subq
                            if 'source_subq' not in rhs_values:
                                actions_to_remove[rule_type].add(rule_id)
                        else:
                            # the unused table is a schema aliased table
                            if 'source_subq' in rhs_values:
                                actions_to_remove[rule_type].add(rule_id)
                # the last source table should be from the unused but selected by columns set.
                elif lhs == 'table_name' \
                        and (self.tables_used_by_columns - self.tables_used) \
                        and len(self.current_stack[-3][1]) == 1 \
                        and self.current_stack[-3][1][0] == 'single_source':
                    # the state of the stack is ['source',['single_source']],['single_source',
                    # ['source_table']],['source_table',['table_name]]
                    candidate_table = rhs_values[0].strip('"')
                    if candidate_table not in self.tables_used_by_columns - self.tables_used:
                        # trying to select a single table but used other table(s) in columns
                        actions_to_remove[rule_type].add(rule_id)
                # the last source is a sub query.
                # since selecting the name is part of the sub query, we have to operate on the previous state
                # in the subqueries_stack!
                elif lhs == 'subq_alias' \
                    and self.current_stack[-1][0] == 'source_subq' \
                    and self.current_stack[-3][1] == 'single_source'\
                    and (self.subqueries_stack[-1].tables_used_by_columns - self.subqueries_stack[-1].tables_used):
                    # the state of the stack is
                    # ['source',['single_source']],['single_source',['source_subq']], ['source_subq', ['subq_alias']]
                    candidate_alias = rhs_values[0].strip('"')
                    if candidate_alias not in \
                            self.subqueries_stack[-1].tables_used_by_columns - self.subqueries_stack[-1].tables_used:
                        # trying to select a single alias but used other alias(es) in columns
                        actions_to_remove[rule_type].add(rule_id)

                if lhs == "table_name" or lhs == "subq_alias":
                    candidate_table = rhs_values[0].strip('"')
                    if candidate_table in self.tables_used:
                        actions_to_remove[rule_type].add(rule_id)


        new_valid_actions = {}
        new_global_actions = self._remove_actions(valid_actions, 'global',
                                                  actions_to_remove['global']) if 'global' in valid_actions else None
        new_linked_actions = self._remove_actions(valid_actions, 'linked',
                                                  actions_to_remove['linked']) if 'linked' in valid_actions else None

        if new_linked_actions is not None:
            new_valid_actions['linked'] = new_linked_actions
        if new_global_actions is not None:
            new_valid_actions['global'] = new_global_actions

        if new_valid_actions == {}:
            print('stop')
            print(f"\nValid Actions Rules:"
                  f"\n--------------------\n{valid_actions_rules}")
            print(f"\nAction History:"
                  f"\n---------------\n{self.action_history}")
            print(f"\nCurrent Stack:"
                  f"\n---------------\n{self.current_stack}")
            print(f"\nSelected Tables by columns:"
                  f"\n--------------------------\n{self.tables_used_by_columns}")
            print(f"\nUsed Tables:"
                  f"\n--------------\n{self.tables_used}")
            print(
                f"\nDerived_tables:"
                f"\n----------------------------------------\n{self.derived_tables}")
            print(f"\nDerived_columns:"
                  f"\n---------------------------------------\n{self.derived_columns}")
            print(f"\nUsed agg:"
                  f"\n---------------------------------------\n{self.used_agg}")
            if len(self.subqueries_stack) > 0:
                print("\nSAME FOR PREVIOUS STATE IN SUB-QUERIES STACK:\n")
                print(f"\nSelected Tables by columns:"
                      f"\n--------------------------\n{self.subqueries_stack[-1].tables_used_by_columns}")
                print(f"\nUsed Tables:"
                      f"\n--------------\n{self.subqueries_stack[-1].tables_used}")
                print(
                    f"\nDerived_tables:"
                    f"\n----------------------------------------\n{self.subqueries_stack[-1].derived_tables}")
                print(f"\nDerived_columns:"
                      f"\n---------------------------------------\n{self.subqueries_stack[-1].derived_columns}")
                print(f"\nCurrent clause:\n------------------------\n{current_clause}")
            return valid_actions
        return new_valid_actions
        # return valid_actions

    @staticmethod
    def _remove_actions(valid_actions, key, ids_to_remove):
        if len(ids_to_remove) == 0:
            return valid_actions[key]

        if len(ids_to_remove) == len(valid_actions[key][2]):
            return None

        current_ids = valid_actions[key][2]
        keep_ids = []
        keep_ids_loc = []

        for loc, rule_id in enumerate(current_ids):
            if rule_id not in ids_to_remove:
                keep_ids.append(rule_id)
                keep_ids_loc.append(loc)

        items = list(valid_actions[key])
        items[0] = items[0][keep_ids_loc]
        items[1] = items[1][keep_ids_loc]
        items[2] = keep_ids

        if len(items) >= 4:
            items[3] = items[3][keep_ids_loc]
        return tuple(items)

    def _get_current_open_clause(self):
        relevant_clauses = [
            'where_clause',
            'orderby_clause',
            'select_core',
            'groupby_clause',
        ]
        for rule in self.current_stack[::-1]:
            if rule[0] in relevant_clauses:
                return rule[0]

        return None
