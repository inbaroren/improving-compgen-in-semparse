import os
import re
import json
from pathlib import Path
import copy


def main():
    base_path_s = '/media/disk1/inbaro/data/tmp_semparse/advising/'
    base_path = Path(base_path_s)
    for j in base_path.glob('*/*.json'):
        remove_join_from_file(j, j.parent / f'no_join_{j.name}')


def remove_join_from_file(path, new_path):
    with open(path) as f:
        data = json.load(f)
    new_data = []
    count_orig_sql = 0
    count_sql = 0
    for entry in data:
        new_entry = copy.deepcopy(entry)
        new_sqls = []
        for sql in entry['sql']:
            count_orig_sql += 1
            new_sqls.append(remove_join_string(sql))
        new_entry['sql'] = new_sqls
        new_data.append(new_entry)
    for entry in new_data:
        for sql in entry['sql']:
            count_sql += 1
    assert count_sql == count_orig_sql
    with open(new_path, 'w+') as f:
        json.dump(new_data, f)


def remove_join_string(sql_string):
    new_sql_string = ""
    max_explored_index = 0
    for from_with_join_clause in re.finditer(r"FROM [A-Z_0-9]+ AS [A-Z_0-9]+alias[0-9](?: INNER JOIN [A-Z_0-9]+ AS "
                                             r"[A-Z_0-9]+alias[0-9] ON [A-Z_0-9]+alias[0-9].[A-Z_0-9]+ = "
                                             r"[A-Z_0-9]+alias[0-9].[A-Z_0-9]+)+(?: WHERE)*", sql_string):
        from_cols = []
        where_conds = []
        # add to the string all the parts between from joins
        new_sql_string += sql_string[max_explored_index:from_with_join_clause.span()[0]]
        # update max_explored_index
        max_explored_index = from_with_join_clause.span()[1]
        start_match = re.search(r"FROM ([A-Z_0-9]+ AS [A-Z_0-9]+alias[0-9]) INNER JOIN "
                                r"([A-Z_0-9]+ AS [A-Z_0-9]+alias[0-9]) ON "
                                r"([A-Z_0-9]+alias[0-9].[A-Z_0-9]+ = [A-Z_0-9]+alias[0-9].[A-Z_0-9]+)",
                                from_with_join_clause.group(0))
        from_cols.append(start_match.group(1))
        from_cols.append(start_match.group(2))
        where_conds.append(start_match.group(3))
        for m in re.finditer(r" INNER JOIN ([A-Z_0-9]+ AS [A-Z_0-9]+alias[0-9]) ON "
                             r"([A-Z_0-9]+alias[0-9].[A-Z_0-9]+ = [A-Z_0-9]+alias[0-9].[A-Z_0-9]+)",
                             from_with_join_clause.group(0)[start_match.span()[1]:]):
            from_cols.append(m.group(1))
            where_conds.append(m.group(2))
        # add the new FROM clause without the join
        # add the "ON" conditions as part of the WHERE clause
        new_sql_string += f"FROM {' , '.join(from_cols)}"
        if len(where_conds) > 0:
            new_sql_string += f" WHERE {' AND '.join(where_conds)}"
        if "WHERE" in from_with_join_clause.group(0):
            new_sql_string += " AND"

    new_sql_string += sql_string[max_explored_index:]
    return new_sql_string


def test_remove_join():
    with open('/media/disk1/inbaro/data/adv_join_sqls.json') as f:
        data = json.load(f)
    no_on=0
    for sql in data:
        if not " ON " in sql:
            no_on += 1
            continue
        n_sql = remove_join_string(sql)
        assert len(re.findall("WHERE", n_sql)) == len(re.findall("WHERE", sql)), f"too many WHERE. \n{sql}, \n{n_sql}"
        assert len(re.findall("FROM", n_sql)) == len(re.findall("FROM", sql)), f"too many FROM. \n{sql}, \n{n_sql}"
        all_names_orig = set(re.findall(r"[A-Z_0-9]+ AS [A-Z_0-9]+alias[0-9]", sql))
        all_names_n = set(re.findall(r"[A-Z_0-9]+ AS [A-Z_0-9]+alias[0-9]", n_sql))
        assert all_names_orig.intersection(all_names_n) == all_names_n, f"missed columns. " \
            f"orig: {all_names_orig}, n: {all_names_n}"
        assert n_sql.endswith(' ;'), "no ; at the end"
        assert len(re.findall("INNER JOIN", n_sql)) == 0, "still inner join in new query..."

    print(no_on)


if __name__ == '__main__':
    main()
