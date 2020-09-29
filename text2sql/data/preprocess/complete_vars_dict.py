# util functions used to complete missing details in the text2sql json files
# before training

import re
import json
from pathlib import Path
import copy
from collections import Counter, defaultdict


CASES_STATS = defaultdict(int)


def all_names(entry):
    return [x['name'] for x in entry['variables']]


def complete_vars_dict(entry):
    '''
    Cases:
        1. just a string, so all a-zA-Z between \"\" or \"%%\"
        2. a year, .YEAR = [0-9]{4}
        3. BETWEEN level0 AND level0 + 100 -> switch level0 + 100 to level1
        4. any number...
        5. hour like 10:00:00
        6. just level0 + 100...
    '''

    changes = False

    for i, sql in enumerate(entry['sql']):

        for m in re.finditer(r"\"%?([a-zA-Z0-9\s\-]+)%?\"", sql): # case 1
            m_name = m.group(1) if m.group(1).replace(" ","") == m.group(1) else m.group(1).replace(" ","_")
            if m_name in all_names(entry):
                continue
            entry['variables'].append({
                'name': m_name,
                'example': m_name,
                'type': 'semester',
                'location': 'sql-only'
            })
            changes = True
            CASES_STATS['case1'] += 1

        for m in re.finditer(r"\.YEAR = ([0-9]{4})", sql): # case 2
            m_name = m.group(1)
            if m_name in all_names(entry):
                continue
            entry['variables'].append({
                'name': m_name,
                'example': m_name,
                'type': 'year',
                'location': 'sql-only'
            })
            changes = True
            CASES_STATS['case2'] += 1

        for m in re.finditer(r"alias[0-9]\.([A-Z_0-9]+) (?:=|!=|<|>|>=|<=|<>) ([0-9]+)", sql):  # case 4
            m_name = m.group(2)
            if m_name in all_names(entry):
                continue
            entry['variables'].append({
                'name': m_name,
                'example': m_name,
                'type': m.group(1).lower(),
                'location': 'sql-only'
            })
            changes = True
            CASES_STATS['case4'] += 1

        gen = re.finditer(r"BETWEEN ([a-z]+[0-9]) AND (([a-z]+[0-9]) [+-] ([0-9]+))", sql)
        while re.findall(r"BETWEEN ([a-z]+[0-9]) AND (([a-z]+[0-9]) [+-] ([0-9]+))", sql):
            for m in gen: # case 3
                if m.group(2) in [x['example'] for x in entry['variables']]:
                    m_name = [x['name'] for x in entry['variables'] if x['example'] == m.group(2)][0]
                else:
                    times_counter = len([x['name'] for x in entry['variables'] if x['name'].startswith(m.group(3)[:-1])])
                    m_name = f"{m.group(3)[:-1]}{times_counter}"
                    entry['variables'].append({
                        'name': m_name,
                        'example': m.group(1),
                        'type': 'number',
                        'location': 'sql-only'
                    })
                sql = sql.replace(m.group(2), m_name)
                entry['sql'][i] = sql
                gen = re.finditer(r"BETWEEN ([a-z]+[0-9]) AND (([a-z]+[0-9]) [+-] ([0-9]+))", sql)

                changes = True
                CASES_STATS['case3'] += 1

        gen = re.finditer(r" [<>=]{1,2} (([a-z]+[0-9]) [+-] ([0-9]+)) ", sql)
        while re.findall(r" [<>=]{1,2} (([a-z]+[0-9]) [+-] ([0-9]+)) ", sql):
            for m in gen: # case 6
                if m.group(1) in [x['example'] for x in entry['variables']]:
                    m_name = [x['name'] for x in entry['variables'] if x['example'] == m.group(1)][0]
                else:
                    times_counter = len([x['name'] for x in entry['variables'] if x['name'].startswith(m.group(2)[:-1])])
                    m_name = f"{m.group(2)[:-1]}{times_counter}"
                    entry['variables'].append({
                        'name': m_name,
                        'example': m.group(1),
                        'type': 'number',
                        'location': 'sql-only'
                    })
                sql = sql.replace(m.group(1), m_name)
                entry['sql'][i] = sql
                gen = re.finditer(r" [<>=]{1,2} (([a-z]+[0-9]) [+-] ([0-9]+)) ", sql)

                changes = True
                CASES_STATS['case6'] += 1

        gen = re.finditer(r" [<>=]+ \"(\d\d\:\d\d(?:\:\d\d)*)\" ", sql)
        while re.findall(r" [<>=]+ \"(\d\d\:\d\d(?:\:\d\d)*)\" ", sql):
            for m in gen: # case 5
                if m.group(1) in [x['example'] for x in entry['variables']]:
                    m_name = [x['name'] for x in entry['variables'] if x['example'] == m.group(1)][0]
                else:
                    times_counter = len([x['name'] for x in entry['variables'] if x['name'].startswith('time')])
                    m_name = f"time{times_counter}"
                    entry['variables'].append({
                        'name': m_name,
                        'example': m.group(1),
                        'type': 'number',
                        'location': 'sql-only'
                    })
                sql = sql.replace(m.group(1), m_name)
                entry['sql'][i] = sql
                gen = re.finditer(r" [<>=]+ \"(\d\d\:\d\d(?:\:\d\d)*)\" ", sql)

                CASES_STATS['case5'] += 1

    return entry, changes


def update_vars_dicts(path, new_path):
    with open(path) as f:
        data = json.load(f)
    new_data = []
    count_orig_sql = 0
    count_sql = 0
    for i in range(len(data)):
        count_orig_sql += len(data[i]['sql'])
        entry, changes = complete_vars_dict(data[i])
        new_data.append(entry)
    for entry in new_data:
        count_sql += len(entry['sql'])
    assert count_sql == count_orig_sql, f"{count_sql} != {count_orig_sql}"
    print(f"File name: {path}\nTotal entries: {len(data)}\nTotal number of sql queries: {count_sql}\nCases stats: {CASES_STATS}")
    with open(new_path, 'w+') as f:
        json.dump(new_data, f)
