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


def main():
    base_path_s = '/media/disk1/inbaro/data/tmp_semparse/advising/'
    base_path = Path(base_path_s)
    for j in base_path.glob('*/no_join_*.json'):
        update_vars_dicts(j, j.parent / f'new_{j.name}')


if __name__ == '__main__':
  main()
  #   x = complete_vars_dict({
  #   "query-split": "train",
  #   "sentences": [
  #     {
  #       "question-split": "train",
  #       "text": "How many level0 -level classes are being offered in Fall and Winter term ?",
  #       "variables": {
  #         "department0": "",
  #         "level0": "300"
  #       }
  #     },
  #     {
  #       "question-split": "train",
  #       "text": "How many level0 -level classes are in the Fall or Winter term ?",
  #       "variables": {
  #         "department0": "",
  #         "level0": "300"
  #       }
  #     },
  #     {
  #       "question-split": "train",
  #       "text": "During the Fall and Winter term , how many level0 -level classes are you guys offering ?",
  #       "variables": { "level0": "300" }
  #     },
  #     {
  #       "question-split": "test",
  #       "text": "What is the number of level0 -level courses being offered in the Fall and Winter terms ?",
  #       "variables": { "level0": "200" }
  #     },
  #     {
  #       "question-split": "train",
  #       "text": "During the Fall and Winter term , how many level0 level classes are there ?",
  #       "variables": { "level0": "200" }
  #     },
  #     {
  #       "question-split": "train",
  #       "text": "During the Fall and Winter terms , how many level0 -level classes will be offered ?",
  #       "variables": { "level0": "500" }
  #     },
  #     {
  #       "question-split": "train",
  #       "text": "How many classes in the Fall and Winter term are level0 -level classes ?",
  #       "variables": { "level0": "500" }
  #     },
  #     {
  #       "question-split": "train",
  #       "text": "In the Fall and Winter term how many level0 -level classes are being offered ?",
  #       "variables": { "level0": "300" }
  #     },
  #     {
  #       "question-split": "train",
  #       "text": "What is the number of level0 -level classes available in Fall and Winter term ?",
  #       "variables": { "level0": "100" }
  #     },
  #     {
  #       "question-split": "train",
  #       "text": "In the Fall and Winter terms , how many level0 -level classes are being offered ?",
  #       "variables": { "level0": "200" }
  #     },
  #     {
  #       "question-split": "test",
  #       "text": "What is the number of level0 -level classes being offered in the Fall and Winter term ?",
  #       "variables": { "level0": "100" }
  #     }
  #   ],
  #   "sql": [ "SELECT COUNT( DISTINCT COURSEalias0.COURSE_ID , SEMESTERalias0.SEMESTER ) FROM COURSE AS COURSEalias0 , COURSE_OFFERING AS COURSE_OFFERINGalias0 , SEMESTER AS SEMESTERalias0 WHERE COURSEalias0.COURSE_ID = COURSE_OFFERINGalias0.COURSE_ID AND COURSEalias0.DEPARTMENT = \"department0\" AND COURSEalias0.NUMBER BETWEEN level0 AND level0 + 100 AND SEMESTERalias0.SEMESTER IN ( \"FA\" , \"WN\" ) AND SEMESTERalias0.SEMESTER_ID = COURSE_OFFERINGalias0.SEMESTER AND SEMESTERalias0.YEAR = 2016 ;" ],
  #   "variables": [
  #     {
  #       "example": "EECS",
  #       "location": "sql-only",
  #       "name": "department0",
  #       "type": "department"
  #     },
  #     {
  #       "example": "400",
  #       "location": "both",
  #       "name": "level0",
  #       "type": "number"
  #     }
  #   ]
  # })
  #   y = complete_vars_dict({"sql": [ "SELECT DISTINCT COURSEalias0.DEPARTMENT , COURSEalias0.NAME , COURSEalias0.NUMBER FROM COURSE AS COURSEalias0 WHERE ( COURSEalias0.DESCRIPTION LIKE \"%topic0%\" OR COURSEalias0.NAME LIKE \"%topic0%\" ) AND COURSEalias0.DEPARTMENT = \"EECS\" ;" ],
  #   "variables": [
  #     {
  #       "example": "networks",
  #       "location": "both",
  #       "name": "topic0",
  #       "type": "topic"
  #     }
  #   ]})
  #
  #   z = complete_vars_dict({"sql": [ "SELECT DISTINCT COURSEalias0.DEPARTMENT , COURSEalias0.NAME , COURSEalias0.NUMBER , INSTRUCTORalias0.NAME FROM COURSE AS COURSEalias0 , COURSE_OFFERING AS COURSE_OFFERINGalias0 , INSTRUCTOR AS INSTRUCTORalias0 , OFFERING_INSTRUCTOR AS OFFERING_INSTRUCTORalias0 WHERE COURSEalias0.COURSE_ID = COURSE_OFFERINGalias0.COURSE_ID AND COURSEalias0.NAME LIKE \"%topic0%\" AND INSTRUCTORalias0.NAME NOT LIKE \"%instructor0%\" AND OFFERING_INSTRUCTORalias0.INSTRUCTOR_ID = INSTRUCTORalias0.INSTRUCTOR_ID AND OFFERING_INSTRUCTORalias0.OFFERING_ID = COURSE_OFFERINGalias0.OFFERING_ID ;" ],
  #   "variables": [
  #     {
  #       "example": "algorithms",
  #       "location": "both",
  #       "name": "topic0",
  #       "type": "topic"
  #     },
  #     {
  #       "example": "stout",
  #       "location": "both",
  #       "name": "instructor0",
  #       "type": "instructor"
  #     }
  #   ]})
  #
  #   print(f"{x}\n\n\n{y}\n\n\n{z}")