from typing import List, Dict
import re
from overrides import overrides
import mysql.connector
from allennlp.training.metrics import Metric
from collections import defaultdict

import json
from pathlib import Path
import pandas as pd


@Metric.register("sql_validity")
class SqlValidity(Metric):
    """
    Use My Sql connector to connect to the database and test if the input program is executable
    """

    def __init__(self, mydatabase, 
                 localhost='localhost', 
                 port='3306', 
                 yourusername='user', 
                 yourpassword='password') -> None:
        """
        :param mydatabase: name of the database: atis/advising/geography/scholar
        :param localhost: host of MySql 
        :param port: port for MySql
        :param yourusername: username for MySql
        :param yourpassword: passowrd for MySql
        """
        try:
            self._conn = mysql.connector.connect(
                host=localhost,
                port=port,
                user=yourusername,
                passwd=yourpassword,
                database=mydatabase
            )
        except Exception:
            self._conn = None

        self._tables_capital = False
        self._columns_capital = False

        if self._conn is not None:
            self._cursor = self._conn.cursor()
            self.get_is_capital()

        self._correct_counts = 0.
        self._total_counts = 0.

        self._errors_dict = defaultdict(int)

    @overrides
    def reset(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.

    @overrides
    def __call__(self,
                 predictions: List[List[str]],
                 gold_targets: List[List[str]]) -> None:
        if self._conn is not None:
            self._total_counts += len(predictions)
            assert gold_targets[0][0] in ("SELECT", "select")
            self._cursor = self._conn.cursor()
            for predicted_tokens, gold_tokens in zip(predictions, gold_targets):
                sql_query = self.tokens_to_sql_query(predicted_tokens)
                gold_query = self.tokens_to_sql_query(gold_tokens)
                try:
                    self._cursor.execute(sql_query)
                    self._cursor.fetchall()
                    self._correct_counts += 1
                except mysql.connector.errors.DatabaseError as e:
                    self._correct_counts += 0
                    # if str(e).startswith('1054'): print(sql_query)
                    self._errors_dict[str(e)] += 1
                try:
                    self._cursor.execute(gold_query)
                    self._cursor.fetchall()
                except mysql.connector.errors.DatabaseError as e:
                    print(f'gold query is incorrect: {e}')
                    print(gold_query)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self._total_counts == 0:
            accuracy = 0.
        else:
            accuracy = self._correct_counts / self._total_counts

        if reset:
            self.reset()

        return {"sql_validity": accuracy, 'errors': self._errors_dict}

    def tokens_to_sql_query(self, tokens, aliases_v2=True):
        if not aliases_v2:
            tokens = [token if '@' not in token else token.split('@')[1] for token in tokens]
        sql = ' '.join(tokens)

        if aliases_v2:
            # replace table_plasceholder with the tables name!
            sql = re.sub(r" TABLE_PLACEHOLDER AS ([A-Z_]+)(alias[0-9]) ", r" \g<1> AS \g<1>\g<2> ", sql)

        # omit spaces between tables and aliases
        sql = re.sub(r" ([A-Z_]+)[\s]+(alias[0-9]) ", r" \g<1>\g<2> ", sql)

        # omit spaces between tables names and columns
        sql = re.sub(r" ([A-Z_]+alias[0-9])[\s]+\.[\s]+([A-Z_]+) ", r" \g<1>.\g<2> ", sql)
        sql = re.sub(r" ([A-Z_]+alias[0-9])[\s]+\.[\s]+(DERIVED_FIELDalias[0-9]) ", r" \g<1>.\g<2> ", sql)

        # omit spaces between functions and parenthesis
        sql = re.sub(r" (COUNT|SUM|MIN|MAX|AVG|ALL|LOWER|UPPER|CURDATE|YEAR)[\s]+\(", " \g<1>(", sql)

        # replace int values with 0
        sql = re.sub(" ([a-z_]+[0-9]) ", " 0 ", sql)

        if not self._tables_capital and not self._columns_capital:
            sql = sql.lower()
        else:
            sql = self.adapt_lower_case(sql)

        return sql

    def adapt_lower_case(self, sql):
        if not self._columns_capital:
            # find all columns
            sql = re.sub(r" ([A-Z_]+alias[0-9])\.([A-Z_]+) ", lambda pat: f" {pat.group(1)}.{pat.group(2).lower()} ", sql)
        if not self._tables_capital:
            sql = re.sub(r" ([A-Z_]+alias[0-9])\.([A-Z_]+|DERIVED_FIELDalias[0-9]) ",
                         lambda pat: f" {pat.group(1).lower()}.{pat.group(2)} ",
                         sql)
            sql = re.sub(r" ([A-Z_]+) AS ([A-Z_])alias([0-9]) ",
                         lambda pat: f" {pat.group(1).lower()} AS {pat.group(2).lower()}alias{pat.group(3)} ",
                         sql)
        return sql

    def get_is_capital(self):
        self._cursor.execute('SHOW TABLES ;')
        tables = self._cursor.fetchall()
        self._tables_capital = tables[0][0].isupper()

        self._cursor.execute(f'DESCRIBE {tables[0][0]} ;')
        columns_details = self._cursor.fetchall()
        column_name = columns_details[0][0]
        self._columns_capital = column_name.isupper()


def test_tokens_to_sql():
    SQL = ["""SELECT DISTINCT COUNT ( PAPER alias0 . PAPER@PAPERID )
                FROM KEYPHRASE AS KEYPHRASE alias0 ,
                     PAPER AS PAPER alias0 ,
                     PAPERKEYPHRASE AS PAPERKEYPHRASE alias0
                WHERE KEYPHRASE alias0 . KEYPHRASE@KEYPHRASENAME = 'keyphrasename0'
                  AND PAPERKEYPHRASE alias0 . PAPERKEYPHRASE@KEYPHRASEID = KEYPHRASE alias0 . KEYPHRASE@KEYPHRASEID
                  AND PAPER alias0 . PAPER@PAPERID = PAPERKEYPHRASE alias0 . PAPERKEYPHRASE@PAPERID ;""",
           """SELECT DISTINCT COUNT ( PAPER alias0 . PAPER@PAPERID ) , WRITES alias0 . WRITES@AUTHORID
                FROM PAPER AS PAPER alias0 ,
                     WRITES AS WRITES alias0 ,
                     AUTHOR AS AUTHOR alias0
                WHERE AUTHOR alias0 . AUTHOR@AUTHORNAME = 'author_name0'
                  AND WRITES alias0 . WRITES@AUTHORID = AUTHOR alias0 . AUTHOR@AUTHORID
                  AND WRITES alias0 . WRITES@PAPERID = PAPER alias0 . PAPER@PAPERID
                GROUP BY WRITES alias0 . WRITES@AUTHORID
                ORDER BY COUNT ( PAPER alias0 . PAPER@PAPERID ) DESC ;"""
           ]

    EXPECTED = ["""select distinct count( paperalias0.paperid )
                from keyphrase as keyphrasealias0 ,
                     paper as paperalias0 ,
                     paperkeyphrase as paperkeyphrasealias0
                where keyphrasealias0.keyphrasename = 'keyphrasename0'
                  and paperkeyphrasealias0.keyphraseid = keyphrasealias0.keyphraseid
                  and paperalias0.paperid = paperkeyphrasealias0.paperid ;""",
                """select distinct count( paperalias0.paperid ) , writesalias0.authorid
                            from paper as paperalias0 ,
                                 writes as writesalias0 ,
                                 author as authoralias0
                            where authoralias0.authorname = 'author_name0'
                              and writesalias0.authorid = authoralias0.authorid
                              and writesalias0.paperid = paperalias0.paperid
                            group by writesalias0.authorid
                            order by count( paperalias0.paperid ) desc ;"""
                ]

    metric = SqlValidity('127.0.0.1', 'inbaro', 'inbaro', 'scholar')
    for sql, exp_sql in zip(SQL, EXPECTED):
        valid_sql = metric.tokens_to_sql_query(sql.replace('\r\n', ' ').split())
        exp_valid = ' '.join(exp_sql.replace('\r\n', ' ').split())
        assert valid_sql == exp_valid, f"Expected {exp_valid}, got {valid_sql}"

        metric([sql.split()], [sql.split()])
        accuracy = metric.get_metric(reset=False)

        assert accuracy['sql_validity'] == 1.0, f"{accuracy['sql_validity']}"
        accuracy = metric.get_metric(reset=True)
        assert accuracy['sql_validity'] == 1.0, f"{accuracy['sql_validity']}"
        accuracy = metric.get_metric(reset=True)
        assert accuracy['sql_validity'] == 0.0, f"{accuracy['sql_validity']}"
