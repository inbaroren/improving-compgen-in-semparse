from typing import List

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


import re


@Tokenizer.register("whitespace")
class WhitespaceTokenizer(Tokenizer):
    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t) for t in text.split()]


@Tokenizer.register("standard")
class StandardTokenizer(Tokenizer):
    """
    splits columns and table names, splits quotes from values,
    splits parenthesis from sql tokens
    """

    @staticmethod
    def clean(text):
        """
        fix aliased tables and columns to be the same token, sql values as
        :param text:
        :return:
        """
        if not text.startswith('SELECT'):
            return text
        text = re.sub(r'([^ ]+alias[0-9])\.([^ ]+)', r'\g<1> . \g<2>', text)
        text = text.replace('"', ' " ').replace('(', ' ( ').replace(')', ' ) ')
        # I added this last replace() change after running seq2seq experiments
        # text = text.replace(',', ' ,')
        text = re.sub(r'[ ]+', ' ', text)
        return text

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t) for t in self.clean(text).split()]


@Tokenizer.register("geca_tokenizer")
class Text2SqlTokenizer(Tokenizer):
    """
    A ``Tokenizer`` that assumes you've already done your own tokenization somehow and have
    separated the tokens by spaces.  We just split the input string on whitespace and return the
    resulting list.
    Note that we use ``text.split()``, which means that the amount of whitespace between the
    tokens does not matter.  This will never result in spaces being included as tokens.
    """

    @staticmethod
    def clean(text):
        """
        Fix aliased tables and columns to be the same token, sql values separated from "
        Examples:
        >>clean("tab1alias0 . col1")
        >>tab1alias0.col1
        >>clean("\"value0\"")
        >>" value0 "
        >>clean("SELECT COUNT( * )")
        >>SELECT COUNT ( * )
        """
        if not text.startswith('SELECT'):
            return text
        matches = re.finditer(r'([^ ]+alias[0-9]) \. ([^ ]+)', text)
        for m in reversed(list(matches)):
            span = m.span(0)
            text = text[:span[0]] + m.group(1) + "." + m.group(2) + text[span[1]:]
        text = text.replace('"', ' " ').replace('(', ' ( ').replace(')', ' ) ')
        text = re.sub('[ ]+', ' ', text)
        return text

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        return [Token(t) for t in self.clean(text).split()]


if __name__ == '__main__':
    pairs  = [
        ('SELECT STATEalias0.POPULATION FROM STATE AS STATEalias0 WHERE STATEalias0.STATE_NAME = \"state_name0\" ;',
         'SELECT STATEalias0.POPULATION FROM STATE AS STATEalias0 WHERE STATEalias0.STATE_NAME = \" state_name0 \" ;',
         'SELECT STATEalias0 . POPULATION FROM STATE AS STATEalias0 WHERE STATEalias0 . STATE_NAME = \" state_name0 \" ;'),
        ('SELECT STATEalias0.STATE_NAME FROM STATE AS STATEalias0 WHERE STATEalias0.POPULATION = ( SELECT MIN( STATEalias1.POPULATION ) FROM STATE AS STATEalias1 ) ;',
         'SELECT STATEalias0.STATE_NAME FROM STATE AS STATEalias0 WHERE STATEalias0.POPULATION = ( SELECT MIN ( STATEalias1.POPULATION ) FROM STATE AS STATEalias1 ) ;',
         'SELECT STATEalias0 . STATE_NAME FROM STATE AS STATEalias0 WHERE STATEalias0 . POPULATION = ( SELECT MIN ( STATEalias1 . POPULATION ) FROM STATE AS STATEalias1 ) ;'),
        ('SELECT LAKEalias0.LAKE_NAME FROM LAKE AS LAKEalias0 WHERE LAKEalias0.AREA > 750 AND LAKEalias0.STATE_NAME = \"state_name0\" ;',
         'SELECT LAKEalias0.LAKE_NAME FROM LAKE AS LAKEalias0 WHERE LAKEalias0.AREA > 750 AND LAKEalias0.STATE_NAME = \" state_name0 \" ;',
         'SELECT LAKEalias0 . LAKE_NAME FROM LAKE AS LAKEalias0 WHERE LAKEalias0 . AREA > 750 AND LAKEalias0 . STATE_NAME = \" state_name0 \" ;')
        ]
    toki = Text2SqlTokenizer()
    toki2 = StandardTokenizer()
    for x, y, z in pairs:
        stan = " ".join([tok.text for tok in toki2.tokenize(x)])
        geca = " ".join([tok.text for tok in toki.tokenize(x)])
        assert y == geca, f"{x} failed to otkenize by geca {geca}"
        assert z == stan, f"{x} failed to tokenize by whitespace: {stan}"



