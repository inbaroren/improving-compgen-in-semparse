from typing import Callable, List, Set

from allennlp_semparse import DomainLanguage, predicate
from allennlp_semparse.common.util import lisp_to_nested_expression


class QuestionSpan():
    def __init__(self):
        pass


class PassageSpan():
    pass


class Number():
    pass


# class Set():
#     pass


class SetProp():
    pass


class Bool():
    pass


class SetPassage2SetPassage():
    """<Set[Passage]:Set[Passage]> function type"""
    pass


class Number2Bool():
    """Number to bool function"""
    pass


class Passage2SetPassage():
    """Used in GROUP_count as a function that is needed"""
    pass


class Passage2SetNumber():
    """Used in GROUP_sum as a function that is needed"""
    pass


class Passage2Number():
    """To be used as mapping function in COMPARATIVE"""
    pass


class YearDiff():
    """ Represent year-difference between two dates. """
    pass


class QuestionNumber():
    """ Type to represent a number in the question. """
    pass


def project(string, passage_span):
    return string + " " + passage_span


from functools import partial

partial_project = partial(project, "a")


class DROPLanguage(DomainLanguage):
    """This language is tailored for DROP. The starting-step for this was the QDMRLanguage."""
    def __init__(self):
        super().__init__(start_types={Bool, Set[PassageSpan], Set[Number], YearDiff})

    @predicate
    def GET_QUESTION_SPAN(self) -> QuestionSpan:
        pass

    @predicate
    def GET_QUESTION_NUMBER(self) -> QuestionNumber:
        pass

    @predicate
    def SELECT(self, question_span: QuestionSpan) -> Set[PassageSpan]:
        pass

    @predicate
    def SELECT_NUM(self, s: Set[PassageSpan]) -> Set[Number]:
        pass

    @predicate
    def FILTER(self, s: Set[PassageSpan], question_span: QuestionSpan) -> Set[PassageSpan]:
        pass

    @predicate
    def PROJECT(self, question_span: QuestionSpan, s: Set[PassageSpan]) -> Set[PassageSpan]:
        pass

    @predicate
    def AGGREGATE_count(self, s: Set[PassageSpan]) -> Set[Number]:
        pass

    @predicate
    def COMPARISON_max(self, s1: Set[PassageSpan], s2: Set[PassageSpan]) -> Set[PassageSpan]:
        """This should implicitly be calling SELECT_NUM  / SELECT_DATE and doing comparisons."""
        pass

    @predicate
    def COMPARISON_min(self, s1: Set[PassageSpan], s2: Set[PassageSpan]) -> Set[PassageSpan]:
        """This should implicitly be calling SELECT_NUM  / SELECT_DATE and doing comparisons."""
        pass

    @predicate
    def COMPARISON_count_max(self, s1: Set[PassageSpan], s2: Set[PassageSpan]) -> Set[PassageSpan]:
        """This should implicitly be calling AGGREGATE_count on both sets and doing comparisons."""
        pass

    @predicate
    def COMPARISON_count_min(self, s1: Set[PassageSpan], s2: Set[PassageSpan]) -> Set[PassageSpan]:
        """This should implicitly be calling AGGREGATE_count on both sets and doing comparisons."""
        pass

    @predicate
    def COMPARISON_sum_max(self, s1: Set[PassageSpan], s2: Set[PassageSpan]) -> Set[PassageSpan]:
        """This should implicitly be calling AGGREGATE_sum on both sets and doing comparisons."""
        pass

    @predicate
    def COMPARISON_sum_min(self, s1: Set[PassageSpan], s2: Set[PassageSpan]) -> Set[PassageSpan]:
        """This should implicitly be calling AGGREGATE_sum on both sets and doing comparisons."""
        pass

    @predicate
    def UNION(self, s1: Set[PassageSpan], s2: Set[PassageSpan]) -> Set[PassageSpan]:
        pass

    @predicate
    def DISCARD(self, s1: Set[PassageSpan], s2: Set[PassageSpan]) -> Set[PassageSpan]:
        pass

    @predicate
    def INTERSECTION(self, s1: Set[PassageSpan], s2: Set[PassageSpan]) -> Set[PassageSpan]:
        pass

    @predicate
    def ARITHMETIC_sum(self, n1: Set[Number], n2: Set[Number]) -> Set[Number]:
        pass

    @predicate
    def ARITHMETIC_difference(self, n1: Set[Number], n2: Set[Number]) -> Set[Number]:
        pass

    @predicate
    def ARITHMETIC_divison(self, n1: Set[Number], n2: Set[Number]) -> Set[Number]:
        pass

    @predicate
    def ARITHMETIC_multiplication(self, n1: Set[Number], n2: Set[Number]) -> Set[Number]:
        pass

    @predicate
    def AGGREGATE_min(self, s: Set[PassageSpan]) -> Set[PassageSpan]:
        """Similar to find-min-num -- implicitly associate each element with a number and output the one with the min"""
        pass

    @predicate
    def AGGREGATE_max(self, s: Set[PassageSpan]) -> Set[PassageSpan]:
        """Similar to find-max-num -- implicitly associate each element with a number and output the one with the max"""
        pass

    @predicate
    def AGGREGATE_sum(self, s: Set[PassageSpan]) -> Set[Number]:
        """Assumes implicit association of each element with number.
        Even though output should be a single number, we're typing it as a Set
        """
        pass

    @predicate
    def AGGREGATE_avg(self, s: Set[PassageSpan]) -> Set[Number]:
        """Similar to AGGREGATE_sum"""
        pass

    @predicate
    def CONDITION(self, question_span: QuestionSpan) -> Number2Bool:
        pass

    @predicate
    def COMPARATIVE(self, s: Set[PassageSpan], passage2num: Passage2Number, condition: Number2Bool) -> Set[PassageSpan]:
        """Prunes a set of passage-spans by associating each span with a number and then applying a filter-condition
        on that number.

        Useful for "teams that scored more than 3 field goals" where the input is a set of "teams", which gets mapped to
        the number of field goals each kicked, and finally the output are teams for which this number is >3.

        Arguments:
        ----------
        s: Set[PassageSpan]
        passage2num: Is a function that maps a passage-span to a number. It can be as simple as SELECT-NUM, or could
        first project the span to a set of spans and output the cardinality of that set, among many other options
        condition: Is a function that maps a number to a boolean based on the condition mentioned in the question.
        """
        # TODO(nitish): The first argument should be Set[Entity] where Entity is represented as a set of passage spans,
        #  or in our world a passage_attention attending to multiple mentions
        #  Then passage2num can be take each one of these passage_attentions for execution
        pass

    @predicate
    def SUPERLATIVE_max(self, s: Set[PassageSpan], passage2num: Passage2Number) -> Set[PassageSpan]:
        """Given a set of passage spans and a function that maps a span to a number, output the span w/ the largest num
        Needs to implement 'max' within this predicate itself.
        """
        pass

    @predicate
    def SUPERLATIVE_min(self, s: Set[PassageSpan], passage2num: Passage2Number) -> Set[PassageSpan]:
        """Given a set of passage spans and a function that maps a span to a number, output the span w/ the largest num
        Needs to implement 'min' within this predicate itself.
        """
        pass

    @predicate
    def PARTIAL_GROUP_count(self, passage2setpassage: Passage2SetPassage) -> Passage2Number:
        """ This is a partial implementation of the GROUP_count function.

        Here, the passage2setpassage argument is already supplied; this would return a function <Passage:Number>
        Implementation of this function should be as simple as `return partial(GROUP_count, passage2setpassage)`
        """
        pass

    @predicate
    def GROUP_count(self, passage2setpassage: Passage2SetPassage, passage_span: PassageSpan) -> Number:
        """This function maps a passage-span to a set of passage-spans and outputs the size of the resultant set.

         On a high-level this function is `count(passage2setpassage(passasge_span))`

         The arguments are ordered in this manner since PARTIAL_GROUP_count (which would be needed in COMPARATIVE)
         supplies passage2setpassage argument and python's partial function fills arguments in a left-to-right manner.

        Arguments:
        ----------
        passage2setpassage: A function that maps a passage-span to a set of passage-spans
        passage_span: PassageSpan
        """
        pass

    @predicate
    def PARTIAL_GROUP_sum(self, passage2setnumber: Passage2SetNumber) -> Passage2Number:
        """ This is a partial implementation of the GROUP_sum function.

        Here, the passage2setnumber argument for GROUP_sum is already supplied; hence this would return a
        function <Passage:Number> -- that could just take a passage-span and map it to a number.
        Behind the scenes it would call the already supplied passage2setnumber function.
        Implementation of this function should be as simple as `return partial(GROUP_sum, passage2setnumber)`
        """
        pass

    @predicate
    def GROUP_sum(self, passage2setnumber: Passage2SetNumber, passage_span: PassageSpan) -> Number:
        """This function maps a passage-span to a set of numbers and outputs their sum.
         On a high-level this function is `sum(passage2setnumber(passasge_span))`

         The arguments are ordered in this manner since PARTIAL_GROUP_sum (which would be needed in COMPARATIVE)
         supplies passage2setnumber argument and python's partial function fills arguments in a left-to-right manner.

        Arguments:
        ----------
        passage2setnumber: A function that maps a passage-span to a set of numbers
        passage_span: PassageSpan
        """
        pass

    @predicate
    def PARTIAL_PROJECT(self, question_span: QuestionSpan) -> Passage2SetPassage:
        """Partial(Project(string)) -- String arg to project is supplied.
        This partial can now take a span and map to Passage2SetPassage"""
        pass

    @predicate
    def PARTIAL_SELECT_NUM(self, question_span: QuestionSpan) -> Passage2SetNumber:
        """This function can wrap the input span to Set[Passage], run project with ques_span and return SELECT_NUM"""
        pass

    @predicate
    def SELECT_NUM_SPAN(self) -> Passage2SetNumber:
        """This function can wrap the input span to Set[Passage] and run SELECT_NUM"""
        pass

    @predicate
    def BOOLEAN(self, s: Set[PassageSpan], string: QuestionSpan) -> Bool:
        # TODO(nitish): Fix this
        pass

    @predicate
    def COMPARISON_true(self, b1: Bool, b2: Bool) -> Bool:
        # TODO(nitish): Fix this
        pass

    @predicate
    def Year_Diff_Single_Event(self, passage_span: Set[PassageSpan]) -> YearDiff:
        pass

    @predicate
    def Year_Diff_Two_Events(self, passage_span_1: Set[PassageSpan], passage_span_2: Set[PassageSpan]) -> YearDiff:
        pass

    @predicate
    def FILTER_NUM_EQ(self, passage_span: Set[PassageSpan], ques_number: QuestionNumber) -> Set[PassageSpan]:
        pass

    @predicate
    def FILTER_NUM_LT(self, passage_span: Set[PassageSpan], ques_number: QuestionNumber) -> Set[PassageSpan]:
        pass

    @predicate
    def FILTER_NUM_GT(self, passage_span: Set[PassageSpan], ques_number: QuestionNumber) -> Set[PassageSpan]:
        pass

    @predicate
    def FILTER_NUM_LT_EQ(self, passage_span: Set[PassageSpan], ques_number: QuestionNumber) -> Set[PassageSpan]:
        pass

    @predicate
    def FILTER_NUM_GT_EQ(self, passage_span: Set[PassageSpan], ques_number: QuestionNumber) -> Set[PassageSpan]:
        pass

    @predicate
    def COMPARISON_DATE_max(self, s1: Set[PassageSpan], s2: Set[PassageSpan]) -> Set[PassageSpan]:
        """This should implicitly be calling SELECT_DATE and doing comparisons."""
        pass

    @predicate
    def COMPARISON_DATE_min(self, s1: Set[PassageSpan], s2: Set[PassageSpan]) -> Set[PassageSpan]:
        """This should implicitly be calling SELECT_DATE and doing comparisons."""
        pass

    @predicate
    def PARTIAL_SELECT_SINGLE_NUM(self, question_span: QuestionSpan) -> Passage2Number:
        """This predicate, given a question_span returns a function that can map a passage span to a number.
        Can be used to map passage_span to a number for comparative.
        question_span could be "height of", "yards of", etc. for example
        """
        pass

    @predicate
    def SELECT_IMPLICIT_NUM(self) -> Set[Number]:
        """This predicate can select an implicit number from a predefined global set. E.g. 100 for percent of not X"""
        pass


def main():
    def nested_expression_to_lisp(nested_expression):
        if isinstance(nested_expression, str):
            return nested_expression

        elif isinstance(nested_expression, List):
            lisp_expressions = [nested_expression_to_lisp(x) for x in nested_expression]
            return "(" + " ".join(lisp_expressions) + ")"
        else:
            raise NotImplementedError

    drop_language = DROPLanguage()
    DROP_predicates = sorted(list(drop_language._functions.keys()))
    print(DROP_predicates)


    print("Non termincal prods")
    non_terminal_prods = drop_language.get_nonterminal_productions()
    print("\n".join(non_terminal_prods))

    print("\n")
    print("All possible prods")
    all_possible_prods = drop_language.all_possible_productions()
    print("\n".join(all_possible_prods))

    exit()


    # program = "(COMPARATIVE (SELECT GET_QUESTION_SPAN) (PARTIAL_GROUP_count (PARTIAL_PROJECT GET_QUESTION_SPAN)) (CONDITION GET_QUESTION_SPAN))"
    program = "(COMPARATIVE (SELECT GET_QUESTION_SPAN) (PSSA GET_QUESTION_SPAN) (CONDITION GET_QUESTION_SPAN))"
    nested_expression = lisp_to_nested_expression(program)
    action_seq = drop_language.logical_form_to_action_sequence(program)
    print(nested_expression)
    print(action_seq)


    nested_expression = ['COMPARATIVE',
                             ['SELECT', 'nationalities registered in Bilbao'],
                             ['PARTIAL_GROUP_count', ['PARTIAL_PROJECT', 'people of #REF']],
                             ['CONDITION', 'is higher than 10']
                        ]





if __name__=='__main__':
    main()



