from typing import List, Tuple, Dict
from allennlp.predictors.predictor import Predictor
import allennlp_models.syntax.constituency_parser


class EcpSpanExtractor:
    """
    Extracts spans from strings using a pretrained elmo-constituency-parser
    """
    def __init__(self):
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")

    @staticmethod
    def _convert_indices(input_tokens: List[str], output_tokens: List[str], spans: List[Tuple[int, int]]):
        """
        converts the indices in the spans produced for output_tokens into the indices in input_tokens
        assumes there is a mapping from input_tokens to output_tokens (no overlapping tokens)
        if a span (i,j) in output_tokens:
            1. output_tokens[i] -> input_tokens[k,f] & output_tokens[j] -> input_tokens[s]
                then return(k,s)
            2. output_tokens[i] -> input_tokens[s] & output_tokens[j] -> input_tokens[k,f]
                then return(s,f)

        Example: The lab-free classes are taught by which professors ?
                input: ['The', 'lab-free', 'classes', 'are', 'taught', 'by', 'which', 'professors', '?']
                output: ['The', 'lab', '-', 'free', 'classes', 'are', 'taught', 'by', 'which', 'professors?']
                spans: [(0, 5), (5, 10), (10, 11), (0, 2), (2, 3), (3, 5), (5, 6), (6, 10), (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 10), (7, 8), (8, 10), (8, 9), (9, 10)]
        :param input_tokens: the tokens given to extract function
        :param output_tokens: the tokens by the constituency parser tokenizer
        :param spans - spans found in output_tokens
        :return: spans: List[Tuple[int, int]
        """

        # change to lower case in case one of the tokenizers changed it
        output_tokens = [t.lower() for t in output_tokens]
        input_tokens = [t.lower() for t in input_tokens]

        output2input: Dict[int, Tuple[int, int]] = {}
        input2output: Dict[int, Tuple[int, int]] = {}
        i = j = 0
        while i < len(output_tokens):
            while j < len(input_tokens):
                if output_tokens[i] == input_tokens[j]:
                    output2input[i] = (j, j)
                    input2output[j] = (i, i)
                    i += 1
                    j += 1
                    break
                elif output_tokens[i].startswith(input_tokens[j]):
                    s = j
                    while not output_tokens[i].endswith(input_tokens[j]):
                        input2output[j] = (i, i)
                        j += 1
                    input2output[j] = (i, i)
                    output2input[i] = (s, j)
                    i += 1
                    j += 1
                    break
                elif input_tokens[j].startswith(output_tokens[i]):
                    s = i
                    while not input_tokens[j].endswith(output_tokens[i]):
                        output2input[i] = (j, j)
                        i += 1
                    output2input[i] = (j, j)
                    input2output[j] = (s, i)
                    i += 1
                    j += 1
                    break
                else:
                    print(input_tokens, output_tokens, i, j, 'boo')

        new_spans: List[Tuple[int, int]] = []
        for s, e in spans:
            ss, se = output2input[s]
            es, ee = output2input[e - 1]  # the spans are exclusice
            new_spans.append((ss, ee + 1))

        return new_spans

    def extract(self, tokens: List[str]) -> List[Tuple[int, int]]:
        prediction = self.predictor.predict(
            sentence=' '.join(tokens)
        )

        # get the spans from the constituency parser prediction
        # go down the tree
        total_text_length = len(prediction['tokens'])
        spans: List[Tuple[int, int]] = []

        v = prediction['hierplane_tree']['root']
        v['span'] = (0, total_text_length)
        unvisited_stack = [v]

        while unvisited_stack:
            # pop next
            v = unvisited_stack[0]
            unvisited_stack = unvisited_stack[1:]
            # add spans to the children (all descendants words complete v's word)
            v_children = v.get('children', [])
            offset, _ = v['span']
            for i in range(len(v_children)):
                end_index = offset + len(v_children[i]['word'].split())
                assert v_children[i]['word'] == ' '.join(prediction['tokens'][offset:end_index])
                spans.append((offset, end_index))
                v_children[i]['span'] = spans[-1]
                offset = end_index
            # add children to the queue
            assert not (v['span'][1] != end_index and v_children)
            unvisited_stack.extend(v_children)
        spans = list(set(spans))
        return self._convert_indices(input_tokens=tokens,
                                     output_tokens=prediction['tokens'],
                                     spans=spans)
