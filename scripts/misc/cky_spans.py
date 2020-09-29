from typing import List, Tuple
import json

from text2sql.data.dataset_readers.dataset_utils.span_utils import EcpSpanExtractor
from text2sql.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from text2sql.data.preprocess.text2sql_canonicalizer import process_sentence as preprocess_text

# This scripts read the data json and adds to each sentence the spans by the constituency parser


def main(path, tokenizer, span_extractor):
    using_anonymized_for_parsing = 0
    with open(path, "r") as data_file:
        data = json.load(data_file)
    for i, ent in enumerate(data):
        for j, sent_info in enumerate(ent['sentences']):
            text = sent_info["text"]
            built_text = text
            for k, v in sent_info["variables"].items():
                built_text = built_text.replace(k, v)
            built_text = preprocess_text(built_text)
            text = preprocess_text(text)
            # fix `` to " back
            built_text = built_text.replace('``', '\"')
            text = text.replace('``', '\"')

            tokenized_built_text = [t.text for t in tokenizer.tokenize(built_text)]
            tokenized_text = [t.text for t in tokenizer.tokenize(text)]
            if len(tokenized_built_text) != len(tokenized_text):
                using_anonymized_for_parsing +=1
                tokenized_built_text = tokenized_text
            spans: List[Tuple[int, int]] = []
            for start, end in span_extractor.extract(tokenized_built_text):
                # indices should be changed since @START token is added to source_field!
                # no need to change the end index, so it will be inclusive
                spans.append((start, end))
            data[i]['sentences'][j]['constituency_parser_spans'] = spans

    with open(path,'w') as f:
        json.dump(data, f)
    print(path, using_anonymized_for_parsing)

