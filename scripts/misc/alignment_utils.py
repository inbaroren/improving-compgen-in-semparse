"""
Fast align utils for predicting unsupervised alignments over the lexical source domains
"""
import os
import subprocess
from pathlib import Path
import re
import json
import numpy as np

from text2sql.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer, StandardTokenizer
from text2sql.data.dataset_readers.grammar_based_text2sql_v3 import GrammarBasedText2SqlDatasetReader

MAIN = Path('/datainbaro2/text2sql/parsers_models/allennlp_text2sql')
DATA_TRAIN_PATH = MAIN/Path('data/sql data')
DATA_IN_PATH = MAIN/Path('data/alignments/in_short')
DATA_IN_PATH.mkdir(exist_ok=True)
DATA_OUT_PATH = MAIN/Path('data/alignments/out_short')
DATA_OUT_PATH.mkdir(exist_ok=True)
FAST_ALIGN_PATH = Path('/datainbaro2/text2sql/fast_align/build/fast_align')
DELIMITER = ' ||| '

SQL_TERMS = ['SELECT', 'FROM', 'WHERE', 'LIMIT', 'NULL', 'AS']


def is_global_rule(production_rule_str):
    lhs, rhs = production_rule_str.strip().split('->')
    rhs_values = rhs.strip('[] ').split(',')
    rhs_value = rhs_values[0].strip()
    if len(rhs_values) == 1 and rhs_value[0] == '"' and (
            re.sub('\"[A-Z_]+alias[0-9]\.[A-Z_]+\"', '', rhs_value) == '' or \
            '' == re.sub('\"[A-Z_]+alias[0-9]\"', '', rhs_value) or \
            re.sub('\"\'[a-z_]+[0-9]\'\"', '', rhs_value) == ''):
        return True
    return False


def clean(schema_dependent_rule):
    return schema_dependent_rule.replace(' ', '').split('->')[1].strip('[] ').strip('\"')


def shorten_sql_string(s):
    s = re.sub(r'[A-Z_]+ AS [A-Z_]+alias[0-9]', '', s)
    for term in SQL_TERMS:
        s = s.replace(term, '')
    for p in (",", ".", ";"):
        s = s.replace(p, " ")
    s = re.sub(r'[ ]+', ' ', s)
    return s


def shorten_sql_tokens(s_toks):
    """
    filter tokens, return the map
    :param s_toks: tokenized sql string
    :return: filtered tokens, mapping to s_toks
    """
    new_toks = []
    mapping = []
    i = 0
    while i < len(s_toks) - 1: # remove all ;
        # remove all "FROM" clauses with commas
        if i < len(s_toks) - 3 \
                and re.sub(r'[A-Z_]+ AS [A-Z_]+alias[0-9]', '',  ' '.join(s_toks[i:i+3])) == '':
            i += 3
        # remove SELECT FROM WHERE
        elif s_toks[i] in ('SELECT', 'FROM', ',', '"'):
            i += 1
        # concatenate table . column
        elif i < len(s_toks) - 3 \
                and re.sub(r'[A-Z_]+alias[0-9] \. [A-Z_]+', '',  ' '.join(s_toks[i:i+3])) == '':
            new_toks.extend([''.join(s_toks[i:i+3])])
            mapping.append((i, i+2))
            i += 3
        # concatenate GROUPBY AND ORDERBY
        elif i < len(s_toks) - 2 \
                and ' '.join(s_toks[i:i + 2]) in ("GROUP BY", "ORDER BY"):
            new_toks.extend([''.join(s_toks[i:i + 2])])
            mapping.append((i, i + 1))
            i += 2
        # remove LIMIT
        elif i < len(s_toks) - 3 \
                and re.sub(r'LIMIT [0-9]', '',  ' '.join(s_toks[i:i+2])) == '':
            i += 2
        else:
            new_toks.append(s_toks[i])
            mapping.append((i,i))
            i += 1
    assert all([isinstance(e, tuple) for e in mapping]), f"non tuple in the mapping file! {mapping}"
    return new_toks, mapping


def create_align_input(align_file_path, train_file_path, train_file_name, mapping_file=None, filter=lambda x: x, reverse=False):
    """
    Create fast_align format
    :param align_file_path:  path to save the output file (input to aligner)
    :param train_file_path: path to load the inupt, the train data
    :param train_file_name: name of the both the input trin file and output file
    :param schema_path: path to the dataset schema
    :param mapping_file: if a filter function is given, this file saves the mapping from the tokenized original
            text to the tokenized filtered text
    :param filter: a function that filters a sql string, applied before tokenization. The tokenized filtered string
            will be added to teh align file
    """
    text_tokenizer = WhitespaceTokenizer()
    sql_tokenizer = StandardTokenizer()
    dataset_path = train_file_path / (train_file_name+'.json')
    with open(dataset_path) as f:
        data = json.load(f)
    all_data_raw = []
    all_mappings = []
    for inst in data:
        # clean the query once
        y_orig = [t.text for t in sql_tokenizer.tokenize(inst['sql'][0])]
        y, mapping = filter(y_orig)
        all_mappings.append(mapping)
        y = ' '.join(y)
        for sent_info in inst["sentences"]:
            x = sent_info["text"]
            x = ' '.join([t.text for t in text_tokenizer.tokenize(x)])
            if not reverse:
                all_data_raw.append(x + DELIMITER + y + '\n')
            else:
                all_data_raw.append(y + DELIMITER + x + '\n')
    print('loaded {} train examples from file {}'.format(len(all_data_raw), input))
    input_align_file = train_file_name + '.align'
    with open(os.path.join(align_file_path, input_align_file), 'w') as f:
        for example in all_data_raw:
            f.write(example)
    if mapping_file is not None:
        with open(os.path.join(align_file_path, mapping_file), 'w') as f:
            json.dump(all_mappings, f)

    return input_align_file


def train_alignment(input_align_path, output_aligned_path, file_name):
    output_align_file = file_name +'.alignment'
    input_align_file = (file_name + '.align')

    input_align_file_path = str(input_align_path / input_align_file)
    output_aligned_file_path = str(output_aligned_path / output_align_file)

    command = str(FAST_ALIGN_PATH) + ' -i ' + input_align_file_path + ' -v  > ' + output_aligned_file_path
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)

    return output_align_file, ""


def load_alignment(p):
    outputs = []
    for row in open(p):  # read alignments
        alignments = [(int(pair.split('-')[0]), int(pair.split('-')[1])) for pair in row.rstrip().split(' ')]
        outputs.append(alignments)
    return outputs


def load_cond_prob(output_path):
    mapping = dict()
    for row in open(output_path):  # read alignments
        items = row.rstrip().split('\t')
        if output_path.endswith('for'):
            lf = items[1]
            nl = items[0]
        else:
            lf = items[0]
            nl = items[1]
        score = items[2]

        if lf not in mapping:
            mapping[lf] = dict()
        mapping[lf][nl] = float(score)
    return mapping


def load_cond_probs(paths):
    mappings = {}
    for p in paths:
        mappings[p] = load_cond_prob(p)
    return mappings


def inspect_alignment_file(input_path, output_path, inspection_file_path, only_entities=True, action_maps=None):
    inputs = []
    outputs = []

    for row in open(input_path):   # read training data
        x, y = row.rstrip().split(DELIMITER)
        inputs.append((x, y))

    for row in open(output_path):   # read alignments
        outputs.append(row.rstrip().split(' '))

    current_action_map = None
    with open(inspection_file_path, 'w') as insp_f:
        for i, example in enumerate(inputs):
            if action_maps is not None:
                current_action_map = action_maps[i]
            lines = inspect_alignment(example[0], example[1], outputs[i], only_entities, current_action_map)
            insp_f.writelines(lines)


def inspect_alignment(x, y, alignment, only_entities=True, action_map=None):
    x_toks = x.split()
    y_toks = y.split()
    if not only_entities and action_map is not None:
        y_toks = [action_map[int(t)].replace(' ','') for t in y_toks]

    align_map = {}
    for a in alignment:
        x_ind = a.split('-')[0]
        y_ind = a.split('-')[1]
        align_map[int(y_ind)] = int(x_ind)

    lines_to_log = [f"{x}\n", f"{y}\n"]

    for i, y_tok in enumerate(y_toks):
        # if not only_entities:
        #     if not is_global_rule(y_tok):
        #         continue
        align = align_map.get(i)
        if align is None:
            x_tok = 'NO_ALIGN'
            x_ind = '_'
        else:
            x_tok = x_toks[align]
            x_ind = align
        lines_to_log.append(str(i) + '\t' + y_tok + '\t-\t' + str(x_ind) + '\t' + x_tok+'\n')
    return lines_to_log


def preprocess_alignment_file(input_path, output_path, processed_file_path):
    inputs = []
    outputs = []

    for row in open(input_path):   # read training data
        x, y = row.rstrip().split(DELIMITER)
        inputs.append((x, y))

    for row in open(output_path):   # read alignments
        outputs.append(row.rstrip().split(' '))

    processed_output = []
    for i, example in enumerate(inputs):
        lines = preprocess_alignment_to_print(example[0], example[1], outputs[i])
        processed_output.append(lines)

    with open(processed_file_path, 'w') as insp_f:
        json.dump(processed_output, insp_f)


def preprocess_alignment_to_print(x, y, alignment):
    x_toks = x.split()
    y_toks = y.split()

    align_map = {}
    for a in alignment:
        x_ind = a.split('-')[0]
        y_ind = a.split('-')[1]
        align_map[int(y_ind)] = int(x_ind)

    lines_to_log = {"sentence": x_toks,
                    "sql": y_toks,
                    "align_map": align_map,
                    "alignment": []}

    for i, y_tok in enumerate(y_toks):
        # if not only_entities:
        #     if not is_global_rule(y_tok):
        #         continue
        align = align_map.get(i)
        if align is None:
            x_tok = 'NO_ALIGN'
            x_ind = '_'
        else:
            x_tok = x_toks[align]
            x_ind = align
        lines_to_log["alignment"].append((y_tok, x_tok))
    return lines_to_log


def ignore_alignment(y_tok):
    if len(y_tok) < 2 or y_tok in SQL_TERMS:
        return True
    else:
        return False


def preprocess_alignment(x, y, alignment):
    x_toks = x.split()
    y_toks = y.split()
    result = []
    align_map = {}

    for a in alignment:
        x_ind = a.split('-')[0]
        y_ind = a.split('-')[1]
        align_map[int(y_ind)] = int(x_ind)

    for i, y_tok in enumerate(y_toks):
        # if not only_entities:
        #     if not is_global_rule(y_tok):
        #         continue
        align = align_map.get(i)
        if align is None or ignore_alignment(y_tok):
            x_tok = 'NO_ALIGN'
        else:
            x_tok = x_toks[align]
        result.append(x_tok)
    return result


def update_alignments_in_file(input_to_align_path, alignment_file_path, orig_data_json_path, new_data_json_path):
    """
    Reads the alignment_file_path and adds the alignmnets to the original data json file
    The alignments in alignment_file_path are of form input_token_index-output_token_index
    Each alignment is added to the correspoding "senetence" obeject under the field "alignment"
    (the alignment of the first SQL query in the entry "sql" list to this sentence).
    The alignment is converted to the input tokens. in case no input token was aligned at step t,
    a special token is added
    """
    inputs = []
    outputs = []

    for row in open(input_to_align_path):  # read training data
        x, y = row.rstrip().split(DELIMITER)
        inputs.append((x, y))

    for row in open(alignment_file_path):  # read alignments
        outputs.append(row.rstrip().split(' '))

    processed_alignments = []
    for i, example in enumerate(inputs):
        result = preprocess_alignment(example[0], example[1], outputs[i])
        processed_alignments.append(result)

    with open(orig_data_json_path) as f:
        input_data = json.load(f)
    counter = 0
    for i, ent in enumerate(input_data):
        for j, sent_info in enumerate(ent["sentences"]):
            input_data[i]["sentences"][j]["alignment"] = ' '.join(processed_alignments[counter])
            counter += 1
    with open(new_data_json_path, 'w') as f:
        json.dump(input_data, f)


def fast_align_text2sql(datasets, splits):
    # for dataset in ['test', 'advising', 'atis', 'geography', 'scholar']:
    # for dataset in ['test']:
    # for dataset in ['scholar', 'atis']:
    for dataset in datasets:
        dataset_train_path = DATA_TRAIN_PATH / dataset
        dataset_align_out = DATA_IN_PATH / dataset
        dataset_align_out.mkdir(exist_ok=True)
        dataset_aligned_out = DATA_OUT_PATH / dataset
        dataset_aligned_out.mkdir(exist_ok=True)
        # for split in ["schema_full_split", "new_question_split"]:
        for split in splits:
            for train_file_name in ['train', 'final_dev']:
                # fix the names of the files in case it is advising :0
                if dataset == 'advising':
                    if train_file_name == 'train':
                        train_file_name = 'new_no_join_train'
                    else:
                        train_file_name = 'final_new_no_join_dev'

                split_dataset_train_path = dataset_train_path / split

                split_dataset_align_out = dataset_align_out / split
                split_dataset_align_out.mkdir(exist_ok=True)

                split_dataset_aligned_out = dataset_aligned_out / split
                split_dataset_aligned_out.mkdir(exist_ok=True)

                input_align_file = create_align_input(train_file_path=split_dataset_train_path,
                                                                   align_file_path=split_dataset_align_out,
                                                                   train_file_name=train_file_name)

                output_align_file, _ = train_alignment(input_align_path=split_dataset_align_out,
                                                                           output_aligned_path=split_dataset_aligned_out,
                                                                           file_name=train_file_name)
                # inspection_file = train_file_name + '.inspection'
                #
                # inspect_alignment_file(split_dataset_align_out / input_align_file,
                #                        split_dataset_aligned_out / output_align_file,
                #                        split_dataset_aligned_out / inspection_file)

                update_alignments_in_file(split_dataset_align_out / input_align_file,
                                          split_dataset_aligned_out / output_align_file,
                                          orig_data_json_path=split_dataset_train_path / f"{train_file_name}.json",
                                          new_data_json_path=split_dataset_train_path / f"aligned_{train_file_name}.json")


def fast_align_text2sql_only_update(datasets, splits):
    for dataset in datasets:
        dataset_train_path = DATA_TRAIN_PATH / dataset
        dataset_align_out = DATA_IN_PATH / dataset
        dataset_aligned_out = DATA_OUT_PATH / dataset
        for split in splits:
            for train_file_name in ['train', 'final_dev']:
                # fix the names of the files in case it is advising :0
                if dataset == 'advising':
                    if train_file_name == 'train':
                        train_file_name = 'new_no_join_train'
                    else:
                        train_file_name = 'final_new_no_join_dev'
                # create the paths
                split_dataset_train_path = dataset_train_path / split
                split_dataset_align_out = dataset_align_out / split
                split_dataset_aligned_out = dataset_aligned_out / split
                input_align_file = train_file_name + '.align'
                output_align_file = train_file_name + '.alignment'
                # read alignments and update the data files
                update_alignments_in_file(split_dataset_align_out / input_align_file,
                                          split_dataset_aligned_out / output_align_file,
                                          orig_data_json_path=split_dataset_train_path / f"{train_file_name}.json",
                                          new_data_json_path=split_dataset_train_path / f"aligned_{train_file_name}.json")


def fast_align_text2filteredsql(datasets, splits, filter=shorten_sql_tokens, reverse=False):
    for dataset in datasets:
        dataset_train_path = DATA_TRAIN_PATH / dataset
        dataset_align_out = DATA_IN_PATH / dataset
        dataset_align_out.mkdir(exist_ok=True)
        dataset_aligned_out = DATA_OUT_PATH / dataset
        dataset_aligned_out.mkdir(exist_ok=True)
        # for split in ["schema_full_split", "new_question_split"]:
        for split in splits:
            for train_file_name in ['train', 'final_dev']:
                # fix the names of the files in case it is advising :0
                if dataset == 'advising':
                    if train_file_name == 'train':
                        train_file_name = 'new_no_join_train'
                    else:
                        train_file_name = 'final_new_no_join_dev'

                split_dataset_train_path = dataset_train_path / split

                split_dataset_align_out = dataset_align_out / split
                split_dataset_align_out.mkdir(exist_ok=True)

                split_dataset_aligned_out = dataset_aligned_out / split
                split_dataset_aligned_out.mkdir(exist_ok=True)

                input_align_file = create_align_input(train_file_path=split_dataset_train_path,
                                                      align_file_path=split_dataset_align_out,
                                                      train_file_name=train_file_name,
                                                      mapping_file='mapping_'+train_file_name+'.json',
                                                      filter=filter,
                                                      reverse=reverse)

                output_align_file, _ = train_alignment(input_align_path=split_dataset_align_out,
                                                       output_aligned_path=split_dataset_aligned_out,
                                                       file_name=train_file_name)
                inspection_file = train_file_name + '.inspection'

                inspect_alignment_file(split_dataset_align_out / input_align_file,
                                       split_dataset_aligned_out / output_align_file,
                                       split_dataset_aligned_out / inspection_file)


if __name__ == '__main__':
    # fast_align_text2filteredsql(datasets=["scholar", "atis", "advising", "geography"],
    #                             splits=["schema_full_split", "new_question_split"],
    #                             filter=shorten_sql_tokens,
    #                             reverse=True)
    fast_align_text2filteredsql(datasets=["scholar", "atis", "advising", "geography"],
                            splits=["schema_full_split", "new_question_split"],
                            filter=shorten_sql_tokens,
                            reverse=False)
    # fast_align_text2filteredsql(datasets=["test"],
    #                              splits=["schema_full_split"])