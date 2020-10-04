# Improving Compositional Generalization In Semantic Parsing
Official Github repo for the paper ["Improving Compositional Generalization In Semantic Parsing"](https://arxiv.org/abs/0000.0000).

This repo is basically an allennlp package, so allennlp installation is required. Notice that Text2SQL models require version 0.9.0. 

For each model, an example for a configuration file is available at /training_config. To train a model, update the path, dataset name, and split name (for iid split use 'new_question_split', for program split use 'schema_full_split') in the configuration file.

For example:
`allennlp train ./improving-compgen-in-semparse/training_config/iid_ati_seq2seq_glove_config.jsonnet -s YOUR_OUTPUT_LOCATION --include-package text2sql`

The parameters for each of the experiments in the paper are listed in /training_config/best_params.xlsx .