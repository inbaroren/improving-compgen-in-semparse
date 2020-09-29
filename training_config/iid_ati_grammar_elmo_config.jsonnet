local local_dir = ""; 
{
  "dataset_reader": {
    "type": "grammar_based_text2sql_v3",
    "database_file": "",
    "load_cache": true,
    "save_cache": false,
    "schema_path": local_dir +"data/sql data/atis-schema.csv",
    "token_indexers": {
      "elmo": { "type": "elmo_characters" },
      "tokens": { "type": "single_id" }
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 4,
    "padding_noise": 0,
    "sorting_keys": [ [ "tokens", "num_tokens" ] ]
  },
  "model": {
    "type": "my_text2sql_parser",
    "action_embedding_dim": 100,
    "decoder_beam_search": { "beam_size": 5 },
    "dropout": 0.5,
    "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "dropout": 0,
      "hidden_size": 300,
      "input_size": 1124,
      "num_layers": 1
    },
    "input_attention": { "type": "dot_product" },
    "max_decoding_steps": 300,
    "mydatabase": "atis",
    "schema_path": local_dir +"data/sql data/atis-schema.csv",
    "utterance_embedder": {
      "elmo": {
        "type": "elmo_token_embedder",
        "do_layer_norm": false,
        "dropout": 0,
        "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
      },
      "tokens": {
        "type": "embedding",
        "embedding_dim": 100,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "trainable": true,
        "vocab_namespace": "tokens"
      }
    }
  },
  "train_data_path": local_dir +"data/sql data/atis/new_question_split/aligned_train.json",
  "validation_data_path": local_dir +"data/sql data/atis/new_question_split/aligned_final_dev.json",
  "trainer": {
    "cuda_device": 1,
    "learning_rate_scheduler": {
      "type": "noam",
      "model_size": 600,
      "warmup_steps": 800
    },
    "num_epochs": 30,
    "num_serialized_models_to_keep": 1,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "patience": 5,
    "validation_metric": "+seq_acc"
  }
}
