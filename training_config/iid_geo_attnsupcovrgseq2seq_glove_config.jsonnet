local local_dir = ""; 

{
  "dataset_reader": {
    "type": "attn_sup_seq2seq",
    "database_path": null,
    "remove_unneeded_aliases": false,
    "schema_path": local_dir + "data/sql data/geography-schema.csv",
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    },
    "target_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "target_tokens"
      }
    },
    "use_prelinked_entities": true
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 1,
    "padding_noise": 0,
    "sorting_keys": [ [ "target_tokens", "num_tokens" ] ]
  },
  "model": {
    "type": "attn_sup_seq2seq",
    "attention": {
      "type": "coveragev2",
      "matrix_dim": 400,
      "vector_dim": 400
    },
    "attn_loss_lambda": 0.1,
    "beam_size": 5,
    "dec_dropout": 0.2,
    "emb_dropout": 0.5,
    "encoder": {
      "type": "lstm",
      "bidirectional": true,
      "dropout": 0,
      "hidden_size": 200,
      "input_size": 100,
      "num_layers": 1
    },
    "max_decoding_steps": 300,
    "schema_path": local_dir + "data/sql data/geography-schema.csv",
    "source_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": 100,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
        "trainable": true,
        "vocab_namespace": "source_tokens"
      }
    },
    "target_embedding_dim": 100,
    "target_namespace": "target_tokens",
    "use_bleu": true
  },
  "train_data_path": local_dir + "data/sql data/geography/new_question_split/aligned_train.json",
  "validation_data_path": local_dir + "data/sql data/geography/new_question_split/aligned_final_dev.json",
  "trainer": {
    "cuda_device": 0,
    "learning_rate_scheduler": {
      "type": "noam",
      "model_size": 400,
      "warmup_steps": 800
    },
    "num_epochs": 65,
    "num_serialized_models_to_keep": 1,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "patience": 17,
    "validation_metric": "+seq_acc"
  }
}
