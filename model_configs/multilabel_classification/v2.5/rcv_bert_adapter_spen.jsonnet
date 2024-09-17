local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'rcv1'; //std.parseJson(std.extVar('dataset_name'));
local dataset_metadata = (import '../datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

// model variables
// // common
local ff_activation = 'softplus';
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local ff_linear_layers = 2;
local weight_decay = std.parseJson(std.extVar('weight_decay'));
local dropout = std.parseJson(std.extVar('dropout_10x')) / 10.0;

// // score_nn
local transformer_model = 'bert-base-uncased';  // huggingface name of the model
local transformer_dim = 768;
local transformer_vocab_size = 30522;
local score_nn_weight_decay = weight_decay;
local global_score_hidden_dim = std.parseJson(std.extVar('global_score_hidden_dim'));
local score_nn_dropout = dropout;
// // task_nn
local task_nn_dropout = dropout;
local task_nn_weight_decay = weight_decay;

// model variables
local gbi_lr = std.parseJson(std.extVar('gbi_lr'));
local gbi_optim = 'adam';

local feature_network = {
  text_field_embedder: {
    token_embedders: {
      x: {
        type: 'pretrained_transformer_with_adapter',
        model_name: transformer_model,
      },
    },
  },
  seq2vec_encoder: {
    type: 'bert_pooler',
    pretrained_model: transformer_model,
  },
  final_dropout: 0,
  feedforward: {
    input_dim: transformer_dim,
    num_layers: ff_linear_layers,
    activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + [ff_activation]),
    hidden_dims: ([transformer_dim * 2 for i in std.range(0, ff_linear_layers - 2)] + [transformer_dim]),
    dropout: ([task_nn_dropout for i in std.range(0, ff_linear_layers - 2)] + [0]),
  },
};

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  // Data
  dataset_reader: {
    type: 'rcv',
    //[if test == '1' then 'max_instances']: 100,
    token_indexers: {
      x: {
        type: 'pretrained_transformer',
        model_name: transformer_model,
      },
    },
    tokenizer: {
      type: 'pretrained_transformer',
      model_name: transformer_model,
      max_length: 512,
    },
  },
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),

vocabulary: {
        type: 'from_files',
        directory: data_dir + '/' + dataset_metadata.dir_name + '/' + 'bert_vocab'
    },
  // Model
  model: {
    type: 'multi-label-classification',
    sampler: {
      type: 'gradient-based-inference-tasknn-init',
      log_key: 'sampler',
      inference_nn: {
        type: 'multi-label-text-classification',
        feature_network: feature_network,
        label_embeddings: {
          embedding_dim: transformer_dim,
          vocab_namespace: 'labels',
        },
      },
      gbi_sampler: //GBI
        {
          log_key: 'gbi',
          gradient_descent_loop: {
            optimizer: {
              lr: gbi_lr,  //0.1
              weight_decay: 0,
              type: gbi_optim,
            },
          },
          loss_fn: {
            type: 'multi-label-inference',
            oracle_cost_weight: 1.0,
            inference_score_weight: 1,
            log_key: 'margin_loss',
          },
          output_space: { type: 'multi-label-relaxed', num_labels: num_labels, default_value: 0.0 },
          stopping_criteria: 20,
          sample_picker: { type: 'best' },
          number_init_samples: 1,
          random_mixing_in_init: 1.0,
        },
    },
    inference_module: {
      type: 'gradient-based-inference-tasknn-init',
      log_key: 'sampler',
      inference_nn: {
        type: 'multi-label-text-classification',
        feature_network: feature_network,
        label_embeddings: {
          embedding_dim: transformer_dim,
          vocab_namespace: 'labels',
        },
      },
      gbi_sampler: //GBI
        {
          log_key: 'gbi',
          gradient_descent_loop: {
            optimizer: {
              lr: gbi_lr,  //0.1
              weight_decay: 0,
              type: gbi_optim,
            },
          },
          loss_fn: {
            type: 'multi-label-score-loss',
            reduction: 'none',
            log_key: 'score_loss',
          },
          output_space: { type: 'multi-label-relaxed', num_labels: num_labels, default_value: 0.0 },
          stopping_criteria: 20,
          sample_picker: { type: 'best' },
          number_init_samples: 1,
          random_mixing_in_init: 1.0,
        },
    },
    oracle_value_function: { type: 'per-instance-f1', differentiable: false },
    score_nn: {
      type: 'multi-label-classification',
      task_nn: {
        type: 'multi-label-text-classification',
        feature_network: feature_network,
        label_embeddings: {
          embedding_dim: transformer_dim,
          vocab_namespace: 'labels',
        },
      },
      global_score: {
        type: 'multi-label-feedforward',
        feedforward: {
          input_dim: num_labels,
          num_layers: 1,
          activations: ff_activation,
          hidden_dims: global_score_hidden_dim,
        },
      },
    },
    loss_fn: {
      type: 'multi-label-structured-svm',
      oracle_cost_weight: 1.0,
      perceptron_loss_weight: 1,
      reduction: 'mean',
      log_key: 'margin_loss',
    },
    initializer: {
      regexes: [
        //[@'.*_feedforward._linear_layers.0.weight', {type: 'normal'}],
        [@'.*_linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 8,  // effective batch size = batch_size*num_gradient_accumulation_steps
      sorting_keys: ['x'],
    },
    num_workers: 5,
    max_instances_in_memory: if test == '1' then 10 else 1000,
    start_method: 'spawn',
  },
  trainer: {
    type: 'gradient_descent_minimax',
    num_epochs: if test == '1' then 10 else 300,
    grad_norm: { score_nn: 10.0 },
    num_gradient_accumulation_steps: 2,
    patience: 20,
    validation_metric: '+fixed_f1',
    cuda_device: std.parseInt(cuda_device),
    learning_rate_schedulers: {
      score_nn: {
        type: 'reduce_on_plateau',
        factor: 0.5,
        mode: 'max',
        patience: 5,
        verbose: true,
      },
    },
    optimizer: {
      optimizers: {
        score_nn: {
          lr: 5e-5,
          weight_decay: score_nn_weight_decay,
          type: 'adamw',
        },
      },
    },
    checkpointer: {
      keep_most_recent_by_count: 1,
    },
    callbacks: [
      'track_epoch_callback',
      'slurm',
    ] + (
      if use_wandb then [
        {
          type: 'wandb_allennlp',
          sub_callbacks: [{ type: 'log_best_validation_metrics', priority: 100 }],
        },
      ]
      else []
    ),
    inner_mode: 'task_nn',
    num_steps: { task_nn: 0, score_nn: 1 },
  },
}
