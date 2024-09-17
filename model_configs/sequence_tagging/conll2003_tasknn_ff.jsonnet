local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'conll2003ner';
local dataset_metadata = (import 'datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local transformer_model = 'bert-base-uncased';
local transformer_hidden_dim = 768;
local max_length = 256;

local ff_hidden = std.parseJson(std.extVar('ff_hidden'));
local label_space_dim = ff_hidden;
local ff_dropout = std.parseJson(std.extVar('ff_dropout_10x'))/10.0;
//local ff_activation = std.parseJson(std.extVar('ff_activation'));
local ff_activation = 'softplus';
//local ff_activation = 'softplus';
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
local inference_score_weight = 1; //std.parseJson(std.extVar('inference_score_weight'));
local cross_entropy_loss_weight = 1; //std.parseJson(std.extVar('cross_entropy_loss_weight'));
local ff_weight_decay = std.parseJson(std.extVar('ff_weight_decay'));
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local task_nn = {
  type: 'sequence-tagging',
  text_field_embedder: {
    token_embedders: {
      tokens: {
        type: 'pretrained_transformer_mismatched',
        model_name: transformer_model,
        max_length: max_length,
      },
    },
  },
  dropout: ff_dropout,
  feedforward: {
    input_dim: transformer_hidden_dim,
    num_layers: ff_linear_layers,
    activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + [ff_activation]),
    hidden_dims: ff_hidden,
    dropout: ([ff_dropout for i in std.range(0, ff_linear_layers - 2)] + [0]),
  }
};

{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  dataset_reader: {
    type: 'conll2003',
    tag_label: 'ner',
    coding_scheme: 'BIOUL',
    token_indexers: {
      tokens: {
        type: 'pretrained_transformer_mismatched',
        model_name: transformer_model,
        max_length: max_length,
      },
    },
  },
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),
  // Model
  model: {
    type: 'seal-ner',
    label_encoding: 'BIOUL',
    sampler: {
      type: 'appending-container',
      log_key: 'sampler',
      constituent_samplers: [],
    },
    task_nn: task_nn,
    inference_module: {
      type: 'sequence-tagging-inference-net-normalized',
      log_key: 'inference_module',
      loss_fn: {
        type: 'combination-loss',
        log_key: 'loss',
        constituent_losses: [
          {
            type: 'sequence-tagging-masked-cross-entropy',
            log_key: 'ce',
            reduction: 'none',
            normalize_y: false,
          },
        ],
        loss_weights: [cross_entropy_loss_weight],
        reduction: 'mean',
      },
    },
    oracle_value_function: { type: 'manhattan', differentiable: true},
    score_nn: {
      type: 'sequence-tagging',
      task_nn: task_nn,
      global_score: {
        type: 'linear-chain',
        num_tags: num_labels,
      },
    },
    loss_fn: {
      type: 'sequence-tagging-margin-based',
      reduction: 'mean',
      oracle_cost_weight: 1.0,
      perceptron_loss_weight: inference_score_weight,
      log_key: 'margin_loss'
},
    initializer: {
      regexes: [
        //[@'.*_feedforward._linear_layers.0.weight', {type: 'normal'}],
        [@'.*feedforward._linear_layers.*weight', (if std.member(['tanh', 'sigmoid'], ff_activation) then { type: 'xavier_uniform', gain: gain } else { type: 'kaiming_uniform', nonlinearity: 'relu' })],
        [@'.*linear_layers.*bias', { type: 'zero' }],
      ],
    },
  },
  data_loader: {
    batch_sampler: {
      type: 'bucket',  // bucket is only good for tasks that involve seq
      batch_size: 16,
    },
  },
  trainer: {
    type: 'gradient_descent_minimax',
    num_epochs: if test == '1' then 10 else 300,
    grad_norm: { task_nn: 1.0 },
    patience: 20,
    validation_metric: '+f1-measure-overall',
    cuda_device: std.parseInt(cuda_device),
    learning_rate_schedulers: {
      task_nn: {
        type: 'reduce_on_plateau',
        factor: 0.5,
        mode: 'max',
        patience: 5,
        verbose: true,
      },
    },
    optimizer: {
      optimizers: {
        task_nn:
          {
            lr: 0.00001,
            weight_decay: ff_weight_decay,
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
          save_model_archive: false,
        },
      ]
      else []
    ),
    inner_mode: 'score_nn',
    num_steps: { task_nn: 1, score_nn: 1 },
  },
}
