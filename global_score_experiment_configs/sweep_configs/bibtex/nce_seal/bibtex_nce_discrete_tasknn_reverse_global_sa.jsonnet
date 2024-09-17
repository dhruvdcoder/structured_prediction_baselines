local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = std.parseJson(std.extVar('dataset_name'));
local dataset_metadata = (import '../../../../model_configs/multilabel_classification/datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

// model variables
local ff_hidden = std.parseJson(std.extVar('ff_hidden'));
local label_space_dim = ff_hidden;
local ff_dropout = std.parseJson(std.extVar('ff_dropout_10x')) / 10.0;
local ff_activation = 'softplus';
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
local ff_weight_decay = std.parseJson(std.extVar('ff_weight_decay'));
local global_score_hidden_dim = std.parseJson(std.extVar('global_score_hidden_dim'));
local global_score_num_layers = std.parseJson(std.extVar('global_score_num_layers'));
local global_score_num_heads = std.parseJson(std.extVar('global_score_num_heads'));
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local cross_entropy_loss_weight = std.parseJson(std.extVar('cross_entropy_loss_weight'));
local dvn_score_loss_weight = std.parseJson(std.extVar('dvn_score_loss_weight'));
local task_temp = std.parseJson(std.extVar('task_nn_steps')); # variable for task_nn.steps
local task_nn_steps = (if std.toString(task_temp) == '0' then 1 else task_temp);
local score_temp = std.parseJson(std.extVar('score_nn_steps')); # variable for score_nn.steps
local score_nn_steps = (if std.toString(score_temp) == '0' then 1 else score_temp);
{
  [if use_wandb then 'type']: 'train_test_log_to_wandb',
  evaluate_on_test: true,
  // Data
  dataset_reader: {
    type: 'arff',
    num_labels: num_labels,
  },
  validation_dataset_reader: {
    type: 'arff',
    num_labels: num_labels,
  },
  train_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                    dataset_metadata.train_file),
  validation_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                         dataset_metadata.validation_file),
  test_data_path: (data_dir + '/' + dataset_metadata.dir_name + '/' +
                   dataset_metadata.test_file),

  // Model
  model: {
    type: 'multi-label-classification-with-infnet',
    sampler: {
      type: 'appending-container',
      log_key: 'sampler',
      constituent_samplers: [],
    },
    task_nn: {
      type: 'multi-label-classification',
      feature_network: {
        input_dim: num_input_features,
        num_layers: ff_linear_layers,
        activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + [ff_activation]),
        hidden_dims: ff_hidden,
        dropout: ([ff_dropout for i in std.range(0, ff_linear_layers - 2)] + [0]),
      },
      label_embeddings: {
        embedding_dim: ff_hidden,
        vocab_namespace: 'labels',
      },
    },
    inference_module: {
      type: 'multi-label-inference-net-normalized',
      log_key: 'inference_module',
      loss_fn: {
        type: 'combination-loss',
        log_key: 'loss',
        constituent_losses: [
          {
            type: 'multi-label-score-loss',
            log_key: 'neg.nce_score',
            normalize_y: true,
            reduction: 'none',
          },  //This loss can be different from the main loss // change this
          {
            type: 'multi-label-bce',
            reduction: 'none',
            log_key: 'bce',
          },
        ],
        loss_weights: [dvn_score_loss_weight, cross_entropy_loss_weight],
        reduction: 'mean',
      },
    },
    oracle_value_function: { type: 'per-instance-f1', differentiable: false },
    score_nn: {
      type: 'multi-label-classification',
      task_nn: {
        type: 'multi-label-classification',
        feature_network: {
          input_dim: num_input_features,
          num_layers: ff_linear_layers,
          activations: ([ff_activation for i in std.range(0, ff_linear_layers - 2)] + [ff_activation]),
          hidden_dims: ff_hidden,
          dropout: ([ff_dropout for i in std.range(0, ff_linear_layers - 2)] + [0]),
        },
        label_embeddings: {
          embedding_dim: ff_hidden,
          vocab_namespace: 'labels',
        },
      },
      global_score: {
        type: 'multi-label-feedforward-global-self-attention',
        feedforward_x: {
          input_dim: ff_hidden,
          num_layers: 1,
          activations: 'linear',
          hidden_dims: global_score_hidden_dim,
        },
        feedforward_label1: {
          input_dim: global_score_hidden_dim,
          num_layers: 1,
          activations: 'linear',
          hidden_dims: global_score_hidden_dim,
        },
        feedforward_label2: {
          input_dim: global_score_hidden_dim,
          num_layers: global_score_num_layers,
          activations: 'linear',
          hidden_dims: global_score_hidden_dim,
        },
        hidden_dim: global_score_hidden_dim,
        num_labels: num_labels,
        num_heads: global_score_num_heads,
        num_layers: global_score_num_layers
      },
    },
    loss_fn: {
      type: 'multi-label-nce-ranking-with-discrete-sampling',
      log_key: 'nce',
      num_samples: 10,
      sign: '-',
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
    shuffle: true,
    batch_size: 32,
  },
  trainer: {
    type: 'gradient_descent_minimax',
    num_epochs: if test == '1' then 10 else 300,
    grad_norm: { task_nn: 10.0 },
    patience: 20,
    validation_metric: '+fixed_f1',
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
            lr: 0.001,
            weight_decay: ff_weight_decay,
            type: 'adamw',
          },
        score_nn: {
          lr: 0.005,
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
    num_steps: { task_nn: task_nn_steps, score_nn: score_nn_steps },
  },
}
