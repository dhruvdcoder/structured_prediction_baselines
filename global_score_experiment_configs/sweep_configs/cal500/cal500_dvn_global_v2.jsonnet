local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = std.parseJson(std.extVar('dataset_name'));
//local dataset_name = 'bibtex_strat';
local dataset_metadata = (import '../../../model_configs/multilabel_classification/datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local num_input_features = dataset_metadata.input_features;

// model variables
local ff_hidden = std.parseJson(std.extVar('ff_hidden'));
local ff_dropout = std.parseJson(std.extVar('ff_dropout_10x')) / 10.0;
local ff_activation = 'softplus';
local ff_linear_layers = std.parseJson(std.extVar('ff_linear_layers'));
local ff_weight_decay = std.parseJson(std.extVar('ff_weight_decay'));
local global_score_hidden_dim = std.parseJson(std.extVar('global_score_hidden_dim'));
local global_score_num_layers = std.parseJson(std.extVar('global_score_num_layers'));
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local gbi_lr = std.parseJson(std.extVar('gbi_lr'));

//local gbi_optim = std.parseJson(std.extVar('gbi_optim'));
local gbi_optim = 'adam';
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

  [if dataset_name == 'eurlexev' then 'vocabulary']: {type: "from_files", directory: (data_dir + '/' + dataset_metadata.dir_name + '/' + 'eurlex-ev-vocab'),},
  // Model
  model: {
    type: 'multi-label-classification',
    sampler: {
      type: 'appending-container',
      log_key: 'sampler',
      constituent_samplers: [
        //GBI
        {
          type: 'gradient-based-inference',
          log_key: 'gbi',
          gradient_descent_loop: {
            optimizer: {
              lr: gbi_lr,  //0.1
              weight_decay: 0,
              type: gbi_optim,
            },
          },
          loss_fn: { type: 'multi-label-dvn-score', reduction: 'none', log_key: 'neg_dvn_score' },  //This loss can be different from the main loss // change this
          output_space: { type: 'multi-label-relaxed', num_labels: num_labels, default_value: 0.0 },
          stopping_criteria: 20,
          sample_picker: { type: 'best' },
          number_init_samples: 1,
          random_mixing_in_init: 1.0,
        },
        // Adversarial
        {
          type: 'gradient-based-inference',
          log_key: 'adv',
          gradient_descent_loop: {
            optimizer: {
              lr: gbi_lr,  //0.1
              weight_decay: 0,
              type: gbi_optim,
            },
          },
          loss_fn: {
            type: 'negative',
            log_key: 'neg',
            constituent_loss: { type: 'multi-label-dvn-bce', reduction: 'none', log_key: 'dvn_bce' },
            reduction: 'none',
          },
          output_space: { type: 'multi-label-relaxed', num_labels: num_labels, default_value: 0.0 },
          stopping_criteria: 20,
          sample_picker: { type: 'best' },
          number_init_samples: 1,
          random_mixing_in_init: 1.0,
        },

        { type: 'ground-truth' },
      ],
    },
    inference_module: {
      type: 'gradient-based-inference',
      log_key: 'inference',
      gradient_descent_loop: {
        optimizer: {
          lr: gbi_lr,  //0.1
          weight_decay: 0,
          type: gbi_optim,
        },
      },
      loss_fn: { type: 'multi-label-dvn-score', reduction: 'none', log_key: 'neg_dvn_score' },  //This loss can be different from the main loss
      output_space: { type: 'multi-label-relaxed', num_labels: num_labels, default_value: 0.0 },
      stopping_criteria: 30,
      sample_picker: { type: 'best' },
      number_init_samples: 1,
      random_mixing_in_init: 1.0,
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
        type: 'multi-label-feedforward-global-v2',
        num_labels: num_labels,
        activation: ff_activation,
        feedforward: {
          input_dim: ff_hidden,
          num_layers: global_score_num_layers,
          activations: ff_activation,
          hidden_dims: global_score_hidden_dim * num_labels,
        },
      },
    },
    loss_fn: { type: 'multi-label-dvn-bce', log_key: 'dvn_bce' },
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
    grad_norm: { score_nn: 10.0 },
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
          lr: 0.0008,
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
        },
      ]
      else []
    ),
    inner_mode: 'task_nn',
    num_steps: { task_nn: 0, score_nn: 1 },
  },
}