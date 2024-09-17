local test = std.extVar('TEST');  // a test run with small dataset
local data_dir = std.extVar('DATA_DIR');
local cuda_device = std.extVar('CUDA_DEVICE');
local use_wandb = (if test == '1' then false else true);

local dataset_name = 'conll2003ner';
local dataset_metadata = (import 'datasets.jsonnet')[dataset_name];
local num_labels = dataset_metadata.num_labels;
local transformer_model = 'bert-base-uncased';
local transformer_hidden_dim = 768;
local max_length = 512;

local ff_activation = 'softplus';
local cross_entropy_loss_weight = std.parseJson(std.extVar('cross_entropy_loss_weight'));
local dvn_score_loss_weight = std.parseJson(std.extVar('dvn_score_loss_weight'));
local weight_decay = std.parseJson(std.extVar('weight_decay'));
local tasknn_lr = std.parseJson(std.extVar('tasknn_lr'));
local gain = (if ff_activation == 'tanh' then 5 / 3 else 1);
local score_temp = std.parseJson(std.extVar('score_nn_steps')); # variable for score_nn.steps
local score_nn_steps = (if std.toString(score_temp) == '0' then 1 else score_temp);

local task_nn = {
  type: 'sequence-tagging',
  text_field_embedder: {
    token_embedders: {
      tokens: {
        type: 'pretrained_transformer_mismatched_with_adapter',
        model_name: transformer_model,
        max_length: max_length,
      },
    },
  },
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
  vocabulary: {
    type: 'from_files',
    directory: data_dir + '/' + dataset_metadata.dir_name + '/' + 'bert_vocab',
  },
  // Model
  model: {
    type: 'seal-ner',
    label_encoding: 'BIOUL',
    task_nn: task_nn,
    inference_module: {
      type: 'sequence-tagging-inference-net-normalized',
      log_key: 'inference_module',
      loss_fn: {
        type: 'combination-loss',
        log_key: 'loss',
        constituent_losses: [
          {
            type: 'sequence-tagging-score-loss', // jy: fix here.. 
            log_key: 'neg.nce_score',
            normalize_y: true,
            reduction: 'none',
          },  //This loss can be different from the main loss // change this
          {
            type: 'sequence-tagging-masked-cross-entropy',
            log_key: 'ce',
            reduction: 'none',
            normalize_y: false,
          },
        ],
        loss_weights: [dvn_score_loss_weight, cross_entropy_loss_weight],
        reduction: 'mean',
      },
    },
    score_nn: {
      type: 'sequence-tagging',
      task_nn: task_nn,
      global_score: {
        type: 'cnn', //type: 'linear-chain', 
        num_tags: num_labels, 
      },
    },
    loss_fn: {
      type: 'seqtag-nce-ranking-with-discrete-sampling', 
      reduction: 'mean',
      log_key: 'seq_nce_loss',
      num_samples: 200,
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
      type: 'bucket',
      batch_size: 32,  // effective batch size = batch_size*num_gradient_accumulation_steps
      sorting_keys: ['tokens'],
    },
    num_workers: 5,
    max_instances_in_memory: if test == '1' then 10 else 1000,
    start_method: 'spawn',
  },
  trainer: {
    type: 'gradient_descent_minimax',
    num_epochs: if test == '1' then 10 else 50,
    grad_norm: { task_nn: 1.0 },
    patience: 4,
    validation_metric: '+f1-measure-overall',
    cuda_device: std.parseInt(cuda_device),
    learning_rate_schedulers: {
      task_nn: {
        type: 'reduce_on_plateau',
        factor: 0.5,
        mode: 'max',
        patience: 2,
        verbose: true,
      },
    },
    optimizer: {
      optimizers: {
        task_nn:
          {
            lr: tasknn_lr,
            weight_decay: weight_decay,
            type: 'huggingface_adamw',
          },
        score_nn: {
          lr: 0.00001,
          weight_decay: weight_decay,
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
          watch_model: false,
        },
      ]
      else []
    ),
    inner_mode: 'score_nn',
    num_steps: { task_nn: 1, score_nn: 1 },
  },
}
