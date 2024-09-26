# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os, sys
path = os.path.dirname(os.path.abspath('.'))
if not path in sys.path: sys.path.append(path)

# %%
import torch
from pathlib import Path
from allennlp.models import Model
from allennlp.common import Params
from allennlp.common import util as common_util
common_util.import_module_and_submodules('structured_prediction_baselines')

# %%
from allennlp.data import (
    Field,
    DataLoader,
    DatasetReader,
    Instance,
    Batch,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.training import Trainer
from allennlp.nn import util

from structured_prediction_baselines.common import ModelMode

# %% [markdown]
# # Load model, dataset-reader, data and vocab

# %%
dataset_name = 'expr_fun'
serialization_dir = "../.allennlp_models/dvn_model_wts/expr_fun_dvn/"  # path to the directory containing config, model weights and vocab
config_file = os.path.join(serialization_dir, "config.json")
vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
weights_file = os.path.join(serialization_dir, "best.th")

# %%
loaded_params = Params.from_file(config_file)
loaded_model = Model.load(loaded_params, serialization_dir, weights_file)
loaded_model.eval()

dataset_reader = DatasetReader.from_params(loaded_params.get("dataset_reader"))

data_loader_params = loaded_params.get("data_loader")
train_data_loader = DataLoader.from_params(
    params=data_loader_params.duplicate(),
    reader=dataset_reader,
    data_path='../' + loaded_params.get("train_data_path")
)

dev_data_loader = DataLoader.from_params(
    reader=dataset_reader,
    data_path='../' + loaded_params.get("validation_data_path"),
    params=data_loader_params,
)

#vocab = Vocabulary.from_instances(train_data_loader.iter_instances())
vocab = Vocabulary.from_files(vocabulary_dir)

# %%
loaded_model

# %%
train_data_loader.index_with(vocab)
dev_data_loader.index_with(vocab)

# %%
# uncomment if you want to load trainer
# loaded_params['trainer']['cuda_device'] = -1
# loaded_params['trainer']['num_epochs'] = 1

# trainer = Trainer.from_params(
#     model=loaded_model,
#     serialization_dir=serialization_dir,
#     data_loader=train_data_loader,
#     validation_data_loader=dev_data_loader,
#     params=loaded_params.pop("trainer"),
# )

# %% [markdown]
# ## Get an instance from the data

# %%
dev_instances = dev_data_loader.iter_instances()
inst0 = next(dev_instances)
instances = [inst0]

# %%
loaded_model.forward_on_instance(inst0, mode=ModelMode.COMPUTE_SCORE)

# %%
device = 0 if torch.cuda.is_available() else -1
with torch.no_grad():
    dataset = Batch(instances)
    dataset.index_instances(vocab)
    model_input = util.move_to_device(dataset.as_tensor_dict(), device=device)

# %%
model_input

# %% [markdown]
# # Contour Plots

# %%
# %matplotlib inline
# #%matplotlib notebook
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import matplotlib


# %%
def compute_score_values(model, model_input, dim1, dim2, rmin=-0.5, rmax=1.5, n=100, sigmoid=False):
    "Compute scores on a 2D mesh of (yi, yj) with other yk values and the input x fixed."
    dim1_values = np.linspace(rmin, rmax, n)
    dim2_values = np.linspace(rmin, rmax, n)
    dim1_mesh, dim2_mesh = np.meshgrid(dim1_values, dim2_values)
    z = np.zeros((n, n))
    
    x = model_input['x']
    y = model_input['labels'].clone().detach().float()
    # y = 1 - y
    for i in range(len(dim1_values)):
        for j in range(len(dim2_values)):
            y[0][dim1] = dim1_mesh[i][j]
            y[0][dim2] = dim2_mesh[i][j]
            with torch.no_grad():
                z[j][i] = loaded_model(x, y, mode=ModelMode.COMPUTE_SCORE)['score']
    if sigmoid:
        z = torch.sigmoid(torch.from_numpy(z)).numpy()
    return dim1_mesh, dim2_mesh, z


# %%
def compute_global_score_values(model, dim1, dim2, rmin=-0.5, rmax=1.5, n=100, sigmoid=False):
    """Same as 'compute_score_values' but only uses global energy and hence does not take x."""
    dim1_values = np.linspace(rmin, rmax, n)
    dim2_values = np.linspace(rmin, rmax, n)
    dim1_mesh, dim2_mesh = np.meshgrid(dim1_values, dim2_values)
    z = np.zeros((n, n))
    
    y = torch.rand(1,loaded_model.vocab.get_vocab_size('labels'))
    # y = 1 - y
    for i in range(len(dim1_values)):
        for j in range(len(dim2_values)):
            y[0][dim1] = dim1_mesh[i][j]
            y[0][dim2] = dim2_mesh[i][j]
            with torch.no_grad():
                z[j][i] = loaded_model.score_nn.compute_global_score(y, buffer={})
    if sigmoid:
        z = torch.sigmoid(torch.from_numpy(z)).numpy()
    return dim1_mesh, dim2_mesh, z


# %%
Y0, Y1, z = compute_global_score_values(loaded_model, 46, 280, rmin=-0.5, rmax=1.5,  n=200, sigmoid=False)


# %%
def compute_grad_2d(model, dim1, dim2, inner_loop=100, outer_loop=100):
    x = []
    y = [] 
    z = []
    for i in range(outer_loop):
        Y = torch.rand(1,loaded_model.vocab.get_vocab_size('labels'))
        Y.requires_grad = True
        x_inner = []
        y_inner = []
        z_inner = []
        for j in range(inner_loop):
            v1 = np.random.rand(1)
            v2 = np.random.rand(1)
            Y[0][dim1] = float(v1)
            Y[0][dim2] = float(v2)
            Z = torch.autograd.grad(loaded_model.score_nn.compute_global_score(Y, buffer={}), Y)[0][0][dim2]
            x_inner.append(float(Y[0][dim1]))
            y_inner.append(float(Y[0][dim2]))
            z_inner.append(float(Z))
        x+=x_inner
        y+=y_inner
        #z+=z_inner
        #z+=((np.array(z_inner) - np.mean(z_inner))/np.mean(z_inner)).tolist()
        #z+=((np.array(z_inner))/np.sum(z_inner)).tolist()
        temp = np.array(z_inner)
        z += ((temp - temp.mean())/temp.std()).tolist()
        
    return x, y, z


# %%
# How to get indices for various labels? Uncomment the following code and make appropriate changes.
# Related
related_indices = [(vocab.get_token_index(yi, namespace='labels'), vocab.get_token_index(yk, namespace='labels')) for yi, yk in [
    ('18.01', '18'), 
    ('01.01', '01'), 
    ('18.02.01', '18.02'), 
    ('16.01', '16'), 
    ('10.01', '10'),
    ('01.05', '01')]]
print(related_indices)


# %%
# Generate and save plots
base_path = Path('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/')
for yi, yk in related_indices:
    # yi \implies yk
    x2,y2,z2 = compute_grad_2d(loaded_model, yk , yi, inner_loop=100, outer_loop=10,)
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(x2, y2, c=z2, cmap=cm)
    cb = plt.colorbar(sc,shrink=0.7, aspect=20*0.7)
    cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
    plt.ylabel(r"$y_k$", fontsize=15)
    plt.xlabel(r"$y_i$", fontsize=15)
    plt.savefig(base_path / f'grads_positive_rel_{dataset_name}_{yi}_{yk}.pdf')
    plt.show()    

# %%
# Generate and save plots
# How to get indices for various labels? Uncomment the following code and make appropriate changes.
# Related
un_related_indices = [(vocab.get_token_index(yi, namespace='labels'), vocab.get_token_index(yk, namespace='labels')) for yi, yk in [
    ('32.01', '30.01'), 
    ('20.01', '01.01.03.05'),
    ('01.01.03.05', '18.02.01.01'),
    ('01.01.03.05', '18.02'),
    ('01.01.03.05', '16.21.15'),
    ('30.05', '01.01.03.05')
    ]]
print(un_related_indices)
base_path = Path('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/')
for yi, yk in un_related_indices:
    # yi \implies yk
    x2,y2,z2 = compute_grad_2d(loaded_model, yk , yi, inner_loop=100, outer_loop=10,)
    cm = plt.cm.get_cmap('RdYlBu')
    sc = plt.scatter(x2, y2, c=z2, cmap=cm)
    cb = plt.colorbar(sc,shrink=0.7, aspect=20*0.7)
    cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
    plt.ylabel(r"$y_k$", fontsize=15)
    plt.xlabel(r"$y_i$", fontsize=15)
    plt.savefig(base_path / f'grads_positive_no_rel_{dataset_name}_{yi}_{yk}.pdf')
    plt.show()    

# %% [markdown]
# # Rough work

# %%
#plt.rcParams['text.usetex'] = True

x2,y2,z2 = compute_grad_2d(loaded_model, 55, 45, inner_loop=100, outer_loop=10,)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(x2, y2, c=z2, cmap=cm)
plt.colorbar(sc)
#plt.xlabel('y_i')
#plt.ylabel('y_j')
plt.show()

# %%
yk = 45
yi = 55
# yi \implies yk
x2,y2,z2 = compute_grad_2d(loaded_model, yk , yi, inner_loop=100, outer_loop=10,)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(x2, y2, c=z2, cmap=cm)
cb = plt.colorbar(sc,shrink=0.7, aspect=20*0.7)
cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
plt.ylabel(r"$y_k$", fontsize=15)
plt.xlabel(r"$y_i$", fontsize=15)
plt.savefig('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/grads_positive_rel_{}.pdf')
plt.show()

# %%
x2,y2,z2 = compute_grad_2d(loaded_model, 21, 55, inner_loop=100, outer_loop=10,)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(x2, y2, c=z2, cmap=cm)
cb = plt.colorbar(sc,shrink=0.7, aspect=20*0.7)
cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
plt.ylabel(r"$y_k$", fontsize=15)
plt.xlabel(r"$y_i$", fontsize=15)
plt.savefig('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/grads_no_rel.pdf')

plt.show()

# %%
x2 = np.random.rand(1000)
y2 = np.random.rand(1000)
temp = 1/y2
z2 = (temp -temp.mean())/temp.std()
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(x2, y2, c=z2, cmap=cm, vmax=1)
cb = plt.colorbar(sc,shrink=0.7, aspect=20*0.7)
cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
plt.ylabel(r"$y_k$", fontsize=15)
plt.xlabel(r"$y_i$", fontsize=15)
plt.savefig('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/grads_ce.pdf')

plt.show()

# %%
x1,y1,z1 = compute_grad_2d(loaded_model, 55, 45, inner_loop=100, outer_loop=10,)
x2,y2,z2 = compute_grad_2d(loaded_model, 21, 55, inner_loop=100, outer_loop=10,)
x3 = np.random.rand(1000)
y3 = np.random.rand(1000)
temp = 1/y3
z3 = (temp -temp.mean())/temp.std()
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10,3),  constrained_layout=True)
cm = plt.cm.get_cmap('RdYlBu')
axs[0].scatter(x3, y3, c=z3, cmap=cm)
axs[1].scatter(x1, y1, c=z1, cmap=cm)
axs[2].scatter(x2, y2, c=z2, cmap=cm)
axs[0].set_ylabel(r"$y_k$", fontsize=15)
axs[1].set_xlabel(r"$y_i$", fontsize=15)
cb = plt.colorbar(sc,shrink=0.7, aspect=40*0.7)
cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
plt.show()
fig.savefig('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/grads.pdf')

# %%
print(w)

# %%
#plt.rcParams['text.usetex'] = True
x2,y2,z2 = compute_grad_2d(loaded_model, 55, 45, inner_loop=100, outer_loop=10,)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(x2, y2, c=z2, cmap=cm)
plt.colorbar(sc)
#plt.xlabel('y_i')
#plt.ylabel('y_j')
plt.show()

# %%
yk = 15
yi = 1
# yi \implies yk
x2,y2,z2 = compute_grad_2d(loaded_model, yk , yi, inner_loop=100, outer_loop=10,)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(x2, y2, c=z2, cmap=cm)
cb = plt.colorbar(sc,shrink=0.7, aspect=20*0.7)
cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
plt.ylabel(r"$y_k$", fontsize=15)
plt.xlabel(r"$y_i$", fontsize=15)
plt.savefig('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/grads_positive_rel_dvn_1_55.pdf')
plt.show()

# %%
yk = 2
yi = 1
# yi \implies yk
x2,y2,z2 = compute_grad_2d(loaded_model, yk , yi, inner_loop=100, outer_loop=10,)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(x2, y2, c=z2, cmap=cm)
cb = plt.colorbar(sc,shrink=0.7, aspect=20*0.7)
cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
plt.ylabel(r"$y_k$", fontsize=15)
plt.xlabel(r"$y_i$", fontsize=15)
plt.savefig('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/grads_positive_rel_dvn_1_55.pdf')
plt.show()

# %%
yk = 26
yi = 125
# yi \implies yk
x2,y2,z2 = compute_grad_2d(loaded_model, yk , yi, inner_loop=100, outer_loop=10,)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(x2, y2, c=z2, cmap=cm)
cb = plt.colorbar(sc,shrink=0.7, aspect=20*0.7)
cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
plt.ylabel(r"$y_k$", fontsize=15)
plt.xlabel(r"$y_i$", fontsize=15)
plt.savefig('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/grads_positive_rel_dvn_1_55.pdf')
plt.show()

# %%
print(vocab.get_token_index('02.01', namespace='labels'))
print(vocab.get_token_index('02.01.01', namespace='labels'))

print(vocab.get_token_from_index(21, namespace='labels'))
print(vocab.get_token_from_index(55, namespace='labels'))

# %%
yk = 2
yi = 7
# yi \implies yk
x2,y2,z2 = compute_grad_2d(loaded_model, yk , yi, inner_loop=100, outer_loop=10,)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(x2, y2, c=z2, cmap=cm)
cb = plt.colorbar(sc,shrink=0.7, aspect=20*0.7)
cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
plt.ylabel(r"$y_k$", fontsize=15)
plt.xlabel(r"$y_i$", fontsize=15)
plt.savefig('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/grads_positive_rel_dvn_1_55.pdf')
plt.show()

# %%
yk = 125
yi = 487
# yi \implies yk
x2,y2,z2 = compute_grad_2d(loaded_model, yk , yi, inner_loop=100, outer_loop=10,)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(x2, y2, c=z2, cmap=cm)
cb = plt.colorbar(sc,shrink=0.7, aspect=20*0.7)
cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
plt.ylabel(r"$y_k$", fontsize=15)
plt.xlabel(r"$y_i$", fontsize=15)
plt.savefig('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/grads_positive_rel_dvn_1_55.pdf')
plt.show()

# %%
yk = 125
yi = 35
# yi \implies yk
x2,y2,z2 = compute_grad_2d(loaded_model, yk , yi, inner_loop=100, outer_loop=10,)
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(x2, y2, c=z2, cmap=cm)
cb = plt.colorbar(sc,shrink=0.7, aspect=20*0.7)
cb.ax.set_title(r"$\frac{\partial E_\Theta^{g}}{\partial y_k}$", fontsize=20)
plt.ylabel(r"$y_k$", fontsize=15)
plt.xlabel(r"$y_i$", fontsize=15)
plt.savefig('/Users/dhruveshpatel/Downloads/scorenn-paper-assets/grads_positive_rel_dvn_1_55.pdf')
plt.show()
