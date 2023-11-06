from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import *

class Dataset():
    def __init__(self, x, y):
        self.y = y
        self.impute_by_col()
        self.x = x
        self.n = x.shape[0]

    def __len__(self):
        return self.n
              
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return torch.Tensor(x), torch.Tensor(y)
    
    def impute_by_col(self):
        y_means = np.nanmean(self.y, axis=0)
        idx = np.where(np.isnan(self.y))
        self.y[idx] = np.take(y_means, idx[1])

class Parametrized_Net(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size, output_size, act_fct=nn.ReLU(), dropout=0.5):
        super(Parametrized_Net, self).__init__()
        self.activation = act_fct
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                in_feats, out_feats = input_size, hidden_size
            elif i == num_layers-1:
                in_feats, out_feats = hidden_size, output_size
            else:
                in_feats, out_feats = hidden_size, hidden_size
            if i != num_layers-1:
                self.layers.append(nn.Sequential(nn.Linear(in_feats, out_feats),
                                                 self.activation,
                                                 nn.Dropout(dropout)))
            else:
                self.layers.append(nn.Linear(in_feats, out_feats))
            self.add_module('layer'+str(i+1), self.layers[-1])
        
    def forward(self, x):        
        for layer in self.layers:
            x = layer(x)
        return x

def train(model, optimizer, criterion, device, train_loader):
    model.train()
    for idx, (xs, ys) in enumerate(train_loader):
        xs, ys = xs.to(device), ys.to(device)
        optimizer.zero_grad()
        outputs = model(xs)
        loss = criterion(outputs, ys)
        loss.backward()
        optimizer.step()
    return model

def test(model, device, test_loader):
    model.eval()
    y_pred = torch.Tensor().to(device)
    with torch.no_grad():
        for idx, (xs, _) in enumerate(test_loader):
            xs = xs.to(device)
            outputs = model(xs)
            y_pred = torch.cat((y_pred, outputs))
    y_pred = y_pred.numpy()
    return y_pred

solver = snakemake.rule.split('_')[-1]
trial = snakemake.config['trial']
imputation = eval(snakemake.wildcards['imputation'])
metric = snakemake.wildcards['metric']
x = pd.read_csv(snakemake.input[0], index_col=0).to_numpy().astype(float)
y = pd.read_csv(snakemake.input[1], index_col=0).to_numpy().astype(float)
if trial:
    y_min = np.nanmin(y, axis=0)
    y_max = np.nanmax(y, axis=0)
    y = (y - y_min)/(y_max - y_min)
y_split = np.load(snakemake.input[2])

# scale x columns between 0 and 1 (necessary for neural networks)
x_min = np.nanmin(x, axis=0)
x_max = np.nanmax(x, axis=0)
x = (x - x_min)/(x_max - x_min)

# y_train = outer training set, y_test = test set
x_train, x_test, y_train, y_test = train_test_split(x, y, y_split, imputation)
if snakemake.rule.startswith('train_and_validate'):
    y_split = np.load(snakemake.input[3])
    # y_train = inner training set, y_test = validation set
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, y_split, imputation)
    num_layers = int(snakemake.wildcards['nl'])
    hidden_size = int(snakemake.wildcards['hs'])
    learning_rate = float(snakemake.wildcards['lr'])
else:
    with open(snakemake.input[3], 'r') as f:
        hyperparams = json.load(f)
    num_layers = int(hyperparams['nl'])
    hidden_size = int(hyperparams['hs'])
    learning_rate = float(hyperparams['lr'])

# kick out datapoints with nan features
mask = np.all(np.isnan(x_train), axis=1)
x_train = x_train[~mask]
y_train = y_train[~mask]

_, d = x_train.shape
_, p = y_train.shape
batch_size = 32
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
np.random.seed(0)
criterion = nn.MSELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if solver == 'ANNind':
    input_size = d
    output_size = 1
    min_scores = np.zeros(p)
    for j in range(p):
        y_nan_indices = np.isnan(y_train[:, j])
        x_trainj = x_train[~y_nan_indices]
        y_trainj = y_train[:, j][~y_nan_indices][:, np.newaxis]
        dataset1 = Dataset(x_trainj, y_trainj)
        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
        dataset2 = Dataset(x_test, x_test) # y is not needed, so we just use x_test for an array of the same shape
        test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)
        
        model = Parametrized_Net(num_layers, hidden_size, input_size, output_size).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95)

        num_epochs = 10
        threshold = 10
        counter = 0
        epochs = 0
        min_score = float('inf')
        while counter < threshold: # break if loss does not improve after num_epochs*threshold epochs
            for i in range(num_epochs):
                model = train(model, optimizer, criterion, device, train_loader)
            epochs += num_epochs
            if snakemake.rule.startswith('test'):
                y_pred = test(model, device, train_loader)
                score = metric_score(y_trainj, y_pred, metric)
            else:
                y_pred = test(model, device, test_loader)
                for j1 in range(j):
                    y_pred = np.hstack((np.zeros((y_pred.shape[0], 1)), y_pred))
                for j2 in range(j+1, p):
                    y_pred = np.hstack((y_pred, np.zeros((y_pred.shape[0], 1))))
                score = score(y_test, y_pred, y_split, imputation, metric, current_j=j)[j]
            if score == float('inf'): # if learning rate too high, loss becomes nan, y_pred too, function.py returns inf
                min_score = float('inf')
                break
            if score == np.nan: # if validation/test set is empty, function.py returns nan
                min_score = np.nan
                break
            if score < min_score:
                min_score = score
                counter = 0
            else:
                counter += 1
        min_scores[j] = min_score
        # print(f'trained for {epochs} epochs')
        if snakemake.rule.startswith('test'):
            y_pred = test(model, device, test_loader)
            for j1 in range(j):
                y_pred = np.hstack((np.zeros((y_pred.shape[0], 1)), y_pred))
            for j2 in range(j+1, p):
                y_pred = np.hstack((y_pred, np.zeros((y_pred.shape[0], 1))))
            min_scores[j] = score(y_test, y_pred, y_split, imputation, metric, current_j=j)[j]
else:
    dataset1 = Dataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
    dataset2 = Dataset(x_test, x_test) # y is not needed, so we just use x_test for an array of the same shape
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

    input_size = d
    output_size = p
    model = Parametrized_Net(num_layers, hidden_size, input_size, output_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95)

    num_epochs = 10
    threshold = 10
    counter = 0
    epochs = 0
    min_mean_score = float('inf')
    min_scores = None
    while counter < threshold: # break if loss does not improve after num_epochs*threshold epochs
        for i in range(num_epochs):
            model = train(model, optimizer, criterion, device, train_loader)
        epochs += num_epochs
        if snakemake.rule.startswith('test'):
            y_pred = test(model, device, train_loader)
            scores = metric_score(y_train, y_pred, metric)
        else:
            y_pred = test(model, device, test_loader)
            scores = score(y_test, y_pred, y_split, imputation, metric)
        mean_score = np.nanmean(scores)
        if mean_score == float('inf'): # if learning rate too high, loss becomes nan, y_pred too, function.py returns inf
            min_scores = np.full(output_size, float('inf'))
            break
        if mean_score == np.nan: # if validation/test set is empty, function.py returns nan
            min_scores = np.full(output_size, np.nan)
            break
        if mean_score < min_mean_score:
            min_scores = scores
            min_mean_score = mean_score
            counter = 0
        else:
            counter += 1
    if snakemake.rule.startswith('test'):
        y_pred = test(model, device, test_loader)
        min_scores = score(y_test, y_pred, y_split, imputation, metric)
    # print(f'trained for {epochs} epochs')
np.save(snakemake.output[0], min_scores)
