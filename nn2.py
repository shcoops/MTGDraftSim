import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit as gss
from torch import nn, optim
import torch.nn.functional as F

# Reproducibility
seed = 4
np.random.seed(seed)
torch.manual_seed(seed)

# Import draft data and generate a list of card names for future use
data = pd.read_csv('draft-data.MID.PremierDraft.csv', usecols=np.r_[2, 7:12, 14:558], nrows=80000)
cards = list(map(lambda x: x.lstrip('pack_card_'), [x for x in data.columns if 'pack_card_' in x]))
card_to_int = dict((c, i) for i, c in enumerate(cards))
int_to_cards = dict((i, c) for i, c in enumerate(cards))


# Define a function to encode the 'pick' string as a one-hot vector
def pick_vector(y):
    integer_encoded = [card_to_int[card] for card in y]
    onehot = list()
    for x in integer_encoded:
        card = [0 for _ in range(len(cards))]
        card[x] = 1
        onehot.append(card)
    return onehot


# Divide into train, test, and validation sets, maintain draft integrity
train_i, split_i = next(gss(test_size=.20, n_splits=5, random_state=seed).split(data, groups=data['draft_id']))
remove = ['pick', 'draft_id', 'event_match_wins', 'event_match_losses']

train_feat = np.asarray(data.loc[train_i, data.columns.difference(remove)])
train_y = np.asarray(pick_vector(data.loc[train_i, 'pick']))

# Split the remaining 20% into test and validation
validation = data.loc[split_i, :].reset_index(drop=True)
test_i, val_i = next(gss(test_size=0.5, n_splits=5, random_state=seed).split(validation, groups=validation['draft_id']))

test_feat = np.asarray(validation.loc[test_i, validation.columns.difference(remove)])
test_y = np.asarray(pick_vector(validation.loc[test_i, 'pick']))

validation_feat = np.asarray(validation.loc[val_i, validation.columns.difference(remove)])
validation_y = np.asarray(pick_vector(validation.loc[val_i, 'pick']))

# Convert to tensor
X_train = torch.from_numpy(train_feat).float()
y_train = torch.from_numpy(train_y).float()

X_test = torch.from_numpy(test_feat).float()
y_test = torch.from_numpy(test_y).float()


class Net(nn.Module):

    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 365)
        self.fc2 = nn.Linear(365, 272)
        self.n_cards = (n_features - 2) / 2

    def forward(self, x):
        pack = x[:, :int(self.n_cards)]
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return pack * x


net = Net(X_train.shape[1])

criterion = nn.BCELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)

# Optimize for GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
net = net.to(device)
criterion = criterion.to(device)


# Accuracy is calculated as max index in y_predi matching the max index in y_true
def calculate_accuracy(y_true, y_predi):
    pred = torch.argmax(y_predi, dim=1)
    return len(torch.where((pred == torch.argmax(y_true, dim=1)))[0]) / y_true.shape[0]


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)


# Network code
for epoch in range(400+1):

    y_pred = net(X_train)

    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)

    if epoch % 50 == 0:
        train_acc = calculate_accuracy(y_train, y_pred)

        y_test_pred = net(X_test)
        y_test_pred = torch.squeeze(y_test_pred)

        test_loss = criterion(y_test_pred, y_test)

        test_acc = calculate_accuracy(y_test, y_test_pred)
        print(
            f'''epoch {epoch}
        Train set - loss: {round_tensor(train_loss)}, accuracy: {train_acc}
        Test  set - loss: {round_tensor(test_loss)}, accuracy: {test_acc}
        ''')
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

# Print test data to CSV
export = validation.loc[test_i, ['pack_number', 'pick_number', 'pick']]
hits = torch.where(torch.argmax(y_test_pred, dim=1) == torch.argmax(y_test, dim=1))[0]
hits = hits.cpu().detach().numpy()
zeros = np.zeros(y_test.size(dim=0))
zeros[hits] = 1
export['hits'] = zeros
export.to_csv('pickhits1')