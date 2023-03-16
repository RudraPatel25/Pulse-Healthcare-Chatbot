import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('health-data.json','r') as f:
    diseases = json.load(f)

all_words = []
names = []
xy = []

for disease in diseases['diseases']:
    name = disease['name']
    names.append(name)
    for symptom in disease['symptoms']:
        w = tokenize(symptom)
        all_words.extend(w)
        xy.append((w, name))

ignore_words = [',', '.', '?', '!', ':', ';', '-', '(', ')','aa', 'ab', 'ad', 'ae', 'ag', 'ah', 'ai', 'al', 'am', 'an',
                'ar', 'as', 'at', 'aw', 'ax', 'ay', 'ba', 'be', 'bi', 'bo', 'by', 'de', 'ed', 'ef', 'eh', 'el', 'em',
                'en', 'er', 'es', 'et', 'ex', 'fa', 'fe', 'gi', 'go', 'ha', 'he', 'hm', 'ho', 'id', 'if', 'in',
                'jo', 'ka', 'ki', 'la', 'li', 'lo', 'ma', 'mi', 'mm', 'mo', 'mu', 'my', 'na', 'ne', 'no', 'nu', 'od', 
                'oe', 'of', 'oh', 'oi', 'om', 'on', 'op', 'or', 'os', 'ow', 'ox', 'oy', 'pa', 'pe', 'pi', 'po', 'qi',
                're', 'sh', 'si', 'so', 'ta', 'ti', 'to', 'uh', 'um', 'un', 'up', 'us', 'ut', 'we', 'wo', 'xi', 'xu',
                'ya', 'ye', 'yo', 'za','the']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
names = sorted(set(names))

X_train = []
y_train = []

for (symptom_sentences, name) in xy:
    bag = bag_of_words(symptom_sentences, all_words)
    X_train.append(bag)

    label = names.index(name)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
#Hyperparamters
batch_size = 128
hidden_size = 128
output_size = len(names)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 2000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "names": names
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
