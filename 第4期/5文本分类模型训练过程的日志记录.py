#task-start
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange
import warnings
warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

class TextClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, nhead=4, num_encoder_layers=2, num_classes=2):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead),
            num_layers=num_encoder_layers
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text):

        embedded = self.embedding(text)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output[0])
        return output

def get_data_loaders():
    data, labels = pickle.load(open('text_classify_training_data.pkl', 'rb'))

    dataset = TensorDataset(data, labels)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    return train_loader, val_loader

def train(model, iterator, criterion, optimizer):
    model.train()
    total_loss = 0

    for text, label in iterator:
        optimizer.zero_grad()
        outputs = model(text.transpose(0, 1))
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    total_loss = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    with torch.no_grad():
        for text, label in iterator:
            outputs = model(text.transpose(0, 1))
            loss = criterion(outputs, label)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            tp = np.sum((predicted[i] == 1 and label[i] == 1) for i in range(len(predicted))).item()
            tn = np.sum((predicted[i] == 0 and label[i] == 0) for i in range(len(predicted))).item()
            fp = np.sum((predicted[i] == 1 and label[i] == 0) for i in range(len(predicted))).item()
            fn = np.sum((predicted[i] == 0 and label[i] == 1) for i in range(len(predicted))).item()

            TP += tp
            TN += tn
            FP += fp
            FN += fn
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = tp / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


    return total_loss / len(iterator), accuracy, precision, recall, f1

def run():
    model = TextClassifier()
    train_loader, val_loader = get_data_loaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    NUM_EPOCHS = 10
    for epoch in trange(NUM_EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy, precision, recall, f1 = evaluate(model, val_loader, criterion)

        import logging
        logging.basicConfig(filename='training.log', 
                            level=logging.INFO,
                            format='%(message)s')
        logging.info(f"Epoch: {epoch+1}")
        logging.info(f"Train Loss: {train_loss:.3f}")
        logging.info(f"Val Loss: {val_loss:.3f} | Val Accuracy: {(val_accuracy*100):.2f}% | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")


    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    run()
#task-end