import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from prose.utils import collate_seq2seq
from torch.nn.utils.rnn import PackedSequence
from prose.utils import pack_sequences, unpack_sequences, collate_emb
from torch import nn
from prose.ALLY.models import LogisticRegressionModel, LinearMapping, DTIPooling, AAVNetwork
from prose.models.lstm import SkipLSTM
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, average_precision_score
from scipy.stats import spearmanr


class SSPClassifier:
    def __init__(self, model, use_cuda, train, test, opts):
        self.model = model.cuda() 
        self.use_cuda = use_cuda
        self.train_data = train
        self.test_data = test
        self.opts = opts

        input_dim = SkipLSTM(21, 21, 512, 3).get_embedding_dim()
        self.classifier = LogisticRegressionModel(input_dim=input_dim, num_classes=self.opts['nclass']).cuda()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.opts['lr'], weight_decay=self.opts['weight_decay'])
        self.num_epochs = self.opts['n']

    def train(self):
        self.model.eval()
        self.classifier.train()
        loader = DataLoader(self.train_data, batch_size=self.opts['batch_size'], collate_fn=collate_emb)

        for i in range(self.num_epochs):
            iterator = iter(loader)
            total_loss = 0.
            total_correct = 0
            n = 0
            for j in range(len(loader)):
                x, unpacked_y, _, order = next(iterator)
                if self.use_cuda: x = PackedSequence(x.data.cuda(), x.batch_sizes)
                _, emb = self.model(x)
                emb = emb.detach()
                packed_emb = PackedSequence(emb, x.batch_sizes)  
                unpacked_emb = unpack_sequences(packed_emb, order) # unpack emb [seq_len, 3093]
                padded_emb = pad_sequence(unpacked_emb, batch_first=True)
                padded_y = pad_sequence(unpacked_y, batch_first=True, padding_value=-1)

                self.optimizer.zero_grad()
                logits = self.classifier(padded_emb) # logits: [batch_size, padded_seq_len, 3], x: [batch_size, padded_seq_len, 3093], y: [batch_size, padded_seq_len]

                # faltten logits and labels
                logits = logits.view(-1, self.opts['nclass']) # [batch_size*padded_seq_len, 3
                padded_y = padded_y.view(-1)

                # mask padding positions
                logits = logits[padded_y!=-1]
                y = padded_y[padded_y!=-1].cuda()
                loss = nn.CrossEntropyLoss()(logits, y)
                loss.backward()
                pred = torch.argmax(logits, dim=1)
                self.optimizer.step()
                total_loss += loss.item()
                total_correct += (pred == y).sum().item()
                n += len(y)

            print(f"Epoch [{i+1}/{self.num_epochs}], Training Loss: {total_loss / len(loader):.4f}, Training Accuracy: {total_correct / n:.4f}")

    def predict(self):
        self.model.eval()
        self.classifier.eval()
        loader = DataLoader(self.test_data, batch_size=self.opts['batch_size'], collate_fn=collate_emb)
        iterator = iter(loader)
        total_correct = 0
        n = 0
        with torch.no_grad():
            for i in range(len(loader)):
                x, unpacked_y, _, order = next(iterator)
                if self.use_cuda: x = PackedSequence(x.data.cuda(), x.batch_sizes)
                _, emb = self.model(x)
                emb = emb.detach()
                packed_emb = PackedSequence(emb, x.batch_sizes)  
                unpacked_emb = unpack_sequences(packed_emb, order) # [batch_size, seq_len, 3093]
                padded_emb = pad_sequence(unpacked_emb, batch_first=True)
                padded_y = pad_sequence(unpacked_y, batch_first=True, padding_value=-1)
                logits = self.classifier(padded_emb)

                # faltten logits and labels
                logits = logits.view(-1, self.opts['nclass'])
                padded_y = padded_y.view(-1)

                # mask padding positions
                logits = logits[padded_y!=-1]
                y = padded_y[padded_y!=-1].cuda()
                pred = torch.argmax(logits, dim=1)
                total_correct += (pred == y).sum().item()
                n += len(y)
        print(f'Testing Accuracy: {total_correct/n:.4f}')

class SCLClassifier:
    def __init__(self, model, datamodule, opts):
        self.model = model.cuda() 
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = datamodule.val_dataloader()
        self.test_loader = datamodule.test_dataloader()
        self.opts = opts

        input_dim = SkipLSTM(21, 21, 512, 3).get_embedding_dim()
        self.classifier = LogisticRegressionModel(input_dim=input_dim, num_classes=self.opts['nclass']).cuda()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.opts['lr'], weight_decay=self.opts['weight_decay'])
        self.num_epochs = self.opts['n']

        self.use_cuda = (self.opts['device']!= -1) and torch.cuda.is_available()

    def validate(self):
        self.model.eval()
        self.classifier.eval()
        iterator = iter(self.val_loader)

        total_loss = 0.
        total_correct = 0
        n = 0

        with torch.no_grad():
            for i in range(len(self.val_loader)):
                x, y, _, order = next(iterator)
                if self.use_cuda: x = PackedSequence(x.data.cuda(), x.batch_sizes)
                else: x = PackedSequence(x.data, x.batch_sizes)
                _, emb = self.model(x)
                emb = emb.detach()
                packed_emb = PackedSequence(emb, x.batch_sizes)  
                unpacked_emb = unpack_sequences(packed_emb, order) # [batch_size, seq_len, 3093]
                emb_mean = torch.stack([x.mean(dim=0) for x in unpacked_emb]) # [batch_size, 3093]

                logits = self.classifier(emb_mean) # [batch_size, num_classes]
                y = torch.tensor(y).cuda()

                loss = nn.CrossEntropyLoss()(logits, y)
                pred = torch.argmax(logits, dim=1)
                total_loss += loss.item()
                total_correct += (pred == y).sum().item()
                n += len(y)

        return total_loss/len(self.val_loader), total_correct/n

    def train(self):
        self.model.eval()
        self.classifier.train()

        for i in range(self.num_epochs):
            total_loss = 0.
            total_correct = 0
            n = 0
            iterator = iter(self.train_loader)

            for j in range(len(self.train_loader)):
                x, y, _, order = next(iterator)
                if self.use_cuda: x = PackedSequence(x.data.cuda(), x.batch_sizes)
                else: x = PackedSequence(x.data, x.batch_sizes)
                _, emb = self.model(x)
                emb = emb.detach()
                packed_emb = PackedSequence(emb, x.batch_sizes)  
                unpacked_emb = unpack_sequences(packed_emb, order) # [batch_size, seq_len, 3093]
                emb_mean = torch.stack([x.mean(dim=0) for x in unpacked_emb]) # [batch_size, 3093]
                self.optimizer.zero_grad()

                logits = self.classifier(emb_mean) # [batch_size, num_classes]
                y = torch.tensor(y).cuda()

                loss = nn.CrossEntropyLoss()(logits, y)
                loss.backward()
                pred = torch.argmax(logits, dim=1)
                self.optimizer.step()

                total_loss += loss.item()
                total_correct += (pred == y).sum().item()
                n += len(y)

            print(f"Epoch [{i+1}/{self.num_epochs}], Training Loss: {total_loss / len(self.train_loader):.4f}, Training Accuracy: {total_correct / n:.4f}")

            val_loss, val_acc = self.validate()
            print(f"Epoch [{i+1}/{self.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


    def predict(self):
        self.model.eval()
        self.classifier.eval()
        iterator = iter(self.test_loader)

        total_correct = 0
        n = 0

        with torch.no_grad():
            for i in range(len(self.test_loader)):
                x, y, _, order = next(iterator)
                if self.use_cuda: x = PackedSequence(x.data.cuda(), x.batch_sizes)
                else: x = PackedSequence(x.data, x.batch_sizes)
                _, emb = self.model(x)
                emb = emb.detach()
                packed_emb = PackedSequence(emb, x.batch_sizes)  
                unpacked_emb = unpack_sequences(packed_emb, order) # [batch_size, seq_len, 3093]
                emb_mean = torch.stack([x.mean(dim=0) for x in unpacked_emb]) # [batch_size, 3093]

                logits = self.classifier(emb_mean) # [batch_size, num_classes]
                y = torch.tensor(y).cuda()

                pred = torch.argmax(logits, dim=1)
                total_correct += (pred == y).sum().item()
                n += len(y)

            print(f'Testing Accuracy: {total_correct/n:.4f}')


class DTIClassifier:
    def __init__(self, model, datamodule, drug_dim, opts):
        self.model = model.cuda() 
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = datamodule.val_dataloader()
        self.test_loader = datamodule.test_dataloader()
        self.opts = opts

        target_dim = SkipLSTM(21, 21, 512, 3).get_embedding_dim()
        self.classifier = DTIPooling(drug_dim, target_dim, self.opts['latent_dim']).cuda()
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.opts['lr'], weight_decay=self.opts['weight_decay'])
        self.num_epochs = self.opts['n']

        self.use_cuda = (self.opts['device']!= -1) and torch.cuda.is_available()

    def validate(self):
        self.model.eval()
        self.classifier.eval()

        with torch.no_grad():
            total_loss = 0.
            total_correct = 0
            n = 0
            all_preds = []
            all_labels = []
            iterator = iter(self.val_loader)

            for j in range(len(self.val_loader)):
                drugs, targets, labels = next(iterator) # drug: [batch_size, 2048], target: [batch_size, 3093], label: [batch_size]
                outputs = self.classifier(drugs, targets) 
                loss = nn.BCELoss()(outputs.cpu(), labels.float())
                total_loss += loss.item()
                preds = (outputs>0.5).long() # [batch_size]
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            val_acc = accuracy_score(all_labels, all_preds)
            val_aupr = average_precision_score(all_labels, all_preds)
            val_loss = total_loss / len(self.val_loader)
        return val_loss, val_acc, val_aupr

    def train(self):
        self.model.eval()
        self.classifier.train()

        for i in range(self.num_epochs):
            total_loss = 0.
            all_preds = []
            all_labels = []
            iterator = iter(self.train_loader)

            for j in range(len(self.train_loader)):
                drugs, targets, labels = next(iterator) # drug: [batch_size, 2048], target: [batch_size, 3093], label: [batch_size]
                outputs = self.classifier(drugs, targets) # outputs: [batch_size]
                labels = labels.float().cuda()
                loss = nn.BCELoss()(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = (outputs>0.5).long() # [batch_size]
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            train_acc = accuracy_score(all_labels, all_preds)
            train_aupr = average_precision_score(all_labels, all_preds)

            val_loss, val_acc, val_aupr = self.validate()

            print(f"Epoch [{i+1}/{self.num_epochs}], Training Loss: {total_loss / len(self.train_loader):.4f}, Training Accuracy: {train_acc:.4f}, Training AUPR: {train_aupr:.4f}")
            print(f"Epoch [{i+1}/{self.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation AUPR: {val_aupr:.4f}")

    def predict(self):
        self.model.eval()
        self.classifier.eval()

        with torch.no_grad():
            total_correct = 0
            all_preds = []
            all_labels = []
            iterator = iter(self.test_loader)

            for j in range(len(self.test_loader)):
                drugs, targets, labels = next(iterator) # drug: [batch_size, 2048], target: [batch_size, 3093], label: [batch_size]
                outputs = self.classifier(drugs, targets) 
                preds = (outputs>0.5).long() # [batch_size]
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            test_acc = accuracy_score(all_labels, all_preds)
            test_aupr = average_precision_score(all_labels, all_preds)
            print(f'Testing Accuracy: {test_acc:.4f}, Testing AUPR: {test_aupr:.4f}')

class AAVRegressor:
    def __init__(self, model, use_cuda, train, validation, test, opts):
        self.model = model.cuda() 
        self.use_cuda = use_cuda
        self.train_data = train
        self.val_data = validation
        self.test_data = test
        self.opts = opts

        input_dim = SkipLSTM(21, 21, 512, 3).get_embedding_dim()
        self.regressor = AAVNetwork(input_dim=input_dim).cuda()
        self.optimizer = torch.optim.Adam(self.regressor.parameters(), lr=self.opts['lr'], weight_decay=self.opts['weight_decay'])
        self.num_epochs = self.opts['n']

    def validate(self):
        self.model.eval()
        self.regressor.eval()
        loader = DataLoader(self.val_data, batch_size=self.opts['batch_size'], collate_fn=collate_emb)

        with torch.no_grad():
            total_loss = 0.
            iterator = iter(loader)

            for j in range(len(loader)):
                x, unpacked_y, _, order = next(iterator)
                if self.use_cuda: x = PackedSequence(x.data.cuda(), x.batch_sizes)
                _, emb = self.model(x)
                emb = emb.detach()
                packed_emb = PackedSequence(emb, x.batch_sizes)  
                unpacked_emb = unpack_sequences(packed_emb, order) # [[seq_len, 3093], ..., [seq_len, 3093]], len=batch_size
                emb_mean = torch.stack([x.mean(dim=0) for x in unpacked_emb]) # [batch_size, 3093]

                unpacked_y = torch.tensor(unpacked_y, dtype=torch.float32).cuda() # len=batch_size

                preds = self.regressor(emb_mean).squeeze(1) # [batch_size]

                loss = nn.MSELoss()(preds, unpacked_y)
                total_loss += loss.item()
            val_loss = total_loss/len(loader)
        return val_loss

    def train(self):
        self.model.eval()
        self.regressor.train()
        loader = DataLoader(self.train_data, batch_size=self.opts['batch_size'], collate_fn=collate_emb)

        best_val_loss = 1000
        epoch_no_improve = 0
        for i in range(self.num_epochs):
            total_loss = 0.
            iterator = iter(loader)

            for j in range(len(loader)):
                x, unpacked_y, _, order = next(iterator)
                if self.use_cuda: x = PackedSequence(x.data.cuda(), x.batch_sizes)
                _, emb = self.model(x)
                emb = emb.detach()
                packed_emb = PackedSequence(emb, x.batch_sizes)  
                unpacked_emb = unpack_sequences(packed_emb, order) # [[seq_len, 3093], ..., [seq_len, 3093]], len=batch_size
                emb_mean = torch.stack([x.mean(dim=0) for x in unpacked_emb]) # [batch_size, 3093]

                unpacked_y = torch.tensor(unpacked_y, dtype=torch.float32).cuda() # len=batch_size

                self.optimizer.zero_grad()
                preds = self.regressor(emb_mean).squeeze(1) # [batch_size]

                loss = nn.MSELoss()(preds, unpacked_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            val_loss = self.validate()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                epoch_no_improve += 1
            if epoch_no_improve > self.opts['patience']:
                break
            print(f"Epoch [{i+1}/{self.num_epochs}], Training Loss: {total_loss / len(loader):.4f}, Validation Loss: {val_loss:.4f}")


    def predict(self):
        self.model.eval()
        self.regressor.eval()
        loader = DataLoader(self.test_data, batch_size=self.opts['batch_size'], collate_fn=collate_emb)

        all_preds = []
        all_targets = []
        iterator = iter(loader)

        with torch.no_grad():
            for j in range(len(loader)):
                x, unpacked_y, _, order = next(iterator)
                if self.use_cuda: x = PackedSequence(x.data.cuda(), x.batch_sizes)
                _, emb = self.model(x)
                emb = emb.detach()
                packed_emb = PackedSequence(emb, x.batch_sizes)  
                unpacked_emb = unpack_sequences(packed_emb, order) # [[seq_len, 3093], ..., [seq_len, 3093]], len=batch_size
                emb_mean = torch.stack([x.mean(dim=0) for x in unpacked_emb]) # [batch_size, 3093]

                unpacked_y = torch.tensor(unpacked_y, dtype=torch.float32).cuda() # len=batch_size

                preds = self.regressor(emb_mean).squeeze(1) # [batch_size]
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(unpacked_y.cpu().numpy())
            test_cor, _ = spearmanr(all_preds, all_targets)
            print(f"Testing Spearman's Correlation Coefficient: {test_cor:.4f}")

            