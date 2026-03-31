import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AdditionDataset
from model import AdditionTransformer
from tokenizer import NUM_CLASSES, PAD_CLASS_ID

MODEL_PATH = 'model.pth'


def train(epochs=10, batch_size=256, lr=1e-3):
    device = torch.device('cpu')

    dataset = AdditionDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"Dataset: {len(dataset):,} samples  |  Batches/epoch: {len(loader)}")

    model = AdditionTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_CLASS_ID)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : {n_params:,} trainable parameters\n")

    header = f"{'Epoch':<8}{'Loss':<14}{'Token Accuracy'}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_seqs = 0
        total_seqs = 0

        for tokens, labels, pad_mask, _, _ in loader:
            tokens    = tokens.to(device)
            labels    = labels.to(device)
            pad_mask  = pad_mask.to(device)

            logits = model(tokens, pad_mask)
            loss = criterion(logits.view(-1, NUM_CLASSES), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            real_mask = ~pad_mask
            correct_seqs += ((preds == labels) & real_mask).sum().item()
            total_seqs += real_mask.sum().item()

        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct_seqs / total_seqs
        print(f"{epoch:<8}{avg_loss:<14.6f}{accuracy:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to '{MODEL_PATH}'. Time to test it using predict.py.")


if __name__ == '__main__':
    train()
