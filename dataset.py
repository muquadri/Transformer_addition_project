import csv
import os
import torch
from torch.utils.data import Dataset
from tokenizer import tokenize, PAD_TOKEN_ID, PAD_CLASS_ID, MAX_LEN

CSV_FILE = 'dataset.csv'


def build_expression(a, b, explicit_plus1=False, explicit_plus2=False):
    sign1_str = '-' if a < 0 else ('+' if explicit_plus1 else '')
    sign2_str = '-' if b < 0 else ('+' if explicit_plus2 else '')
    num1_str = str(abs(a))
    num2_str = str(abs(b))

    expr = f"{sign1_str}{num1_str}+{sign2_str}{num2_str}="

    labels = []
    if sign1_str:
        labels.append(0)
    for _ in num1_str:
        labels.append(1)
    labels.append(2)
    if sign2_str:
        labels.append(3)
    for _ in num2_str:
        labels.append(4)
    labels.append(5)

    return expr, labels


def generate_csv(path=CSV_FILE):
    rows = 0
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Expression', 'Operand_1', 'Operand_2', 'Output'])
        for a in range(-99, 100):
            for b in range(-99, 100):
                # Standard form: no explicit + sign
                expr, _ = build_expression(a, b)
                writer.writerow([expr, a, b, a + b])
                rows += 1
                # also write the explicit-plus variant for non-negative operands
                p1, p2 = a >= 0, b >= 0
                if p1 or p2:
                    expr2, _ = build_expression(a, b, explicit_plus1=p1, explicit_plus2=p2)
                    writer.writerow([expr2, a, b, a + b])
                    rows += 1
    print(f"Dataset saved to {path} ({rows:,} rows)")


class AdditionDataset(Dataset):

    def __init__(self, path=CSV_FILE):
        # Regenerate if the file uses the old padded-to-fixed-width format.
        if os.path.exists(path):
            old_format = False
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or 'Expression' not in reader.fieldnames:
                    old_format = True
                else:
                    for i, row in enumerate(reader):
                        if i >= 200:
                            break
                        if len(row.get('Expression', '')) >= 8:
                            old_format = True
                            break
            if old_format:
                print(f"Old '{path}' detected. Regenerating...")
                os.remove(path)

        if not os.path.exists(path):
            print(f"'{path}' not found. Generating...")
            generate_csv(path)

        self.samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                a = int(row['Operand_1'])
                b = int(row['Operand_2'])
                expr, labels = build_expression(a, b)
                tokens = tokenize(expr)
                seq_len = len(tokens)

                pad_len = MAX_LEN - seq_len
                tokens = tokens + [PAD_TOKEN_ID] * pad_len
                labels = labels + [PAD_CLASS_ID] * pad_len
                pad_mask = [False] * seq_len + [True] * pad_len

                self.samples.append((tokens, labels, pad_mask, a, b))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, labels, pad_mask, a, b = self.samples[idx]
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(pad_mask, dtype=torch.bool),
            a,
            b,
        )
