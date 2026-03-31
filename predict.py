import re
import math
import torch
from tokenizer import tokenize, normalize_to_english, CLASS_NAMES, PAD_TOKEN_ID, MAX_LEN, display_role
from model import AdditionTransformer

MODEL_PATH = 'model.pth'


def load_model():
    model = AdditionTransformer()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
    model.eval()
    return model



def clean_input(user_input):
    cleaned = ''.join(normalize_to_english(ch) for ch in user_input.strip())
    cleaned = cleaned.replace(' ', '').rstrip('=') + '='

    # subtraction is not supported
    if re.fullmatch(r'[+-]?\d{1,2}-\d{1,2}=', cleaned):
        raise ValueError("Perform two digit addition only")

    # validate full pattern
    if not re.fullmatch(r'[+-]?\d{1,2}\+[+-]?\d{1,2}=', cleaned):
        raise ValueError(
            f"Cannot understand '{user_input}'.\n"
            f"  Please use:  20+30=   or   +20++30=   or   -20+-30=   or   +20+-30="
        )
    return cleaned


def predict(user_input, model):
    expr = clean_input(user_input)
    chars = list(expr)
    tokens = tokenize(expr)
    seq_len = len(tokens)

    pad_len       = MAX_LEN - seq_len
    padded_tokens = tokens + [PAD_TOKEN_ID] * pad_len
    pad_mask      = [False] * seq_len + [True] * pad_len

    x = torch.tensor([padded_tokens], dtype=torch.long)
    mask = torch.tensor([pad_mask], dtype=torch.bool)

    with torch.no_grad():
        logits = model(x, mask)
        predictions = logits.argmax(dim=-1)[0].tolist()

    # Print the per-token role table.
    print(f"\n{'=' * 50}")
    print(f"  Input   : {user_input}")
    print(f"  Cleaned : {expr}")
    print(f"{'=' * 50}")
    print(f"  {'Pos':<5} {'Token':<10} {'Predicted Role'}")
    print(f"  {'-' * 38}")
    for pos in range(seq_len):
        role_label = display_role(chars[pos], predictions[pos])
        print(f"  {pos:<5} {repr(chars[pos]):<10} {role_label}")

    # Rebuild both operands from the predicted roles.
    sign1_char = ''
    operand1_chars = []
    sign2_char = ''
    operand2_chars = []

    for pos in range(seq_len):
        ch = chars[pos]
        role = predictions[pos]
        if role == 0:
            sign1_char = ch
        elif role == 1:
            operand1_chars.append(ch)
        elif role == 3:
            sign2_char = ch
        elif role == 4:
            operand2_chars.append(ch)

    if not operand1_chars or not operand2_chars:
        raise ValueError("Model could not identify operands — try retraining.")

    a = int(''.join(operand1_chars))
    if sign1_char == '-':
        a = -a

    b = int(''.join(operand2_chars))
    if sign2_char == '-':
        b = -b

    result = int(math.fsum([a, b]))

    print(f"\n  Operand 1 : {a}")
    print(f"  Operand 2 : {b}")
    print(f"  Result    : {a} + {b} = {result}")
    return result


if __name__ == '__main__':
    _model = load_model()
    print("Model loaded successfully\n")
    print("Enter an addition expression. Examples:")
    print("   20+30=        -> 50")
    print("  -20+30=        -> 10")
    print("  -20+-30=       -> -50")
    print("  +20++30=       -> 50")
    print("   ۲۰+۳۰=        -> 50")
    print("\nType 'quit' to exit.\n")

    while True:
        user_input = input("Expression: ").strip()
        if user_input.lower() == 'quit':
            break
        if not user_input:
            continue
        try:
            predict(user_input, _model)
        except (ValueError, KeyError) as e:
            print(f"  Error: {e}")
