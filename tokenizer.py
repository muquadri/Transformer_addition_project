# Tokenization helpers for the addition model.
# English and Urdu digits share the same token IDs.

URDU_TO_ENGLISH = {
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
}

VOCAB = {str(i): i for i in range(10)}
VOCAB['+'] = 10
VOCAB['-'] = 11
VOCAB['='] = 12

PAD_TOKEN_ID = 13
VOCAB['<PAD>'] = PAD_TOKEN_ID

for _urdu, _eng in URDU_TO_ENGLISH.items():
    VOCAB[_urdu] = VOCAB[_eng]

VOCAB_SIZE = 14
MAX_LEN = 8  # longest expression: '-99+-99='

# Class IDs for token roles
CLASS_NAMES = [
    'Sign_1',        # 0 — '+' or '-' before first number
    'Operand_1',     # 1 — digits of first number
    'Plus Operator', # 2 — the '+' between operands
    'Sign_2',        # 3 — '+' or '-' before second number
    'Operand_2',     # 4 — digits of second number
    'Equals',        # 5
    'PAD',           # 6
]
PAD_CLASS_ID = 6
NUM_CLASSES = len(CLASS_NAMES)


def tokenize(expression):
    return [VOCAB[ch] for ch in expression]


def normalize_to_english(ch):
    return URDU_TO_ENGLISH.get(ch, ch)


def display_role(ch, class_id):
    # '+' or '-' misclassified as a digit — override it
    if ch == '+' and class_id in (1, 4):
        return 'Plus Operator'
    if ch == '-' and class_id in (1, 4):
        return 'Minus Operator'
    # sign classes: check actual character
    if class_id in (0, 3):
        return 'Minus Operator' if ch == '-' else 'Plus Operator'
    return CLASS_NAMES[class_id]
