import math
import re
import time

import streamlit as st
import torch

from model import AdditionTransformer
from tokenizer import CLASS_NAMES, MAX_LEN, PAD_TOKEN_ID, normalize_to_english, tokenize, display_role

MODEL_PATH = "model.pth"
INPUT_PATTERN = re.compile(r"([+-]?\d{1,2})\+([+-]?\d{1,2})=")


# Input validation

def clean_input(user_input):
    cleaned = "".join(normalize_to_english(ch) for ch in user_input.strip())
    cleaned = cleaned.replace(" ", "").rstrip("=") + "="

    if re.fullmatch(r"[+-]?\d{1,3}-\d{1,3}=", cleaned):
        raise ValueError("Must use addition")

    if re.search(r"\d{3,}", cleaned):
        raise ValueError("Use two digit addition")

    if not INPUT_PATTERN.fullmatch(cleaned):
        raise ValueError(
            "Use one expression like: 20+30=   -20+30=   -20+-30=   +5++95="
        )
    return cleaned


# Ground truth labels (used for match comparison in the UI)

def ground_truth_labels(cleaned_expr):
    match = INPUT_PATTERN.fullmatch(cleaned_expr)
    if match is None:
        raise ValueError("Invalid expression after cleaning")

    num1, num2 = match.groups()
    labels = []

    if num1[0] in ('+', '-'):
        labels.append(0)
    labels.extend([1] * len(num1.lstrip('+-')))
    labels.append(2)
    if num2[0] in ('+', '-'):
        labels.append(3)
    labels.extend([4] * len(num2.lstrip('+-')))
    labels.append(5)

    return labels


# Operand extraction from predicted roles

def extract_operands(chars, predictions):
    sign1_char = ""
    sign2_char = ""
    operand1_chars = []
    operand2_chars = []

    for idx, role in enumerate(predictions):
        ch = chars[idx]
        if role == 0:
            sign1_char = ch
        elif role == 1:
            operand1_chars.append(ch)
        elif role == 3:
            sign2_char = ch
        elif role == 4:
            operand2_chars.append(ch)

    if not operand1_chars or not operand2_chars:
        raise ValueError("The model could not identify both operands")

    a = int("".join(operand1_chars))
    b = int("".join(operand2_chars))
    if sign1_char == "-":
        a = -a
    if sign2_char == "-":
        b = -b
    return a, b


# Model loader

@st.cache_resource
def load_model():
    model = AdditionTransformer()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# Inference

def infer_expression(model, user_input):
    cleaned = clean_input(user_input)
    chars = list(cleaned)
    tokens = tokenize(cleaned)
    seq_len = len(tokens)

    pad_len = MAX_LEN - seq_len
    padded_tokens = tokens + [PAD_TOKEN_ID] * pad_len
    pad_mask = [False] * seq_len + [True] * pad_len

    x = torch.tensor([padded_tokens], dtype=torch.long)
    mask = torch.tensor([pad_mask], dtype=torch.bool)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(x, mask)
        probs = logits.softmax(dim=-1)
        predictions = logits.argmax(dim=-1)[0].tolist()
    latency_ms = (time.perf_counter() - start) * 1000.0

    gt = ground_truth_labels(cleaned)
    token_rows = []
    for pos in range(seq_len):
        role_id = predictions[pos]
        conf = float(probs[0, pos, role_id].item())
        ch = chars[pos]
        token_rows.append({
            "Position":       pos,
            "Token":          ch,
            "Predicted Role": display_role(ch, role_id),
            "Confidence":     round(conf * 100.0, 2),
            "Expected Role":  display_role(ch, gt[pos]),
            "Match":          role_id == gt[pos],
        })

    a_pred, b_pred = extract_operands(chars, predictions[:seq_len])
    result_pred = int(math.fsum([a_pred, b_pred]))

    match = INPUT_PATTERN.fullmatch(cleaned)
    a_true = int(match.group(1))
    b_true = int(match.group(2))
    result_true = a_true + b_true

    token_correct = sum(1 for i in range(seq_len) if predictions[i] == gt[i])
    token_acc = 100.0 * token_correct / seq_len
    avg_conf = sum(r["Confidence"] for r in token_rows) / seq_len

    return {
        "input":            user_input,
        "cleaned":          cleaned,
        "operand1":         a_pred,
        "operand2":         b_pred,
        "output":           result_pred,
        "expected_output":  result_true,
        "is_correct":       result_pred == result_true,
        "token_accuracy":   token_acc,
        "average_confidence": avg_conf,
        "latency_ms":       latency_ms,
        "rows":             token_rows,
    }


# Role → CSS color index

ROLE_COLOR_MAP = {
    "Minus Operator": 0,   # purple
    "Operand_1":      1,   # blue
    "Plus Operator":  2,   # yellow
    "Operand_2":      4,   # teal
    "Equals":         5,   # green
    "PAD":            6,   # grey
}

ROLE_COLORS = {
    "Minus Operator": ("#c4a0ff", "#1e1030"),   # purple
    "Operand_1":      ("#7eb4ff", "#0f1e36"),   # blue
    "Plus Operator":  ("#fbbf24", "#1a1408"),   # yellow
    "Operand_2":      ("#34d4ba", "#0d1f2a"),   # teal
    "Equals":         ("#4ade80", "#0f1e10"),   # green
    "PAD":            ("#444444", "#161616"),   # grey
}


# Styles

def inject_css() -> None:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: #0b0e14; color: #e6eaf2; }
        [data-testid="stAppViewContainer"] { background: #0b0e14; }
        h1,h2,h3,h4,h5,h6,p,label,span,div { color: #e6eaf2; }
        [data-testid="stMarkdownContainer"] p { color: #e6eaf2; }
        [data-testid="stMetricLabel"] { color: #7a8aaa !important; }
        [data-testid="stMetricValue"] { color: #e6eaf2 !important; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        [data-testid="stToolbar"] { display: none; }

        /* Hero */
        .hero-wrap {
            background: linear-gradient(135deg, #141924 0%, #1a2236 100%);
            border: 1px solid #2c3650;
            border-radius: 16px;
            padding: 1.6rem 2rem;
            margin-bottom: 1.4rem;
            display: flex;
            align-items: center;
            gap: 1.2rem;
        }
        .hero-icon  { font-size: 2.8rem; line-height: 1; }
        .hero-title { font-size: 1.7rem; font-weight: 700; margin: 0; color: #e6eaf2; }
        .hero-sub   { color: #7a8aaa; font-size: 0.93rem; margin: 0.3rem 0 0 0; }

        /* Section label */
        .section-title {
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.07em;
            color: #4f8cff;
            text-transform: uppercase;
            margin: 1.4rem 0 0.6rem 0;
            border-left: 3px solid #4f8cff;
            padding-left: 0.6rem;
        }

        /* Input */
        .input-label {
            font-size: 0.82rem; font-weight: 600;
            letter-spacing: 0.05em; color: #7a8aaa;
            text-transform: uppercase; margin-bottom: 0.35rem;
        }
        .stTextInput > div > div > input {
            background-color: #111620;
            color: #e6eaf2;
            border: 1.5px solid #2c3650;
            border-radius: 10px;
            min-height: 48px;
            font-size: 1.05rem;
            font-family: monospace;
        }
        .stTextInput > div > div > input:focus { border-color: #4f8cff !important; }

        /* Analyze button */
        .stButton > button {
            min-height: 48px;
            border-radius: 10px;
            background: linear-gradient(135deg, #3a6fff 0%, #5b9bff 100%);
            border: none;
            color: white;
            font-weight: 700;
            font-size: 1rem;
        }

        /* Result banner */
        .result-banner { border-radius: 14px; padding: 1.2rem 1.5rem; margin: 1.2rem 0; text-align: center; }
        .result-eq  { font-size: 2rem; font-weight: 700; letter-spacing: 0.04em; }
        .result-sub { font-size: 0.88rem; margin-top: 0.3rem; }
        .banner-ok  { background: #0d2318; border: 1px solid #1d5c38; }
        .banner-ok  .result-eq  { color: #4ade80; }
        .banner-ok  .result-sub { color: #52996e; }
        .banner-err { background: #261110; border: 1px solid #6b2020; }
        .banner-err .result-eq  { color: #f87171; }
        .banner-err .result-sub { color: #9b5050; }

        /* Token grid */
        .token-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.8rem 0 1.4rem 0; }
        .token-cell {
            border-radius: 10px; padding: 0.55rem 0.7rem;
            min-width: 80px; text-align: center; border: 1px solid transparent;
        }
        .token-char { font-size: 1.3rem; font-weight: 700; font-family: monospace; }
        .token-role { font-size: 0.68rem; margin-top: 0.2rem; letter-spacing: 0.02em; }
        .token-conf { font-size: 0.68rem; color: #7a8aaa; }

        .role-0 { background:#1e1030; border-color:#6b45c2; }
        .role-0 .token-char { color:#c4a0ff; } .role-0 .token-role { color:#8b6ccc; }
        .role-1 { background:#0f1e36; border-color:#2b5da8; }
        .role-1 .token-char { color:#7eb4ff; } .role-1 .token-role { color:#4a7fc4; }
        .role-2 { background:#1a1408; border-color:#7a5a10; }
        .role-2 .token-char { color:#fbbf24; } .role-2 .token-role { color:#a07820; }
        .role-3 { background:#1a1030; border-color:#6b2d8a; }
        .role-3 .token-char { color:#d97bff; } .role-3 .token-role { color:#9050b8; }
        .role-4 { background:#0d1f2a; border-color:#1a6070; }
        .role-4 .token-char { color:#34d4ba; } .role-4 .token-role { color:#2a8a7a; }
        .role-5 { background:#0f1e10; border-color:#2a6030; }
        .role-5 .token-char { color:#4ade80; } .role-5 .token-role { color:#2a8040; }
        .role-6 { background:#161616; border-color:#333; }
        .role-6 .token-char { color:#555;    } .role-6 .token-role { color:#444; }

        /* Legend */
        .legend-row { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.5rem; }
        .legend-chip {
            border-radius: 8px; padding: 0.22rem 0.7rem;
            font-size: 0.78rem; font-weight: 600;
        }

        /* History */
        .hist-row {
            background: #131929; border: 1px solid #1e2d4a;
            border-radius: 10px; padding: 0.6rem 1rem;
            display: flex; justify-content: space-between;
            align-items: center; margin-bottom: 0.45rem;
        }
        .hist-expr   { font-family: monospace; font-size: 1.0rem; color: #e6eaf2; }
        .hist-result { font-weight: 700; font-size: 1.05rem; }
        .hist-ok  { color: #4ade80; }
        .hist-err { color: #f87171; }
    </style>
    """, unsafe_allow_html=True)


# UI components

def render_token_grid(rows: list) -> None:
    cells = ""
    for row in rows:
        role_label = row["Predicted Role"]
        idx = ROLE_COLOR_MAP.get(role_label, 6)
        icon = "✓" if row["Match"] else "✗"
        cells += (
            f'<div class="token-cell role-{idx}">'
            f'<div class="token-char">{row["Token"]}</div>'
            f'<div class="token-role">{role_label}</div>'
            f'<div class="token-conf">{row["Confidence"]:.0f}% {icon}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="token-grid">{cells}</div>', unsafe_allow_html=True)


def render_legend() -> None:
    chips = ""
    for role, (color, bg) in ROLE_COLORS.items():
        if role == "PAD":
            continue
        chips += (
            f'<span class="legend-chip" '
            f'style="background:{bg};color:{color};border:1px solid {color}44;">'
            f'{role}</span>'
        )
    st.markdown(f'<div class="legend-row">{chips}</div>', unsafe_allow_html=True)


def render_history(history: list) -> None:
    if not history:
        return
    st.markdown('<div class="section-title">History</div>', unsafe_allow_html=True)
    for item in reversed(history[-6:]):
        css = "hist-ok" if item["ok"] else "hist-err"
        result_text = str(item["result"]) if item["ok"] else "error"
        st.markdown(
            f'<div class="hist-row">'
            f'<span class="hist-expr">{item["expr"]}</span>'
            f'<span class="hist-result {css}">{result_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# Main

def main() -> None:
    st.set_page_config(page_title="Transformer Addition", page_icon="🧠", layout="wide")
    inject_css()

    # Hero
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-icon">🧠</div>
        <div>
            <p class="hero-title">Transformer Addition</p>
            <p class="hero-sub">
                An encoder-only Transformer that classifies each token's role —
                Plus Operator, Minus Operator, Operand — then computes the sum.
                Supports English &amp; Urdu digits.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("⚠️  model.pth not found. Run `python train.py` first.")
        st.stop()

    # Session state
    if "history" not in st.session_state:
        st.session_state.history = []
    if "expr_input" not in st.session_state:
        st.session_state.expr_input = "20+30="

    # Quick example buttons
    st.markdown('<div class="section-title">Quick Examples</div>', unsafe_allow_html=True)
    examples = ["20+30=", "-20+30=", "-20+-30=", "+5++95=", "۲۰+۳۰=", "99+99=", "-99+-1="]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.expr_input = ex

    st.markdown("---")

    # Input row
    st.markdown('<div class="input-label">Expression Input</div>', unsafe_allow_html=True)
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_text = st.text_input(
            "Expression",
            key="expr_input",
            placeholder="e.g.  20+30=   or   -20+-30=   or   +5++95=",
            label_visibility="collapsed",
        )
    with col_btn:
        clicked = st.button("Analyze ▶", type="primary", use_container_width=True)

    st.markdown(
        '<span style="font-size:0.82rem;color:#7a8aaa;">'
        'Supports: English &amp; Urdu digits &nbsp;·&nbsp; '
        'optional <code>+</code> or <code>-</code> sign before each operand'
        '</span>',
        unsafe_allow_html=True,
    )

    # Analysis
    if clicked:
        if not user_text.strip():
            st.warning("Please enter an expression first.")
            st.stop()

        try:
            details = infer_expression(model, user_text)
        except ValueError as exc:
            st.error(f"⚠️  {exc}")
            st.session_state.history.append({"expr": user_text, "result": None, "ok": False})
            st.stop()

        st.session_state.history.append({
            "expr":   details["cleaned"],
            "result": details["output"],
            "ok":     details["is_correct"],
        })

        # Result banner
        banner_cls = "banner-ok" if details["is_correct"] else "banner-err"
        check_msg  = "✓ Correct" if details["is_correct"] else f"✗ Expected {details['expected_output']}"
        eq_str = (
            f"{details['operand1']} &nbsp;+&nbsp; {details['operand2']}"
            f" &nbsp;=&nbsp; {details['output']}"
        )
        st.markdown(
            f'<div class="result-banner {banner_cls}">'
            f'<div class="result-eq">{eq_str}</div>'
            f'<div class="result-sub">{check_msg}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Stats row
        st.markdown('<div class="section-title">Model Stats</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Token Accuracy",  f"{details['token_accuracy']:.1f}%")
        s2.metric("Avg Confidence",  f"{details['average_confidence']:.1f}%")
        s3.metric("Inference Time",  f"{details['latency_ms']:.1f} ms")
        s4.metric("Cleaned Input",   details["cleaned"])

        # Token grid
        st.markdown('<div class="section-title">Token Role Breakdown</div>', unsafe_allow_html=True)
        render_legend()
        render_token_grid(details["rows"])

        # Full detail table (collapsible)
        with st.expander("📋  Full Detail Table"):
            st.dataframe(details["rows"], use_container_width=True, hide_index=True)

    # History
    render_history(st.session_state.history)


if __name__ == "__main__":
    main()
