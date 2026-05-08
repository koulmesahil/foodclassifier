"""
Dish Classifier — Streamlit app  (RISO COOKBOOK theme)
Pulls a fine-tuned MobileNetV2 from Hugging Face Hub and classifies a dish photo.

The model is trained with a 9th `other` class so it can decline unknown dishes.
We also apply a confidence threshold as an extra safety net.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ─── Configuration ─────────────────────────────────────────────────────────────
HF_REPO_ID = "koulsahil/food-classifier-2"   # <-- EDIT to your repo
MODEL_FILENAME = "food_classifier_mobilenetv2.keras"
LABELS_FILENAME = "class_names.json"
IMG_SIZE = 224
TRUST_THRESHOLD = 0.40   # below this we still show the prediction but mark it "off-menu"
SAMPLE_DIR = Path(__file__).parent / "sample_images"

st.set_page_config(
    page_title="Food Classifier",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Per-class metadata ────────────────────────────────────────────────────────
CLASS_META = {
    "pizza":        {"emoji": "🍕", "letter": "A", "tagline": "wood-fired",      "desc": "Italian flatbread baked with tomato sauce, melted cheese and a parade of toppings."},
    "sushi":        {"emoji": "🍣", "letter": "B", "tagline": "from the sea",    "desc": "Vinegared sushi rice paired with raw fish, seafood or vegetables."},
    "hamburger":    {"emoji": "🍔", "letter": "C", "tagline": "char-grilled",    "desc": "Seasoned beef patty stacked in a soft bun with cheese, lettuce and pickles."},
    "ramen":        {"emoji": "🍜", "letter": "D", "tagline": "slow-simmered",   "desc": "Japanese wheat noodles in a deep, savoury broth with chashu, egg and spring onion."},
    "ice_cream":    {"emoji": "🍨", "letter": "E", "tagline": "freshly churned", "desc": "Cold, creamy and sweet — frozen dairy folded with sugar and flavouring."},
    "tacos":        {"emoji": "🌮", "letter": "F", "tagline": "street-style",    "desc": "Soft or crisp tortilla folded around grilled meat, salsa, lime and herbs."},
    "donuts":       {"emoji": "🍩", "letter": "G", "tagline": "morning glaze",   "desc": "Pillowy fried dough rings, glazed, sprinkled or filled."},
    "french_fries": {"emoji": "🍟", "letter": "H", "tagline": "twice-fried",     "desc": "Golden potato sticks, crisp outside and fluffy inside."},
    "other":        {"emoji": "❔", "letter": "·", "tagline": "off-menu",        "desc": "Not one of the 8 dishes the press was set with. Try a clearer photo or a different dish."},
}

DEFAULT_CLASSES = ["pizza", "sushi", "hamburger", "ramen", "tacos", "donuts", "french_fries", "ice_cream"]
# ─── Model + labels ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model_and_labels():
    import tensorflow as tf
    from huggingface_hub import hf_hub_download
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        try:
            labels_path = hf_hub_download(repo_id=HF_REPO_ID, filename=LABELS_FILENAME)
            class_names = json.loads(Path(labels_path).read_text())
        except Exception:
            class_names = DEFAULT_CLASSES + ["other"]
        model = tf.keras.models.load_model(model_path)
        return model, class_names
    except Exception as e:
        st.error(f"Model could not be loaded from `{HF_REPO_ID}`. Edit HF_REPO_ID at the top of streamlit_app.py.\n\n{e}")
        return None, DEFAULT_CLASSES + ["other"]


def preprocess(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, class_names, image: Image.Image, top_k: int = 3):
    arr = preprocess(image)
    preds = model.predict(arr, verbose=0)[0]
    order = np.argsort(preds)[::-1][:top_k]
    return [(class_names[i], float(preds[i])) for i in order]


# ─── RISO COOKBOOK — global theme CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=EB+Garamond:ital,wght@0,400;0,500;1,400;1,500&family=IBM+Plex+Mono:wght@400;500;600&family=Yeseva+One&display=swap');

:root {
  --paper:    #f4ecd6;
  --paper-2:  #ebe1c5;
  --ink:      #0f1845;
  --ink-soft: #2a3060;
  --pink:     #c9440e;
  --pink-2:   #a83208;
  --gold:     #c89414;
  --rule:     rgba(15, 24, 69, 0.18);
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(rgba(15, 24, 69, 0.16) 0.9px, transparent 1.6px) 0 0/9px 9px,
    radial-gradient(rgba(255, 51, 128, 0.18) 0.9px, transparent 1.6px) 4.5px 4.5px/9px 9px,
    var(--paper) !important;
  background-attachment: fixed !important;
  color: var(--ink) !important;
  font-family: 'EB Garamond', serif !important;
}

/* Nuclear override for all Streamlit buttons */
.stButton > button,
.stButton > button:hover,
.stButton > button:focus,
div.stButton > button,
div[data-testid="stButton"] button,
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"],
[class*="st-emotion-cache"] button {
  background: var(--paper) !important;
  background-color: var(--paper) !important;
  color: var(--ink) !important;
  border: 2px solid var(--ink) !important;
  border-radius: 0 !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"] { display: none !important; }

.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}
[data-testid="stHorizontalBlock"] {
  max-width: 1200px !important;
  margin: 0 auto !important;
  padding: 0 3rem !important;
}
[data-testid="stColumn"] { padding-top: 1.6rem !important; }

/* ── Stat strip ─────────────────────────────────────────────────────────── */
.stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  max-width: 1200px;
  margin: 1.6rem auto 0;
  padding: 0 3rem;
  position: relative; z-index: 2;
}
.stat {
  position: relative;
  background: var(--paper);
  border: 2px solid var(--ink);
  border-radius: 0;
  padding: 1.1rem 1.2rem;
  transition: transform 0.18s, box-shadow 0.25s;
}
.stat::before {
  content: '';
  position: absolute;
  inset: 4px -4px -4px 4px;
  background: var(--pink);
  z-index: -1;
}
.stat:hover { transform: translate(-2px, -2px); }
.stat:hover::before { transform: translate(2px, 2px); }
.stat-val {
  font-family: 'DM Serif Display', serif;
  font-size: 2rem;
  color: var(--ink);
  letter-spacing: -0.02em;
  line-height: 1;
}
.stat-val sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  font-weight: 400;
  vertical-align: baseline;
  color: var(--pink-2);
  margin-left: 4px;
}
.stat-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--ink-soft);
  margin-top: 8px;
}

/* ── Section headers ─────────────────────────────────────────────────── */
.section-tag {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  color: var(--pink-2);
  margin-bottom: 0.4rem;
  display: flex;
  align-items: center;
  gap: 10px;
}
.section-tag::before {
  content: '§';
  font-family: 'EB Garamond', serif;
  font-style: italic;
  font-size: 14px;
  color: var(--ink);
}
.section-title {
  font-family: 'DM Serif Display', serif;
  font-size: 2rem;
  color: var(--ink);
  letter-spacing: -0.02em;
  line-height: 1;
  margin: 0 0 1.4rem;
  position: relative;
  display: inline-block;
}
.section-title em {
  font-style: italic;
  color: var(--pink-2);
}
.section-title::after {
  content: '';
  position: absolute;
  bottom: -10px; left: 0;
  width: 36px; height: 2px;
  background: var(--ink);
}

/* ── TILE BUTTONS — recipe card aesthetic with ink-stamp click ─────── */
.menu-zone .stButton > button {
  height: 100px;
  border-radius: 0 !important;
  background: var(--paper) !important;
  border: 2px solid var(--ink) !important;
  color: var(--ink) !important;
  font-size: 2.5rem !important;
  line-height: 1 !important;
  font-family: 'DM Serif Display', serif !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  padding: 0 !important;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s cubic-bezier(.2,1,.4,1.4),
              background 0.18s,
              color 0.18s,
              box-shadow 0.25s !important;
}
/* hard offset pink shadow that emerges on hover */
.menu-zone .stButton > button::before {
  content: '';
  position: absolute;
  inset: 4px -4px -4px 4px;
  background: var(--pink);
  z-index: -1;
  transition: transform 0.2s;
}
/* halftone burst pseudo for click */
.menu-zone .stButton > button::after {
  content: '';
  position: absolute;
  left: 50%; top: 50%;
  width: 0; height: 0;
  background:
    radial-gradient(rgba(15,24,69,0.55) 1px, transparent 1.6px) 0 0/7px 7px,
    radial-gradient(rgba(255,51,128,0.55) 1px, transparent 1.6px) 3.5px 3.5px/7px 7px;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  pointer-events: none;
  opacity: 0;
}

.menu-zone .stButton > button:hover {
  transform: translate(-3px, -3px);
}
.menu-zone .stButton > button:hover::before {
  transform: translate(3px, 3px);
}
.menu-zone .stButton > button:focus,
.menu-zone .stButton > button:focus-visible {
  outline: none !important;
  background: var(--ink) !important;
  color: var(--paper) !important;
  border-color: var(--ink) !important;
  transform: translate(-3px, -3px);
}
.menu-zone .stButton > button:focus::before {
  transform: translate(3px, 3px);
  background: var(--gold);
}

/* CLICK INK STAMP */
.menu-zone .stButton > button:active {
  transform: translate(-1px, -1px) rotate(-1.5deg) !important;
  background: var(--pink) !important;
  color: var(--paper) !important;
  transition: transform 0.08s, background 0.05s, color 0.05s !important;
}
.menu-zone .stButton > button:active::after {
  animation: ink-stamp 0.6s cubic-bezier(.2,.7,.2,1) forwards;
}
@keyframes ink-stamp {
  0%   { width: 0;     height: 0;     opacity: 1;   }
  60%  { width: 240px; height: 240px; opacity: 0.6; }
  100% { width: 360px; height: 360px; opacity: 0;   }
}
/* tile name caption beneath the button */
.tile-name {
  margin-top: 8px;
  text-align: center;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--ink-soft);
}
.tile-name .num {
  color: var(--pink-2);
  font-weight: 600;
  margin-right: 6px;
}

/* ── Selectbox (recipe index style) ─────────────────────────────────── */
[data-baseweb="select"] > div {
  background: var(--paper) !important;
  border: 2px solid var(--ink) !important;
  border-radius: 0 !important;
  font-family: 'EB Garamond', serif !important;
  font-style: italic !important;
  font-size: 1rem !important;
  color: var(--ink) !important;
}
[data-baseweb="select"] svg { fill: var(--ink) !important; }
[data-baseweb="popover"] {
  background: var(--paper) !important;
  border: 2px solid var(--ink) !important;
}

/* ── Action buttons (Surprise / Clear / Identify) ─────────────────── */
.action-zone .stButton > button {
  background: var(--paper) !important;
  border: 2px solid var(--ink) !important;
  color: var(--ink) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.24em !important;
  text-transform: uppercase !important;
  border-radius: 0 !important;
  padding: 0.7rem 1rem !important;
  position: relative;
  transition: all 0.18s !important;
  
}
.action-zone .stButton > button::before {
  content: '';
  position: absolute;
  inset: 3px -3px -3px 3px;
  background: var(--pink);
  z-index: -1;
  transition: transform 0.18s;
}
.action-zone .stButton > button:hover {
  background: var(--ink) !important;
  color: var(--paper) !important;
  transform: translate(-2px, -2px);
}
.action-zone .stButton > button:hover::before {
  transform: translate(2px, 2px);
  background: var(--gold);
}
.action-zone .stButton > button:active {
  transform: translate(0, 0) !important;
}
.action-zone .stButton > button:active::before {
  transform: translate(0, 0);
}

/* ── File uploader ───────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--paper) !important;
  border: 2px dashed var(--ink) !important;
  border-radius: 0 !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] > section > div {
  background: var(--paper) !important;
  border: none !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: var(--paper) !important;
  border: none !important;
}
/* Hide the duplicate label text */
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
  display: none !important;
}
[data-testid="stFileUploader"] label {
  display: none !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p {
  color: var(--ink-soft) !important;
  font-family: 'EB Garamond', serif !important;
  font-style: italic !important;
}
/* Style the browse button to match theme */
[data-testid="stFileUploaderDropzone"] button {
  background: var(--paper) !important;
  background-color: var(--paper) !important;
  border: 2px solid var(--ink) !important;
  border-radius: 0 !important;
  color: var(--ink) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 0.2em !important;
  text-transform: uppercase !important;
}
/* ── Empty state (a blank recipe page) ──────────────────────────────── */
.empty-state {
  height: 320px;
  background: var(--paper);
  border: 2px solid var(--ink);
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.empty-state::before {
  content: '';
  position: absolute;
  inset: 6px;
  border: 1px solid var(--ink);
  pointer-events: none;
}
.empty-state::after {
  content: '✦';
  position: absolute;
  top: 14px; left: 50%;
  transform: translateX(-50%);
  font-family: 'EB Garamond', serif;
  color: var(--gold);
  font-size: 14px;
}
.empty-state .icon {
  font-size: 3.4rem;
  filter: grayscale(0.3);
}
.empty-state .title {
  margin-top: 1rem;
  font-family: 'DM Serif Display', serif;
  font-style: italic;
  font-size: 1.4rem;
  color: var(--ink);
}
.empty-state .hint {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.24em;
  color: var(--pink-2);
  margin-top: 0.5rem;
  text-transform: uppercase;
}
.empty-state .rule {
  margin-top: 0.8rem;
  width: 80px; height: 1px; background: var(--rule);
}

/* ── Result card (a printed recipe page) ───────────────────────────── */
.result-card {
  position: relative;
  background: var(--paper);
  border: 2px solid var(--ink);
  padding: 2rem 1.8rem 1.6rem;
  margin-top: 1rem;
}
.result-card::before {
  content: '';
  position: absolute;
  inset: 6px;
  border: 1px solid var(--ink);
  pointer-events: none;
}
.result-card::after {
  content: '✦  ✦  ✦';
  position: absolute;
  top: 14px; left: 50%;
  transform: translateX(-50%);
  font-family: 'EB Garamond', serif;
  color: var(--gold);
  font-size: 11px;
  letter-spacing: 0.5em;
}
.recipe-no {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.3em;
  color: var(--pink-2);
  text-transform: uppercase;
  margin-top: 6px;
  margin-bottom: 0.8rem;
}
.result-name {
  font-family: 'DM Serif Display', serif;
  font-size: 3rem;
  line-height: 1;
  letter-spacing: -0.025em;
  color: var(--ink);
  position: relative;
  display: inline-block;
}
.result-name::before {
  content: attr(data-shadow);
  position: absolute;
  top: 4px; left: 4px;
  color: var(--pink);
  z-index: -1;
}
.result-tagline {
  font-family: 'EB Garamond', serif;
  font-style: italic;
  font-size: 1.3rem;
  color: var(--ink-soft);
  margin-top: 0.6rem;
}
.result-tagline::before { content: '— '; color: var(--gold); }
.result-tagline::after  { content: ' —'; color: var(--gold); }

.result-warn {
  display: inline-block;
  margin-top: 0.8rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  padding: 5px 9px;
  border: 2px solid var(--pink-2);
  background: var(--paper);
  color: var(--pink-2);
}

/* dotted-leader confidence row — recipe quantity style */
.conf-row {
  margin-top: 1.5rem;
  display: flex;
  align-items: baseline;
  font-family: 'EB Garamond', serif;
  font-size: 1.05rem;
  color: var(--ink);
}
.conf-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--ink-soft);
}
.conf-fill-row {
  flex: 1;
  border-bottom: 2px dotted var(--ink);
  margin: 0 0.6rem 4px;
}
.conf-pct {
  font-family: 'DM Serif Display', serif;
  font-size: 1.4rem;
  color: var(--pink-2);
  letter-spacing: -0.01em;
}

.conf-bar-track {
  margin-top: 0.7rem;
  height: 6px;
  background: var(--paper-2);
  border: 1px solid var(--ink);
  position: relative;
}
.conf-bar-fill {
  height: 100%;
  background:
    repeating-linear-gradient(45deg,
      var(--pink) 0 4px,
      var(--ink) 4px 8px);
  transition: width 0.8s cubic-bezier(.2,1,.4,1);
}
.result-card.is-other .conf-bar-fill {
  background:
    repeating-linear-gradient(45deg,
      var(--gold) 0 4px,
      var(--ink) 4px 8px);
}

.result-desc {
  margin-top: 1.5rem;
  padding-top: 1.2rem;
  border-top: 1px dashed var(--ink);
  font-family: 'EB Garamond', serif;
  font-size: 1.1rem;
  font-style: italic;
  color: var(--ink-soft);
  line-height: 1.55;
}
.result-desc::first-letter {
  font-family: 'DM Serif Display', serif;
  font-style: normal;
  font-size: 2.4rem;
  float: left;
  margin-right: 6px;
  line-height: 0.85;
  color: var(--pink-2);
}

.alt-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.28em;
  color: var(--pink-2);
  text-transform: uppercase;
  margin: 1.6rem 0 0.7rem;
}
.alt-row {
  display: flex; align-items: baseline;
  margin-bottom: 0.55rem;
  font-family: 'EB Garamond', serif;
  font-size: 1.1rem;
  color: var(--ink);
}
.alt-num {
  font-family: 'DM Serif Display', serif;
  font-size: 1rem;
  width: 24px;
  color: var(--pink-2);
  flex-shrink: 0;
}
.alt-emoji { width: 28px; flex-shrink: 0; font-size: 1.05rem; }
.alt-name {
  flex-shrink: 0;
  font-style: italic;
  width: 130px;
}
.alt-fill { flex: 1; border-bottom: 1.5px dotted var(--ink); margin: 0 8px 4px; }
.alt-pct {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  font-weight: 500;
  color: var(--ink-soft);
  width: 50px; text-align: right; flex-shrink: 0;
}

/* ── Selected image preview ────────────────────────────────────────── */
[data-testid="stImage"] img {
  border: 2px solid var(--ink);
  border-radius: 0;
  box-shadow: 6px 6px 0 var(--pink);
}
[data-testid="stImageCaption"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 10px !important;
  letter-spacing: 0.22em !important;
  text-transform: uppercase !important;
  color: var(--ink-soft) !important;
  text-align: center !important;
  margin-top: 0.6rem !important;
}

/* dish info panel */
.dish-info { padding: 0.5rem 0; }
.dish-info .em {
  font-size: 2.6rem;
  filter: drop-shadow(2px 2px 0 var(--pink));
}
.dish-info .tag {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  color: var(--pink-2);
  margin: 0.9rem 0 0.3rem;
}
.dish-info .name {
  font-family: 'DM Serif Display', serif;
  font-size: 1.7rem;
  letter-spacing: -0.02em;
  color: var(--ink);
}
.dish-info .tagline {
  font-family: 'EB Garamond', serif;
  font-style: italic;
  font-size: 1.1rem;
  color: var(--ink-soft);
  margin-top: 0.1rem;
}
.dish-info .tagline::before { content: '— '; color: var(--gold); }
.dish-info .blurb {
  margin-top: 0.7rem;
  font-family: 'EB Garamond', serif;
  font-size: 1rem;
  color: var(--ink);
  line-height: 1.55;
}

/* ── Footer (colophon) ─────────────────────────────────────────────── */
.footer {
  text-align: center;
  padding: 2.4rem 1rem;
  margin: 2.5rem auto 0;
  max-width: 1200px;
  border-top: 1px solid var(--rule);
  position: relative;
}
.footer::before {
  content: '✦';
  position: absolute;
  top: -10px; left: 50%;
  transform: translateX(-50%);
  background: var(--paper);
  padding: 0 14px;
  color: var(--gold);
  font-size: 14px;
}
.footer-line-1 {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.3em;
  color: var(--ink-soft);
  text-transform: uppercase;
}
.footer-line-2 {
  font-family: 'EB Garamond', serif;
  font-style: italic;
  font-size: 1rem;
  color: var(--ink);
  margin-top: 0.8rem;
}
.footer a {
  color: var(--pink-2);
  text-decoration: none;
  border-bottom: 1px solid var(--pink-2);
}
.footer .ornament {
  margin: 0 0.5rem;
  color: var(--gold);
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── HERO  (cookbook spread with rotating woodcut, halftone, ink cursor) ─────
hero_html = r"""<!DOCTYPE html>
<html><head><style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=EB+Garamond:ital,wght@0,400;1,400;1,500&family=IBM+Plex+Mono:wght@400;500;600&family=Yeseva+One&display=swap');
* { margin:0; padding:0; box-sizing:border-box; }
html, body { height:100%; }
body {
  background: #f4ecd6;
  background-image:
    radial-gradient(rgba(15, 24, 69, 0.16) 0.9px, transparent 1.6px) 0 0/9px 9px,
    radial-gradient(rgba(255, 51, 128, 0.18) 0.9px, transparent 1.6px) 4.5px 4.5px/9px 9px;
  overflow: hidden;
  cursor: none;
  font-family: 'EB Garamond', serif;
  color: #0f1845;
}
.hero {
  position: relative;
  width: 100%;
  height: 460px;
  border-bottom: 2px solid #0f1845;
}
.hero::before {
  /* a hand-printed second border for that mis-registered feel */
  content: '';
  position: absolute;
  bottom: -8px; left: 0; right: 0;
  height: 1px;
  background: #c9440e;
}

/* page-number / issue marker, top-left */
.page-no {
  position: absolute;
  top: 28px; left: 3.2rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.32em;
  color: #c9440e;
  text-transform: uppercase;
  display: flex; align-items: center; gap: 10px;
}
.page-no::after {
  content: '';
  width: 30px; height: 1px;
  background: #c9440e;
}

/* tiny corner ornament */
.corner-mark {
  position: absolute;
  top: 28px; right: 3.2rem;
  font-family: 'EB Garamond', serif;
  font-style: italic;
  font-size: 14px;
  color: #c89414;
  letter-spacing: 0.3em;
}

/* TYPESET TITLE — centered */
.txt {
  position: absolute;
  left: 0; right: 0;
  top: 50%;
  transform: translateY(-50%);
  z-index: 5;
  pointer-events: none;
  text-align: center;
  padding: 0 3.2rem;
}
.title {
  font-family: 'DM Serif Display', serif;
  font-size: clamp(3.4rem, 7.8vw, 6rem);
  line-height: 0.88;
  letter-spacing: -0.035em;
  color: #0f1845;
  position: relative;
}
.title .l1 {
  display: block;
  position: relative;
}


.subtitle {
  font-family: 'EB Garamond', serif;
  font-style: italic;
  font-size: 1.4rem;
  color: #2a3060;
  margin-top: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
}
.subtitle::before, .subtitle::after {
  content: '';
  width: 28px; height: 2px;
  background: #c89414;
}

.body-text {
  margin: 1.2rem auto 0;
  font-family: 'EB Garamond', serif;
  font-size: 1rem;
  color: #0f1845;
  max-width: 480px;
  line-height: 1.55;
}
.body-text strong {
  font-weight: 500;
  color: #c9440e;
  font-style: italic;
}

.specs {
  margin-top: 1.4rem;
  display: flex;
  gap: 1.6rem;
  justify-content: center;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.2em;
  color: #2a3060;
  text-transform: uppercase;
}
.specs .star { color: #c89414; }

/* WOODCUT — hidden, awaiting replacement side animation */
.woodcut-wrap {
  display: none;
  position: absolute;
  right: 5%;
  top: 50%;
  transform: translateY(-50%);
  width: 320px;
  height: 320px;
  z-index: 3;
  pointer-events: none;
}
.woodcut {
  position: relative;
  width: 100%;
  height: 100%;
  transition: transform 0.4s cubic-bezier(.2,1,.4,1);
}
.woodcut .ring {
  position: absolute;
  inset: 0;
  border: 2px solid #0f1845;
  border-radius: 50%;
  background: #f4ecd6;
}
.woodcut .ring-2 {
  position: absolute;
  inset: 14px;
  border: 1px solid #0f1845;
  border-radius: 50%;
}
.woodcut .ring-3 {
  position: absolute;
  inset: 26px;
  border: 6px solid #c9440e;
  border-radius: 50%;
  box-shadow: inset 0 0 0 1px #0f1845;
}
.woodcut .core {
  position: absolute;
  inset: 56px;
  background: #0f1845;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  animation: rotate 60s linear infinite;
}
.woodcut .core::before {
  content: '';
  position: absolute;
  inset: 14px;
  background:
    radial-gradient(rgba(244, 236, 214, 0.55) 1px, transparent 1.6px) 0 0/6px 6px;
  border-radius: 50%;
}
.woodcut .core::after {
  content: '';
  position: absolute;
  width: 70px; height: 70px;
  border-radius: 50%;
  background: #c9440e;
  box-shadow: 0 0 0 4px #f4ecd6, 0 0 0 5px #0f1845;
}
.woodcut .label {
  position: absolute;
  z-index: 2;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.4em;
  color: #f4ecd6;
  text-transform: uppercase;
  text-shadow: 0 0 6px #0f1845;
}
@keyframes rotate { to { transform: rotate(360deg); } }

/* TICK MARKS around outer edge */
.woodcut .tick {
  position: absolute;
  width: 2px; height: 8px;
  background: #0f1845;
  left: 50%; top: -1px;
  transform-origin: 0 161px;
  margin-left: -1px;
}

/* radial laurel of italic words around the woodcut */
.laurel {
  position: absolute; inset: -28px;
  pointer-events: none;
}
.laurel svg { width: 100%; height: 100%; }
.laurel text {
  font-family: 'EB Garamond', serif;
  font-style: italic;
  font-size: 13px;
  fill: #c89414;
  letter-spacing: 0.18em;
}

/* SCATTERED tiny dots / printers' ornaments */
.dot {
  position: absolute;
  width: 4px; height: 4px;
  border-radius: 50%;
  background: #c9440e;
}
.dot.d1 { top: 22%; left: 56%; }
.dot.d2 { bottom: 18%; left: 60%; background: #c89414; }
.dot.d3 { top: 34%; right: 33%; background: #0f1845; width: 6px; height: 6px; }
.dot.d4 { bottom: 30%; right: 30%; }

/* PIZZA CURSOR + trail + burst */
.pizza-cursor {
  position: absolute;
  font-size: 32px;
  line-height: 1;
  pointer-events: none;
  z-index: 100;
  user-select: none;
  transform: translate(-100px, -100px);
  filter: drop-shadow(2px 3px 0 rgba(15, 24, 69, 0.25));
  transition: opacity 0.18s;
}
.pizza-cursor.hidden { opacity: 0; }
.pizza-trail {
  position: absolute;
  pointer-events: none;
  user-select: none;
  z-index: 50;
  line-height: 1;
  transition: opacity 0.6s ease-out, transform 0.6s ease-out;
}
.pizza-burst {
  position: absolute;
  pointer-events: none;
  user-select: none;
  z-index: 60;
  line-height: 1;
  will-change: transform, opacity;
}
</style></head>
<body>
<div class="hero" id="hero">

  <div class="dot d1"></div>
  <div class="dot d2"></div>
  <div class="dot d3"></div>
  <div class="dot d4"></div>

  <div class="txt">
    <div class="title">
      <span class="l1">Food</span>
      <span class="l2">Classifier.</span>
    </div>
    <div class="subtitle">image classification, 9 classes</div>
    <div class="body-text">Drop a photo, get a label. Fine-tuned on 8 dish categories with a built-in <strong>open-set class</strong> that catches anything the model isn't confident about.</div>    <div class="specs">
      <span><span class="star">✦</span> mobilenetv2</span>
      <span><span class="star">✦</span> fine-tuned · food-101</span>
      <span><span class="star">✦</span> 92% val accuracy</span>    </div>
  </div>

  <div class="woodcut-wrap" id="wc-wrap">
    <div class="woodcut" id="wc">
      <div class="ring"></div>
      <div class="ring-2"></div>
      <div class="ring-3"></div>
      <div class="core">
        <span class="label">v1.0</span>
      </div>
      <div class="laurel">
        <svg viewBox="0 0 360 360">
          <defs>
            <path id="circle-path" d="M 180,180 m -160,0 a 160,160 0 1,1 320,0 a 160,160 0 1,1 -320,0" />
          </defs>
          <text>
            <textPath href="#circle-path" startOffset="0%">
              · mobilenetv2 · food-101 · 9 classes · open-set · 224×224 · softmax ·
            </textPath>
          </text>
        </svg>
      </div>
    </div>
  </div>

  <div class="pizza-cursor hidden" id="pz-cursor">🍕</div>
</div>

<script>
(function(){
  var hero = document.getElementById('hero');
  var cursor = document.getElementById('pz-cursor');

  var mx = -100, my = -100;        // raw mouse
  var cx = -100, cy = -100;        // smoothed cursor
  var lastX = -100, lastY = -100;  // for velocity
  var tilt = 0;
  var inside = false;
  var lastTrail = 0;
  var bursts = [];                  // active burst particles
  var GRAVITY = 0.32;

  hero.addEventListener('mouseenter', function(){
    inside = true;
    cursor.classList.remove('hidden');
  });
  hero.addEventListener('mouseleave', function(){
    inside = false;
    cursor.classList.add('hidden');
    mx = -100; my = -100;
  });
  hero.addEventListener('mousemove', function(e){
    var r = hero.getBoundingClientRect();
    mx = e.clientX - r.left;
    my = e.clientY - r.top;
  });

  hero.addEventListener('mousedown', function(e){
    var r = hero.getBoundingClientRect();
    var bx = e.clientX - r.left;
    var by = e.clientY - r.top;
    var N = 24;
    for (var i = 0; i < N; i++) {
      var angle = (Math.PI * 2 * i) / N + (Math.random() - 0.5) * 0.4;
      var speed = 4 + Math.random() * 7;
      var size = 16 + Math.floor(Math.random() * 24);
      var p = document.createElement('div');
      p.className = 'pizza-burst';
      p.textContent = '🍕';
      p.style.fontSize = size + 'px';
      p.style.left = '0';
      p.style.top  = '0';
      hero.appendChild(p);
      bursts.push({
        el: p,
        x: bx, y: by,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 3,
        rot: Math.random() * 360,
        rotSpeed: (Math.random() - 0.5) * 16,
        life: 1,
        decay: 0.012 + Math.random() * 0.008,
      });
    }
  });

  function frame(now){
    /* smooth pizza-cursor follow + tilt by velocity */
    cx += (mx - cx) * 0.32;
    cy += (my - cy) * 0.32;
    var vx = cx - lastX;
    var vy = cy - lastY;
    var targetTilt = Math.max(-25, Math.min(25, vx * 1.2));
    tilt += (targetTilt - tilt) * 0.18;
    cursor.style.transform = 'translate(' + (cx - 16) + 'px, ' + (cy - 16) + 'px) rotate(' + tilt + 'deg)';
    lastX = cx; lastY = cy;

    /* drop a small trail pizza every ~80ms while moving */
    var moving = (Math.abs(vx) + Math.abs(vy)) > 0.6;
    if (inside && moving && (!lastTrail || now - lastTrail > 80)) {
      lastTrail = now;
      var t = document.createElement('div');
      t.className = 'pizza-trail';
      t.textContent = '🍕';
      t.style.fontSize = (10 + Math.random() * 6) + 'px';
      t.style.opacity = '0.55';
      t.style.left = '0';
      t.style.top  = '0';
      var jx = (Math.random() - 0.5) * 6;
      var jy = (Math.random() - 0.5) * 6;
      var rotStart = (Math.random() - 0.5) * 30;
      t.style.transform = 'translate(' + (cx - 8 + jx) + 'px, ' + (cy - 8 + jy) + 'px) rotate(' + rotStart + 'deg)';
      hero.appendChild(t);
      requestAnimationFrame(function(){
        t.style.opacity = '0';
        t.style.transform = 'translate(' + (cx - 8 + jx) + 'px, ' + (cy + 14 + jy) + 'px) rotate(' + (rotStart + 25) + 'deg)';
      });
      setTimeout(function(){ if (t.parentNode) t.parentNode.removeChild(t); }, 650);
    }

    /* update burst particles */
    for (var i = bursts.length - 1; i >= 0; i--) {
      var b = bursts[i];
      b.vy += GRAVITY;
      b.x += b.vx;
      b.y += b.vy;
      b.rot += b.rotSpeed;
      b.life -= b.decay;
      if (b.life <= 0 || b.y > hero.clientHeight + 60) {
        if (b.el.parentNode) b.el.parentNode.removeChild(b.el);
        bursts.splice(i, 1);
        continue;
      }
      b.el.style.opacity = Math.max(0, b.life);
      b.el.style.transform = 'translate(' + (b.x - 16) + 'px, ' + (b.y - 16) + 'px) rotate(' + b.rot + 'deg)';
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
</body></html>"""

st.components.v1.html(hero_html, height=470)


# ─── Stats strip ──────────────────────────────────────────────────────────────



# ─── Layout ───────────────────────────────────────────────────────────────────
left, right = st.columns(2, gap="large")

# ─── LEFT: pick / upload ──────────────────────────────────────────────────────
with left:
    #st.markdown('<div class="section-tag">Section I — Ingredients</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Choose a <em>sample image</em></div>', unsafe_allow_html=True)

    # tiles laid out in rows of (up to) 4
    st.markdown('<div class="menu-zone">', unsafe_allow_html=True)
    PER_ROW = 4
    rows = [DEFAULT_CLASSES[i:i + PER_ROW] for i in range(0, len(DEFAULT_CLASSES), PER_ROW)]
    for row in rows:
        n_cols = max(len(row), 1)
        cols = st.columns(n_cols, gap="small")
        for ci, dish in enumerate(row):
            meta = CLASS_META[dish]
            with cols[ci]:
                if st.button(meta["emoji"], key=f"tile_{dish}", use_container_width=True):
                    candidate = SAMPLE_DIR / f"{dish}.jpg"
                    if candidate.exists():
                        st.session_state["image_bytes"] = candidate.read_bytes()
                        st.session_state["image_label"] = dish
                        st.rerun()
                    else:
                        st.toast(f"sample_images/{dish}.jpg not found — upload your own.", icon="⚠️")
                pretty = dish.replace("_", " ")
                st.markdown(
                    f'<div class="tile-name"><span class="num">{meta["letter"]}.</span>{pretty}</div>',
                    unsafe_allow_html=True,
                )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1.4rem;">Or upload your own <em>image</em></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

# ─── State updates ────────────────────────────────────────────────────────────
if uploaded is not None:
    st.session_state["image_bytes"] = uploaded.read()
    st.session_state["image_label"] = "uploaded"


# ─── RIGHT: prediction ────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-title">Run <em>inference</em></div>', unsafe_allow_html=True)

    has_image = bool(st.session_state.get("image_bytes"))

    if not has_image:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">📷</div>
          <div class="title">No image selected</div>
          <div class="hint">— select a sample or upload a photo —</div>
          <div class="rule"></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(io.BytesIO(st.session_state["image_bytes"]))
        label = st.session_state.get("image_label", "uploaded")

        img_col, info_col = st.columns([1, 1])
        with img_col:
            caption = label.replace("_", " ").title() if label != "uploaded" else "Your upload"
            st.image(image, caption=caption, use_container_width=True)
        with info_col:
            meta = CLASS_META.get(label)
            if meta:
                st.markdown(f"""
                <div class="dish-info">
                  <div class="em">{meta['emoji']}</div>
                  <div class="tag">— Sample {meta['letter']} · selected —</div>
                  <div class="name">{label.replace('_',' ').title()}</div>
                  <div class="tagline">{meta['tagline']}</div>
                  <div class="blurb">{meta['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="dish-info">
                  <div class="em">📷</div>
                  <div class="tag">— Custom upload —</div>
                  <div class="name">User image</div>
                  <div class="tagline">awaiting prediction</div>
                  <div class="blurb">Click below to run the model on this image.</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div style="height:0.7rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-zone">', unsafe_allow_html=True)
        identify = st.button("✦ Run inference", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if identify:
            model, class_names = load_model_and_labels()
            if model is None:
                st.error("Model unavailable — check HF_REPO_ID at the top of streamlit_app.py.")
            else:
                with st.spinner("Running inference…"):
                    time.sleep(0.4)
                    results = predict(model, class_names, image, top_k=3)

                top_label, top_conf = results[0]
                low_conf = top_conf < TRUST_THRESHOLD
                is_other = (top_label == "other") or low_conf

                display_label = "other" if is_other else top_label
                display_conf  = top_conf

                meta = CLASS_META.get(display_label, CLASS_META["other"])
                pretty = display_label.replace("_", " ").title()
                emoji = meta["emoji"]
                letter = meta["letter"]
                tagline = meta["tagline"]
                desc = meta["desc"]
                pct = display_conf * 100

                warn_html = ""
                if is_other and top_label != "other":
                    warn_html = '<div class="result-warn">below trust threshold · routed to OTHER</div>'
                elif top_label == "other":
                    warn_html = '<div class="result-warn">predicted: OTHER</div>'

                klass_extra = " is-other" if is_other else ""
                st.markdown(f"""
                <div class="result-card{klass_extra}">
                  <div class="recipe-no">— Class · {letter} —</div>
                  <div class="result-name" data-shadow="{pretty}">{pretty}</div>
                  <div class="result-tagline">{tagline}</div>
                  {warn_html}
                  <div class="conf-row">
                    <span class="conf-label">Confidence</span>
                    <span class="conf-fill-row"></span>
                    <span class="conf-pct">{pct:.1f}%</span>
                  </div>
                  <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{pct:.1f}%;"></div></div>
                  <div class="result-desc">{desc}</div>
                  <div class="alt-label">— top-3 predictions —</div>
                """, unsafe_allow_html=True)

                for idx, (lbl, conf) in enumerate(results, start=1):
                    pct_a = conf * 100
                    m = CLASS_META.get(lbl, CLASS_META["other"])
                    pretty_a = lbl.replace("_", " ").title()
                    st.markdown(f"""
                    <div class="alt-row">
                      <span class="alt-num">{idx}.</span>
                      <span class="alt-emoji">{m['emoji']}</span>
                      <span class="alt-name">{pretty_a}</span>
                      <span class="alt-fill"></span>
                      <span class="alt-pct">{pct_a:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-line-2">
    Created by -
    <a href="https://koulmesahil.github.io/">Sahil</a>
    <span class="ornament">✦</span>
    <a href="https://www.linkedin.com/in/sahilkoul123/">LinkedIn</a>
    <span class="ornament">✦</span>
    <a href="https://www.linkedin.com/in/sahilkoul123/">Documentation</a>
  </div>
</div>
""", unsafe_allow_html=True)
