import os

import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RECGEN_MODEL_DIR", "HuggingFaceTB/SmolLM2-360M")
DEVICE = os.environ.get("RECGEN_DEMO_DEVICE", "mps")
if DEVICE == "mps" and not torch.backends.mps.is_available():
    DEVICE = "cpu"

GUITAR_ITEMS = [
    "Item: Fender Player Stratocaster. Category: electric guitar. Single-coil pickups, classic twang, versatile clean tones.",
    "Item: Gibson Les Paul Standard. Category: electric guitar. Humbuckers, mahogany body, warm thick sustain, rock tone.",
    "Item: Telecaster Electric Guitar. Category: electric guitar. Bright twang, country and blues classic.",
    "Item: Taylor Dreadnought Acoustic. Category: acoustic guitar. Spruce top, warm resonant projection.",
    "Item: Yamaha Classical Nylon Guitar. Category: acoustic guitar. Nylon strings, mellow tone, fingerstyle.",
    "Item: Boss DS-1 Distortion Pedal. Category: effects pedal. Legendary hard rock distortion, scooped mids.",
    "Item: Ibanez TS9 Tube Screamer. Category: effects pedal. Vintage overdrive, warm mid boost, blues lead tone.",
    "Item: Behringer Fuzz Pedal. Category: effects pedal. Classic silicon fuzz, heavy sustain, psychedelic rock.",
    "Item: TC Electronic Chorus Pedal. Category: effects pedal. Spacious modulation, shimmering ambience.",
    "Item: Fender 8ft Instrument Cable. Category: cable. Braided shield, quiet signal.",
    "Item: Ernie Ball Slinky Strings. Category: strings. Nickel wound 10-46, bright punchy tone.",
    "Item: D'Addario NYXL Strings. Category: strings. 11-49 medium, tuning stability, enhanced midrange.",
    "Item: Marshall MG30GFX Combo Amp. Category: amplifier. 30 watts, overdrive channel, British crunch.",
    "Item: Fender 15W Practice Amp. Category: amplifier. Clean sparkle, bedroom volume.",
    "Item: Mesa Boogie 100W Head. Category: amplifier. High gain, dual rectifier, metal tone.",
    "Item: Shure SM57 Microphone. Category: microphone. Cardioid, industry standard for instrument miking.",
    "Item: Sennheiser HD 280 Headphones. Category: headphones. Closed back, flat response, studio monitoring.",
    "Item: Focusrite Audio Interface. Category: interface. 2 inputs, USB, home recording.",
]

tokens, model = None, None


def load():
    global tokens, model
    if model is None:
        tokens = AutoTokenizer.from_pretrained(MODEL)
        if tokens.pad_token is None:
            tokens.pad_token = tokens.eos_token
        dtype = torch.float16 if DEVICE == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, torch_dtype=dtype).to(DEVICE).eval()
        model.config.use_cache = False


def embed(texts):
    load()
    ids = tokens(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    hidden = out.hidden_states[-1].float()
    attn = ids["attention_mask"]
    return (hidden * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(dim=1, keepdim=True)


def rank_history(history: str, top_k: int):
    items = GUITAR_ITEMS
    ctx = embed([f"The user recently purchased: {history}"])[0]
    embs = embed(items)
    scores = torch.nn.functional.cosine_similarity(ctx.unsqueeze(0), embs)
    order = torch.argsort(scores, descending=True)[:top_k]
    rows = [[items[i], round(float(scores[i]), 4)] for i in order.tolist()]
    return rows


def sim_search(query: str, top_k: int):
    items = GUITAR_ITEMS
    q = embed([query])[0]
    embs = embed(items)
    scores = torch.nn.functional.cosine_similarity(q.unsqueeze(0), embs)
    order = torch.argsort(scores, descending=True)[:top_k]
    return [[items[i], round(float(scores[i]), 4)] for i in order.tolist()]


load()

demo = gr.Blocks(title="recgen — LLM-as-encoder demo")
with demo:
    gr.Markdown(
        """# recgen — LLMs are cheap when you stop generating

A 360M LLM encodes *any* text into a 960-dim semantic embedding in one forward
pass. This demo shows two use cases: **next-item recommendation** from a
purchase history, and **semantic item search** — no training, no generation,
pure frozen-LLM embeddings. Learn more at
[github.com/Bturan19/recgen](https://github.com/Bturan19/recgen)."""
    )
    with gr.Tab("Recommend from history"):
        history = gr.Textbox(
            label="Purchase history",
            value="Boss DS-1 Distortion Pedal, Fender 8ft Instrument Cable",
        )
        k1 = gr.Slider(1, 10, value=3, step=1, label="Top-k")
        out1 = gr.Dataframe(headers=["item", "score"], label="Recommended next items")
        btn1 = gr.Button("Rank")
        btn1.click(rank_history, [history, k1], out1)
    with gr.Tab("Semantic search"):
        query = gr.Textbox(label="Query", value="warm overdrive tone for blues")
        k2 = gr.Slider(1, 10, value=3, step=1, label="Top-k")
        out2 = gr.Dataframe(headers=["item", "score"], label="Similar items")
        btn2 = gr.Button("Search")
        btn2.click(sim_search, [query, k2], out2)

if __name__ == "__main__":
    demo.launch()
