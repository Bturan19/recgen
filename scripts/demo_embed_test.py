import sys
import os

import torch

sys.path.insert(0, "space")
os.environ["RECGEN_MODEL_DIR"] = "models/SmolLM2-360M"
os.environ["RECGEN_DEMO_DEVICE"] = "cpu"
import app

ITEMS = [
    "Item: Fender Player Stratocaster. Category: electric guitar. Single-coil pickups, classic twang, versatile clean tones.",
    "Item: Gibson Les Paul Standard. Category: electric guitar. Humbuckers, mahogany body, warm thick sustain, rock tone.",
    "Item: Boss DS-1 Distortion Pedal. Category: effects pedal. Legendary hard rock distortion, scooped mids.",
    "Item: Ibanez TS9 Tube Screamer. Category: effects pedal. Vintage overdrive, warm mid boost, blues lead tone.",
    "Item: Ernie Ball Slinky Strings. Category: strings. Nickel wound 10-46, bright punchy tone.",
    "Item: D'Addario NYXL Strings. Category: strings. 11-49 medium, tuning stability, enhanced midrange.",
    "Item: Fender 8ft Instrument Cable. Category: cable. Braided shield, quiet signal.",
    "Item: Marshall MG30GFX Combo Amp. Category: amplifier. 30 watts, overdrive channel, British crunch.",
    "Item: Shure SM57 Microphone. Category: microphone. Cardioid, industry standard for instrument miking.",
    "Item: Sennheiser HD 280 Headphones. Category: headphones. Closed back, flat response, studio monitoring.",
]

queries = [
    "warm overdrive tone for blues",
    "studio recording microphone",
    "cheap guitar strings for beginners",
    "portable practice amp",
    "headphones for listening",
]

embs = app.embed(ITEMS)
for q in queries:
    qe = app.embed([q])[0]
    scores = torch.nn.functional.cosine_similarity(qe.unsqueeze(0), embs)
    tops = [ITEMS[i].split(".")[0] for i in torch.argsort(scores, descending=True)[:3]]
    print(" ", q, "->", tops)
