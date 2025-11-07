# 🤖 LLM Ensemble Guide - 5 Model Weighted Voting

## Overview

Your basketball system now uses **5 specialized LLM models** for game predictions with weighted voting:

| Model | Size | Weight | Specialty |
|-------|------|--------|-----------|
| **Mistral-7B Instruct** | 4.1GB | 4.2 | General reasoning (BEST) |
| **OpenChat-7B** | 4.1GB | 4.1 | Conversational analysis (2nd) |
| **Dolphin-Mistral-7B** | 4.1GB | 4.0 | Contrarian views |
| **CodeLlama-7B** | 4.1GB | 4.0 | Analytical reasoning |
| **Neural-Chat-7B** | 3.9GB | 3.9 | Sports insights |

**Total Size:** ~20GB
**Runs Locally:** No API costs!

---

## Quick Setup (5 Steps)

### 1. Install Ollama

**Linux/Mac:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from https://ollama.com/download

### 2. Start Ollama Service

```bash
# In a separate terminal:
ollama serve
```

Or on Mac/Linux, Ollama may auto-start as a service.

### 3. Download Models

```bash
# Run our setup script (downloads all 5 models)
./setup_ollama_models.sh
```

This takes 10-30 minutes depending on internet speed.

**Manual download:**
```bash
ollama pull mistral:7b-instruct
ollama pull openchat:7b
ollama pull dolphin-mistral:7b
ollama pull codellama:7b-instruct
ollama pull neural-chat:7b
```

### 4. Check Installation

```bash
python basketball_main.py --check-llm-models
```

Should show all 5 models with ✅

### 5. Test It!

```bash
python basketball_main.py --llm-test
```

Runs a Duke vs UNC prediction using all 5 models.

---

## Usage

### Check Installed Models

```bash
python basketball_main.py --check-llm-models
```

### Test Ensemble (Sample Game)

```bash
python basketball_main.py --llm-test
```

Predicts: Duke vs North Carolina with all 5 models.

### Interactive Prediction

```bash
python basketball_main.py --llm-predict
```

Enter any teams and get weighted ensemble prediction!

### Standalone Python

```python
import asyncio
from basketball_llm_ensemble import BasketballLLMEnsemble

async def predict():
    ensemble = BasketballLLMEnsemble()

    result = await ensemble.predict_game(
        home_team="Duke",
        away_team="North Carolina",
        home_stats={"win_rate": "0.750", "ppg": "78.5"},
        away_stats={"win_rate": "0.600", "ppg": "72.1"}
    )

    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.1%}")

asyncio.run(predict())
```

---

## How It Works

### 1. Each Model Analyzes the Game

All 5 models receive the same prompt with:
- Team names
- Win rates, PPG, efficiency stats
- Recent form
- Additional context

Each model returns:
- **Prediction:** HOME or AWAY
- **Confidence:** 0.50 to 1.00
- **Reasoning:** 2-3 sentence analysis

### 2. Weighted Voting

Models vote with their weights:
- Mistral (4.2 weight) has most influence
- Neural-Chat (3.9 weight) has least

Example:
```
Mistral:      HOME (0.75 confidence) → 4.2 × 0.75 = 3.15
OpenChat:     HOME (0.68 confidence) → 4.1 × 0.68 = 2.79
Dolphin:      AWAY (0.62 confidence) → 4.0 × 0.62 = 2.48
CodeLlama:    HOME (0.71 confidence) → 4.0 × 0.71 = 2.84
Neural-Chat:  HOME (0.65 confidence) → 3.9 × 0.65 = 2.54

HOME score: 3.15 + 2.79 + 2.84 + 2.54 = 11.32
AWAY score: 2.48

FINAL: HOME wins (11.32 > 2.48)
Confidence: 11.32 / (11.32 + 2.48) = 82%
```

### 3. Output

You get:
- **Final Prediction:** HOME or AWAY
- **Ensemble Confidence:** 0.0 to 1.0
- **Vote Breakdown:** {"home": 4, "away": 1}
- **Individual Model Reasoning:** See what each model thought

---

## Performance

**Prediction Time:** 30-60 seconds (queries 5 models sequentially)

**Accuracy:** TBD (track with `--update-outcomes` and `--monitor-performance`)

**Cost:** $0 (runs locally!)

**GPU:** Optional but speeds up predictions significantly

---

## Troubleshooting

### "ollama not installed"

Install ollama package:
```bash
pip install ollama
```

### "Ollama service not running"

Start the service:
```bash
ollama serve
```

In separate terminal or as background service.

### "Models not available"

Download models:
```bash
./setup_ollama_models.sh
```

Or check what's installed:
```bash
ollama list
```

### Predictions are slow

**Normal:** 30-60 seconds for 5 models on CPU.

**Speed up:**
- Use GPU if available (Ollama auto-detects)
- Reduce models (edit `basketball_llm_ensemble.py`)
- Query models in parallel (advanced)

### Out of memory

**20GB RAM recommended.** If you have less:

Option 1: Use quantized models (smaller):
```bash
ollama pull mistral:7b-instruct-q4_0  # 4-bit quantized
```

Option 2: Reduce number of models in `basketball_llm_ensemble.py`

---

## Advanced

### Change Model Weights

Edit `basketball_llm_ensemble.py`:

```python
self.models = [
    ModelConfig(
        name="Mistral",
        model_id="mistral:7b-instruct",
        weight=4.2,  # ← Change this
        ...
    ),
    ...
]
```

### Add More Models

```python
self.models.append(
    ModelConfig(
        name="Llama3",
        model_id="llama3:8b",
        weight=4.5,
        specialty="Latest reasoning",
        temperature=0.7
    )
)
```

Then download:
```bash
ollama pull llama3:8b
```

### Switch to GGUF Files (Direct)

If you have GGUF files downloaded, you can use `llama-cpp-python` instead of Ollama. Edit `basketball_llm_ensemble.py` to use `llama_cpp` instead of `ollama`.

**Not recommended** - Ollama is simpler!

---

## FAQ

**Q: Can I use this instead of traditional ML models?**
A: Yes! But we recommend using BOTH:
- LLM ensemble for qualitative analysis
- ML models for quantitative predictions
- Combine them for best results

**Q: Which is better - LLM or ML?**
A: Different strengths:
- **LLM:** Context, injuries, narratives, rivalries
- **ML:** Statistics, patterns, speed, scalability

**Q: Can I use cloud LLMs (OpenAI/Claude)?**
A: Yes, but costs add up. Local Ollama is free!

**Q: How much RAM needed?**
A: 20GB recommended. Models loaded one at a time.

**Q: GPU required?**
A: No, but helpful. CPU works fine (slower).

**Q: Can I switch back to regular ML?**
A: Yes! LLM is optional. All your ML code still works.

---

## Next Steps

1. **Run a test:** `python basketball_main.py --llm-test`
2. **Track accuracy:** Use with `--update-outcomes` and `--monitor-performance`
3. **Compare:** Run both LLM and ML predictions, see which performs better
4. **Hybrid:** Use LLM for high-stakes games, ML for volume

**Your football system uses similar LLMs** - you can compare approaches!

---

Questions? Check the code in `basketball_llm_ensemble.py` 🏀
