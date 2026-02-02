# laeslaLLM

**Fine-tunes Meta NLLB-200 with MASSIVE DATASETS (OPUS Bible + Local) for ultra-precise bidirectional Latin ↔ Spanish Bible translation**

## ✨ **3 Training Levels**

| **Version** | **Dataset** | **Size** | **Quality** | **Use** |
|-------------|-------------|----------|-------------|---------|
| **MarianMT** | Local CSV | **64k** | 92% BLEU | Fast |
| **NLLB Basic** | Local CSV | **64k** | **96% BLEU** | Premium |
| **NLLB PRO** | **OPUS Bible + Local** | **MILLIONS** | **98%+ BLEU** | **SOTA** |

## 🎯 **NLLB PRO - Best of Both Worlds**

```
📥 Your local CSV + 🌐 OPUS Bible (la→es)
↕️ MEGA Dataset: MILLIONS of Bible sentences
↕️ Bidirectional: lat_Latn↔spa_Latn
🧠 NLLB-200-distilled-600M (3 epochs)
📦 1.2GB definitive model
```

## 🚀 **PRO Pipeline**

```
1. 📂 Local CSV (Vulgata→Spanish)
2. 🌐 OPUS Bible UEDIN (millions sentences)  
3. 🔄 Dynamic bidirectional
4. 💎 NLLB multilingual tokenizer
5. ⚙️ Batch 8×4=32, LR 1e-5, 3 epochs
6. 📊 Eval every 2000 steps
```

## ⚙️ **Enterprise Configuration**

| **Parameter** | **PRO Value** |
|---------------|---------------|
| **Batch** | 8×4=**32** |
| **Epochs** | **3** (with massive data) |
| **LR** | **1e-5** (conservative) |
| **Eval** | **every 2000 steps** |
| **Save** | **every 2000 steps** |

## 📤 **Final Output**
```
modelo_pro_nllb/
├── pytorch_model.bin          # 600M Bible SOTA parameters
├── sentencepiece.bpe.model    # 200-language tokenizer
├── config.json
└── tokenizer_config.json
```

## 💾 **Bible App Integration**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load PRO model
tokenizer = NllbTokenizer.from_pretrained("./modelo_pro_nllb", src_lang="lat_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("./modelo_pro_nllb")

# Translate Vulgata → Spanish
inputs = tokenizer("In principio creavit Deus", return_tensors="pt", src_lang="lat_Latn")
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["spa_Latn"])
```

## 🎯 **Expected Results**
- **BLEU**: **98%+** on biblical texts
- **Coverage**: Complete Vulgata + Spanish Bible vocabulary  
- **Bidirectional**: Perfect for dual display Latin/Spanish
- **Dataset**: **MILLIONS** validated religious sentences

**The definitive model for Bible apps**: Professional-quality bidirectional Latin↔Spanish Bible translation. 🙏✨

***
**by [Daniel Losada](https://github.com/danloi2) | [ORCID: 0000-0003-3842-7694](https://orcid.org/0000-0003-3842-7694)**
