# I Am No One: Style-Aware Paraphrasing for Text Anonymization

*(InterSpeech 2026 anonymous submission)*
### **The supplementary material can be found in 'supplementary.pdf'**

## Quick Start

### **Step 0 - Create & Activate Environment**

```bash
conda create -n sapta python=3.10
conda activate sapta
pip install -r requirements.txt
```

### **Step 1 - Hugging Face Login**

```bash
huggingface-cli login --token YOUR_HF_TOKEN
```

### **Step 2 - Anonymize**

```bash
python Anonymizer.py \  
  --prompt_type full \
  --max_examples 5 \
  --train_csv data/author10_train.csv \  ## For author10 (you can use your own dataset)
  --test_csv  data/author10_test.csv
```

### **Step 3 - Evaluate Privacy & Utility** 

```bash
python privacy-utility-eval.py \   
  --data-dir   llama_outputs \
  --dataset-type blog \
  --eval-type  both \ 
  --privacy-dir Deberta_trainer       ## privacy classifier trained on author10 (user bert_aa for illinois dataset)
```

### **Step 4 - Compute Weighted KL (optional)**

```bash
python weighted_kl.py \      
  --input_csv  llama_outputs/test_anonymized_llama_full_5ex.csv \
  --output_csv llama_outputs/kl_results.csv
```

---

## Script Reference

| Script                      | Purpose                                  | Core Arguments                                                                                                        |
| --------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Anonymizer.py**           | Build style profiles & rewrite text      | `--prompt_type {full,length,vocab,tone,punc}` · `--max_examples` (2 / 5 / 10 / 20) · `--train_csv` · `--test_csv`     |
| **privacy-utility-eval.py** | Compute cosine, BLEU, PPL, authorship F1 | `--data-dir` directory with anonymized CSVs · `--dataset-type {blog,illinois}` · `--eval-type {utility,privacy,both}` |
| **weighted\_kl.py**         | TF–IDF-weighted KL divergence            | `--input_csv` anonymized CSV · `--model_name` HF tokenizer name                                                       |

---
