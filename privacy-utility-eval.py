import os
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    DebertaV2Tokenizer, DebertaV2ForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import evaluate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score


class NotebookEvaluator:
    def __init__(self, data_dir, dataset_type, eval_type,
                 privacy_model_dir, output_dir):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.eval_type = eval_type
        self.privacy_model_dir = privacy_model_dir
        self.output_dir = output_dir

        self.results = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "thenlper/gte-small"
        ]

        # ------ Utility setup --------------------
        # (CS, BLEU, PPL as in Table 1)

        if self.eval_type in ["utility", "both"]:

            self.embed_models = {
                name: SentenceTransformer(name)
                for name in self.embedding_models
            }
            # GPT-2 for perplexity
            self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")\
                .to(self.device)
            self.gpt2_model.eval()
            # BLEU metric
            self.bleu_metric = evaluate.load("bleu")

        # ------ Privacy setup --------------------
        # (Choosing the Authorship classifier based on the dataset)
        if self.eval_type in ["privacy", "both"]:
            if self.dataset_type == "blog":
                self.privacy_tokenizer = DebertaV2Tokenizer.\
                    from_pretrained(self.privacy_model_dir)
                self.privacy_model = DebertaV2ForSequenceClassification.\
                    from_pretrained(self.privacy_model_dir)\
                    .to(self.device)
            else:
                from transformers import BertTokenizer, BertForSequenceClassification
                self.privacy_tokenizer = BertTokenizer.from_pretrained(
                    'bert-base-uncased'
                )
                self.privacy_model = BertForSequenceClassification.\
                    from_pretrained(self.privacy_model_dir)\
                    .to(self.device)
            self.privacy_model.eval()

    def compute_ppl(self, texts): # Perplexity metric
        total_ll = 0.0
        total_tokens = 0
        for text in tqdm(texts, desc="Computing PPL"):
            if not text.strip():
                continue
            inputs = self.gpt2_tokenizer(
                text, return_tensors="pt",
                max_length=512, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.gpt2_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            if not torch.isnan(loss) and not torch.isinf(loss):
                tokens = inputs["input_ids"].size(1)
                total_ll += loss.item() * tokens
                total_tokens += tokens
        return np.exp(total_ll / total_tokens) if total_tokens else np.nan

    def compute_cosine_sim(self, orig_texts, anon_texts): #Cosine similarity - averaged over three encoders
        results = {}
        batch_size = 32
        for name, model in self.embed_models.items():
            short = name.split("/")[-1]
            sims = []
            for i in tqdm(range(0, len(orig_texts), batch_size),
                          desc=f"Cosine ({short})"):
                ob = orig_texts[i:i+batch_size]
                ab = anon_texts[i:i+batch_size]
                emb_o = model.encode(ob)
                emb_a = model.encode(ab)
                sim_m = cosine_similarity(emb_o, emb_a)
                sims.extend(sim_m.diagonal().tolist())
            avg = np.mean(sims)
            results[f"cosine_{short}"] = avg
        all_vals = [v for k, v in results.items() if k.startswith("cosine_")]
        results["cosine_avg"] = np.mean(all_vals)
        return results

    def compute_bleu(self, orig_texts, anon_texts): #BLEU Score
        return self.bleu_metric.compute(
            predictions=anon_texts,
            references=[[t] for t in orig_texts]
        )["bleu"]

    def predict_batch(self, texts): # Authorship classifier inference
        preds = []
        for i in tqdm(range(0, len(texts), 32), desc="Predicting batches"):
            batch = texts[i:i+32]
            inputs = self.privacy_tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.privacy_model(**inputs)
            preds.extend(torch.argmax(out.logits, dim=1).tolist())
        return preds

    def process_file(self, csv_path):
        df = pd.read_csv(csv_path)
        base = os.path.basename(csv_path)
        logging.info(f"Processing {base}")
        file_res = {
            "filename": base,
            "num_samples": len(df)
        }
        orig = df["original_text"].astype(str).tolist()
        anon = df["anonymized_text"].astype(str).tolist()

        if self.eval_type in ["utility", "both"]:
            file_res["cosine_sim"] = self.compute_cosine_sim(orig, anon)
            file_res["bleu"]       = self.compute_bleu(orig, anon)
            file_res["ppl_orig"]   = self.compute_ppl(orig)
            file_res["ppl_anon"]   = self.compute_ppl(anon)

        if self.eval_type in ["privacy", "both"]:
            true = df["id"].tolist()
            p_o = self.predict_batch(orig)
            p_a = self.predict_batch(anon)
            file_res.update({
                "acc_orig": np.mean(np.array(true) == np.array(p_o)),
                "acc_anon": np.mean(np.array(true) == np.array(p_a)),
                "f1_orig":  f1_score(true, p_o, average="macro"),
                "f1_anon":  f1_score(true, p_a, average="macro")
            })

        self.results.append(file_res)

    def save_results(self):
        os.makedirs(self.output_dir, exist_ok=True)
        df = pd.DataFrame(self.results)
        out = os.path.join(self.output_dir, "evaluation_results.csv")
        df.to_csv(out, index=False)
        logging.info(f"Saved results to {out}")
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate anonymization utility & privacy metrics"
    )
    parser.add_argument("--data-dir",      default="for_eval_llama")
    parser.add_argument("--dataset-type",  choices=["blog","illinois"],
                        default="blog")
    parser.add_argument("--eval-type",     choices=["utility","privacy","both"],
                        default="both")
    parser.add_argument("--privacy-dir",   default="Deberta_trainer")
    parser.add_argument("--output-dir",    default="llama-blog-results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    evaluator = NotebookEvaluator(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        eval_type=args.eval_type,
        privacy_model_dir=args.privacy_dir,
        output_dir=args.output_dir
    )

    logging.info(f"Data dir contents: {os.listdir(evaluator.data_dir)}")
    logging.info(f"Privacy model dir contents: {os.listdir(evaluator.privacy_model_dir)}")

    for fname in os.listdir(evaluator.data_dir):
        if fname.endswith(".csv"):
            evaluator.process_file(os.path.join(evaluator.data_dir, fname))

    # saving
    results_df = evaluator.save_results()
    logging.info("Evaluation Results Preview:\n" + results_df.head().to_string())


if __name__ == "__main__":
    main()
