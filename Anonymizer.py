import os
import logging
import json
import random
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import login

# add Hugging Face token
login(token="")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# "Style Profiling" and "Zero-Shot Rewriting" as described in Section 3

class LlamaHFLLM:
    def __init__(
        self,
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        device=DEVICE
    ):
        logging.info(f"Loading pipeline for model: {model_id}")
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.system_prompt = "You are a helpful assistant, that responds as instructed."

    def format_prompt(self, user_input: str) -> list:

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        return messages

    @torch.no_grad()
    def generate(self, user_prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):

        try:
            # Prompt formatting
            formatted_prompt = self.format_prompt(user_prompt)

            # Generation
            outputs = self.pipe(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.2,  
            )

            
            generated_text = outputs[0]["generated_text"][-1]["content"]

            response = generated_text.strip()
            if self.system_prompt in response:
                response = response.replace(self.system_prompt, "")
            if user_prompt in response:
                response = response.replace(user_prompt, "")

            return response.strip()
        except Exception as e:
            logging.error(f"Generation failed: {str(e)}")
            return ""

    def get_style_profile_from_llm(self, prompt_content, max_retries=3): #(Style Profiling) - build a concise per-author summary (Section 3.1)

        for attempt in range(1, max_retries + 1):
            result = self.generate(
                prompt_content,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9
            )
            if result.strip():
                return result
            logging.warning(f"Attempt {attempt}: Empty result. Retrying...")
        return "Style analysis unavailable"

    def rewrite_text_once(self, original_text, style_info): # (Zero-Shot Rewriting)  mask author-specific cues as described in Section 3.2

        prompt_content = (
            f"Here is the author's style profile:\n{style_info}\n\n"
            "Rewrite the following text so that it does NOT reflect these style cues,"
            " but retains the original meaning:\n\n"
            f'"""{original_text}"""\n\n'
            f"Rewritten text (No commentary):"
        )
        return self.generate(
            prompt_content,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.7
        )

    def rewrite_text_with_retry(self, original_text, style_info):

        rewrite = self.rewrite_text_once(original_text, style_info)
        if rewrite.strip():
            return rewrite
        logging.warning(f"First rewrite empty for text: {original_text[:50]}... Retrying")
        return self.rewrite_text_once(original_text, style_info)

def build_style_prompt(combined_text, prompt_type): # Selecting which style dimension(s) to summarise
    focus_map = {
        'length': "Sentence length and structure",
        'vocab': "Vocabulary choice",
        'tone': "Tone",
        'punc': "Common punctuation patterns"
    }
    if prompt_type == 'full':
        instruction = (
            "Below are example texts from a single author:\n\n"
            + combined_text
            + "\n\nPlease summarize the author's writing style in bullet points. "
            f"Focus on details like:\n"
            "- Sentence length and structure\n"
            "- Vocabulary choice\n"
            "- Tone\n"
            "- Common punctuation patterns\n\n"
            f"Texts:\n{combined_text}"
            "Style summary:"
        )
    else:
        instruction = (
            f"Analyze these texts to identify the author's {focus_map[prompt_type]} only.\n"
            f"Provide a concise overall summary (without quoting individual examples) of the author's {focus_map[prompt_type]} in bullet points.\n\n"
            f"Texts:\n'''{combined_text}'''\n\n"
            f"Summary of author's {focus_map[prompt_type]}:"
        )
    return instruction

def anonymize_with_train_test(
    train_csv,
    test_csv,
    model_id,
    prompt_type,
    max_examples,
    rewrite_train=False
):
    experiment_tag = f"llama_{prompt_type}_{max_examples}ex"
    outputs = {
        "styles": f"llama_outputs/style_profiles_{experiment_tag}.json",
        "test": f"llama_outputs/test_anonymized_{experiment_tag}.csv",
        "train": f"llama_outputs/train_anonymized_{experiment_tag}.csv",
        "log": f"llama_outputs/log_{experiment_tag}.log"
    }

    if not all(os.path.isfile(f) for f in [train_csv, test_csv]):
        raise FileNotFoundError("Missing train/test CSV files")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Initialize model
    llm = LlamaHFLLM(model_id=model_id)

    # Build style profiles
    style_profiles = {}
    for author_id in tqdm(train_df["id"].unique(), desc="Building style profiles"):
        author_texts = train_df.loc[train_df["id"] == author_id, "text"].tolist()
        if not author_texts:
            style_profiles[author_id] = "No style info available"
            continue
        sample_texts = random.sample(author_texts, min(len(author_texts), max_examples))
        combined_text = "\n\n".join(sample_texts)
        prompt_content = build_style_prompt(combined_text, prompt_type)
        profile_text = llm.get_style_profile_from_llm(prompt_content)
        style_profiles[str(author_id)] = profile_text

    # Save profiles
    os.makedirs(os.path.dirname(outputs["styles"]), exist_ok=True)
    with open(outputs["styles"], "w", encoding="utf-8") as f:
        json.dump(style_profiles, f, indent=2, ensure_ascii=False)

    # Rewrite texts
    def process_dataset(df, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(output_path)}"):
            original_text = row["text"]
            style_info = style_profiles.get(row["id"], "No style info available")
            anonymized_text = llm.rewrite_text_with_retry(original_text, style_info)
            results.append({
                **row.to_dict(),
                "original_text": original_text,
                "anonymized_text": anonymized_text
            })
        pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8")

    process_dataset(test_df, outputs["test"])
    if rewrite_train:
        process_dataset(train_df, outputs["train"])

    logging.info(f"Experiment {experiment_tag} completed successfully")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run style anonymization experiments")
    parser.add_argument("--prompt_type", required=True,
                        choices=['length', 'vocab', 'tone', 'punc', 'full'])
    parser.add_argument("--max_examples", type=int, required=True,
                        help="Number of examples for style profiling (5 or 10 or 20)")
    parser.add_argument("--train_csv", default="author10_train.csv")
    parser.add_argument("--test_csv", default="author10_test.csv")
    parser.add_argument("--model_id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--rewrite_train", action="store_true",
                        help="Anonymize training set in addition to test set")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID to use for this experiment")
    args = parser.parse_args()

    if torch.cuda.is_available():
        DEVICE = f"cuda:{args.gpu}"
    else:
        DEVICE = "cpu"

    experiment_tag = f"{args.prompt_type}_{args.max_examples}ex"
    logging.basicConfig(
        filename=f"anonymizer_{experiment_tag}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )

    # Run experiment
    logging.info(f"Starting experiment {experiment_tag} on {DEVICE}")
    anonymize_with_train_test(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        model_id=args.model_id,
        prompt_type=args.prompt_type,
        max_examples=args.max_examples,
        rewrite_train=args.rewrite_train
    )