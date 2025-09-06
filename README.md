# Robust: Evaluating LLM Robustness in Text-to-SQL with Noisy and Multilingual Schemas

## Overview
Large Language Models (LLMs) have recently shown strong performance on **text-to-SQL benchmarks** like the [BIRD Benchmark](https://bird-bench.github.io/). However, real-world databases are rarely clean. 
This project uses the mini dev dataset from the **BIRD benchmark** and implements a robustness evaluation pipeline that applies controlled perturbations to schemas and measures model accuracy across multiple categories of noise:
- Nominal: unperturbed data (baseline).
- Typos: changes in characters simulating spelling mistakes.
- Abbreviations: common shortened forms (e.g., "no." for "number").
- Synonyms: lexical variation using alternative words. (e.g. "customers" for "clients")
- Translations: round-trip translation introducing paraphrastic variation (tried some English to French translation)

## How to run 

```bash
## First install the code
git clone git@github.com:jmsardain/robust.git 
cd robust

## Run the code 
python main.py --model <model_name> --syst <noise>   --n <number_of_events>
# model_name = qwen or gpt-4o-mini
# noise = {nominal, typo, abbrev, synonyms, translate}
# number of events = -1 is all, N otherwise 
```

## Results

Evaluated two models using the provided framework 

| Model                            | Nominal | Typos  | Abbereviation | Synonyms | Translate | 
| -------------------------------- | ------- | -----  | ------------- | -------- | --------- | 
| GPT-4o-mini                      | 0.1902  | 0.0106 | 0.000         | 0.1901   | 0.1731    | 
| Qwen/Qwen2.5-Coder-0.5B-Instruct | 0.0128  | 0.0000 | 0.000         | 0.0107   | 0.0064    | 
| Qwen/Qwen2.5-7B-Instruct         | 0.107   | 0.0021 | 0.000         | 0.0085   | 0.0085    | 


## Observations 

- GPT-4o-mini shows moderate robustness on synonyms and translations, but fails on abbreviations and typos.
- Qwen models struggles across all perturbations, indicating limited robustness for this classification task.
- Robustness to synthetic noise remains an open challenge and highlights the gap between nominal accuracy and real-world performance.

## Problems/Questions ?
For questions or contributions, please contact me and open an issue 🍻