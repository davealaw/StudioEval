# Supported Hugging Face Evaluation Datasets in StudioEval

StudioEval currently supports the following Hugging Face datasets for standardized benchmarking.  
Each entry lists known supported datasets including identifier, suggested subset, split, and number of records.

---

## ARC (AI2 Reasoning Challenge)
Science reasoning benchmark with multiple-choice questions.

- **tinyArc**
  - Dataset: [`tinyBenchmarks/tinyArc`](https://huggingface.co/datasets/tinyBenchmarks/tinyArc)
  - Split: `validation`
  - Records: **100**

- **ARC-Challenge**
  - Dataset: [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc)
  - Subset: `ARC-Challenge`
  - Split: `validation`
  - Records: **299**

- **ARC-Easy**
  - Dataset: [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc)
  - Subset: `ARC-Easy`
  - Split: `validation`
  - Records: **570**

### Compatible Related Datasets

Open Book Questions that require multi-step reasoning, use of additional common and commonsense knowledge, and rich text comprehension

- **OpenBookQA**
  - Dataset: [`allenai/openbookqa`](https://huggingface.co/datasets/allenai/openbookqa)
  - Subset: `main`
  - Split: `validation`
  - Records: **500**

---

## GSM8K (Grade School Math 8K)
Grade-school math word problems requiring multi-step reasoning.

- **tinyGSM8K**
  - Dataset: [`tinyBenchmarks/tinyGSM8K`](https://huggingface.co/datasets/tinyBenchmarks/tinyGSM8K)
  - Split: `validation`
  - Records: **100**

- **GSM8K**
  - Dataset: [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k)
  - Subset: `main`
  - Split: `test`
  - Records: **1.32k**

---

## MMLU (Massive Multitask Language Understanding)
Knowledge across 57 academic/professional subjects.

- **tinyMMLU**
  - Dataset: [`tinyBenchmarks/tinyMMLU`](https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU)
  - Subset: `default`
  - Split: `validation`
  - Records: **100**

- **MMLU** - 57 differnt options / subjects
  - Dataset: [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu)
  - Subset: numerous - check website for various options
  - Split: `test`
  - Records: various depending upon subset

---

## TruthfulQA
Tests whether models produce truthful answers vs misconceptions.

- **tinyTruthfulQA**
  - Dataset: [`tinyBenchmarks/tinyTruthfulQA`](https://huggingface.co/datasets/tinyBenchmarks/tinyTruthfulQA)
  - Split: `validation`
  - Records: **100**

- **TruthfulQA**
  - Dataset: [`truthfulqa/truthful_qa`](https://huggingface.co/datasets/truthfulqa/truthful_qa)
  - Subset: `multiple_choice`
  - Split: `validation`
  - Records: **817**

---

## WinoGrande
Winograd Schema Challenge variant: binary fill-in-the-blank reasoning.

- **tinyWinoGrande**
  - Dataset: [`tinyBenchmarks/tinyWinogrande`](https://huggingface.co/datasets/tinyBenchmarks/tinyWinogrande)
  - Split: `validation`
  - Records: **100**

- **WinoGrande**
  - Dataset: [`allenai/winogrande`](https://huggingface.co/datasets/allenai/winogrande)
  - Subset: `winogrande_debiased`
  - Split: `validation`
  - Records: **1.27k**

---

## HellaSwag
Commonsense reasoning via sentence completion.

- **tinyHellaSwag**
  - Dataset: [`tinyBenchmarks/tinyHellaswag`](https://huggingface.co/datasets/tinyBenchmarks/tinyHellaswag)
  - Split: `validation`
  - Records: **100**

- **HellaSwag**
  - Dataset: [`Rowan/hellaswag`](https://huggingface.co/datasets/Rowan/hellaswag)
  - Split: `validation`
  - Records: **10k**

---

## CommonSenseQA
Commonsense multiple-choice reasoning benchmark.

- **CommonSenseQA**
  - Dataset: [`tau/commonsense_qa`](https://huggingface.co/datasets/tau/commonsense_qa)
  - Split: `validation`
  - Records: **1.22k**

---

## LogiQA
Logical reasoning questions from reading comprehension tests.

- **LogiQA**
  - Dataset: [`lucasmccabe/logiqa`](https://huggingface.co/datasets/lucasmccabe/logiqa)
  - Split: `validation`
  - Records: **651**

---

# Notes
- **TinyBenchmarks** (`tinyArc`, `tinyGSM8K`, `tinyMMLU`, `tinyTruthfulQA`, `tinyWinoGrande`, `tinyHellaSwag`) are recommended for **quick evaluation** and smaller models.  
- Full datasets (CommonSenseQA, LogiQA, full TruthfulQA, full ARC, full GSM8K, full MMLU) can be substituted for **deeper benchmarking**.  
- Record counts shown are approximate based on Hugging Face splits as of 2025.  
