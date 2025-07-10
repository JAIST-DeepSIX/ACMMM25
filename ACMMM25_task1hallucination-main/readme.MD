## Project Overview

This repository contains the pipeline and data for detecting hallucinations in image descriptions. The goal is to segment AI-generated descriptions (`system1_answer`), extract individual claims with their associated keywords, and then verify each claim against the image to identify hallucinated elements.

---
## Result:
TP=880, FP=120, FN=120
Precision=0.8800, Recall=0.8800, F1=0.8800
Micro-F1: 0.8800
## Repository Structure

```
.
├── 1_extract_segment.py
├── 2_extract_kpfromclaim.py
├── 3_delete_emptykeywords_claim.py
├── 4_fs_verifier_vllm.py
├── 5_final_verifier.py
├── 6_final_final.py
├── 7_eval.ipynb
├── data
│   ├── images
│   └── json
│       ├── test.json
│       ├── train.json
│       └── val.json
├── output
│   ├── 1_to_segment.json
│   ├── 2_to_keyphrase.json
│   ├── 3_data_onlyclaim_wkeyphrase.json
│   ├── 4_CoT_ClaimKeyword_wPredict.json
│   ├── 5_pre_final.json
│   └── 6_final_final.json
├── prompt
│   ├── 1_extract_segment.yaml
│   ├── 2_extract_claim_keywords.yaml
│   ├── 4_fs_verifier_vllm
│   │   ├── 4_fs_verifier_vllm_cot.yaml
│   │   └── 4_fs_verifier_vllm_cot_assistant.yaml
│   └── 5_verifier_final_keyword_fs.yaml
├── readme.MD
├── requirements.txt
└── utils
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-310.pyc
    │   └── data_utils.cpython-310.pyc
    └── data_utils.py
```

---

## Data Format

Each record in `output/6_final_final.json` follows this structure:

```json
{
  "image_id": "11e636d22f2e9bd4.jpg",
  "system1_answer": "...full paragraph of AI-generated description...",
  "choices": [
    { "id": "A", "choice": "No hallucination" },
    { "id": "B", "choice": "blue-colored shelves" },
    { "id": "C", "choice": "neatly organized" },
    { "id": "D", "choice": "visible price tags" }
  ],
  "prediction": ["B"],
  "correct_choice": "B",
  "correct_answer": "blue-colored shelves"
}
```

### Sample Record Explanation

* **image\_id**: Filename of the input image.
* **system1\_answer**: Full description from `system1` to be broken into segments.
* **choices**: List of candidate keywords to match against claims.
* **prediction**: Model’s selected choice(s).
* **correct\_choice**: Ground-truth label.
* **correct\_answer**: Text of the ground-truth choice.

---

## Inference Pipeline

1. **Segment Description**

   * Use `1_extract_segment.py` with Qwen2.5-72B-Instruct to split `system1_answer` into logical segments.
2. **Extract Claims & Keywords**

   * Run `2_extract_kpfromclaim.py` using the segmentation output. This script queries Qwen2.5-72B-Instruct to pair each segment with matching keywords from `choices`.

3. **Detect Hallucinations**

   * Feed the extracted claims to from `3_delete_emptykeywords_claim.py` -> `6_final_final.py` which uses Qwen2.5-VL-72B-Instruct with a few-shot prompt to classify each claim as hallucinated or not.

---

## Installation & Setup

1. Clone this repository and navigate to its root:

   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```
2. Create a Python environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Dataset issue:

In the dataset provided by the organizer, we discovered some cases that were mislabeled or contained more than one hallucinated item.
Examples:

1. image_id: 86d3dc21a3458072

    - Issue: We believe this car is an AC Cobra, not a Corvette, yet the correct_choice is “red racing stripes.” This suggests that both “silver Corvette” and “red racing stripes” were treated as equally valid hallucinations.

2. image_id: 16c15b29b30148c2

    - Issue: The model flagged two hallucinations: “dull yellow plate” (which isn’t actually dull) and “black bumper.” However, because only one correct answer is permitted, the model ultimately chose “B.”

3. ...

We believe there are still additional mislabeled cases in this dataset, which prevents our method from achieving top-notch performance.