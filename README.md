# ACMMM25
This is the official repo of team DeepSIX partipated in Responsible AI challenge @ ACMMM 25: Multimodal Hallucination Detection and Fact Checking
# Task 1:
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

# Task 2: Multimodal Fact Checking
![alt text](/ACMMM25/task2/image.png)

This task focuses on verifying the factual accuracy of claims by analyzing multimodal inputs. Participants are given:
• A claim in textual form, which can be a news headline, a sentence from an article, or a social media post.
• An accompanying image related to the claim.
• Additional context in textual form, such as the full text of the news article, a related social media discussion, or other supplementary information.
Participants must determine the factual accuracy of the claim based on all the provided inputs. They need to assign one of four possible labels to the claim: "True", "False", "Partially True", and "Not Verifiable". This task is treated as a four-class classification problem, requiring participants to consider both the visual and textual evidence to assess the claim’s factuality comprehensively.

For this task, we will provide a dataset with a train set ，a dev set and a test set. The annotations are in the form of a JSON file. An example of the task metadata is shown below.

```json
"metadata": {
    "image_id": "0234613.jpg",
    "claim": "Corbin Aoyagi a supporter of gay marriage waves a rainbow flag during a rally at the Utah State Capitol on Jan 28",
    "context": "The attorneys general of Virginia and Utah are bringing their state same-sex marriage bans to the Supreme Court. Utah Attorney General Sean Reyes filed a petition with the court, seeking a review of the ruling that struck down Utah's ban on same-sex marriage. Virginia Attorney General Mark Herring plans to do the same, arguing the ban is discriminatory. There is a push for a swift resolution due to the numerous legal victories for same-sex marriage advocates following last summer's Supreme Court decision. Utah's petition questions if the 14th Amendment prevents states from defining marriage as only between a man and a woman. Reyes emphasizes his duty to defend the state's constitution. Herring supports a quick final resolution to affirm marriage rights for all Virginians.",
    "question": "Based on the provided information, please determine whether the claim is factual.",
    "correct_answer": "True",
    "correct_choice": "A",
    "choices": [
        {
            "id": "A",
            "choice": "True"
        },
        {
            "id": "B",
            "choice": "False"
        },
        {
            "id": "C",
            "choice": "Partially True"
        },
        {
            "id": "D",
            "choice": "Not Verifiable"
        }
    ]
}
```

## Evaluation

The F1 score is computed using precision (P) and recall (R), which are calculated as follows:

`P = TP / ( TP + FP )`

`R = TP / ( TP + FN )`

`F1 = 2 * P * R / ( P + R )`

where TP, FP, and FN represent specific items that are used to calculate the F1 score in the context of a Confusion_matrix. In particular, when computing the micro-F1 score, TP corresponds to the number of predicted tuple that match exactly with those in the gold set.

## Dataset

We will utilize synthetic and real-world multimodal datasets for these tasks, including balanced distributions of various scenarios. Table I and Table II are the dataset statistics. The datasets include diverse scenarios such as indoor, outdoor, social, and news contexts to ensure robust evaluation.

| Field                  | Train  | Dev   | Test  |
|------------------------|--------|-------|-------|
| Number of claims       | 2800   | 200   | 1000  |
| Average context length | 120 words | 115 words | 122 words |
| Number of images       | 2800   | 200   | 1000  |
| Factuality labels      | 4      | 4     | 4     |
| Scene categories       | 16     | 16    | 16    |


## Timeline

- Registration Opens: `March 20, 2025`
- Training Data Release: `March 30, 2025`
- Challenge Result Submission Deadline:	`May 20, 2025`
- Leaderboard Release: `June 1, 2025`
- Challenge Paper Submission Deadline: `June 15, 2025`