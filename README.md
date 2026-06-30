# FICO Score Quantization — JPMorgan Chase Quantitative Research Task 4

## Overview
Maps continuous FICO credit scores (300–850) into optimal rating buckets 
to enable machine learning models to predict mortgage default probability.

## Problem Context
The risk team suspects FICO scores are a strong predictor of mortgage default.
Charlie (the ML engineer) needs categorical inputs for her model architecture.
The task: given n buckets, find the boundary points that best capture the 
default signal in the data — a process called **quantization**.

## Approach

### Data Preparation
- Loaded 10,000 borrower records with FICO scores and default labels (0/1)
- Grouped by unique FICO score, computing total borrowers and defaults per score
- Built cumulative sum arrays for O(1) range queries during DP

Uses **dynamic programming** to maximize the log-likelihood function:

LL = Σ [ kᵢ·ln(pᵢ) + (nᵢ − kᵢ)·ln(1 − pᵢ) ]

Where nᵢ = borrowers per bucket, kᵢ = defaults, pᵢ = default probability.

## Results
Optimal boundaries for 5 buckets (n=5):

| Rating | FICO Range | Credit Quality |
|--------|-----------|----------------|
| 1      | 408 – 520 | Best           |
| 2      | 521 – 580 | Good           |
| 3      | 581 – 640 | Fair           |
| 4      | 641 – 696 | Poor           |
| 5      | 697 – 850 | Worst          |

Best log-likelihood score: **-4255.38**

## Tech Stack
- Python 3.9
- Pandas, NumPy
- Dataset: 10,000 mortgage borrowers

## AI Usage
Used Claude throughout this task in a Socratic learning mode.
Claude asked guiding questions rather than giving answers directly. Concepts 
learned through this process include:

- Why cumulative sums enable O(1) range queries vs O(n) recomputation
- The DP state definition: dp[i][k] and what i and k represent
- Why the inner loop over all j (previous cut points) is necessary
- Edge case handling: log(0) when a bucket has zero defaults
- Boundary reconstruction via traceback through the split table

All code was written by me line by line with Claude's guidance.
The mathematical reasoning, bug identification, and debugging were 
collaborative. Claude identified issues, I implemented the fixes.
