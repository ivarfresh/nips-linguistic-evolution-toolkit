# Draft Findings Summary

Generated from raw v4 simulation JSON by `scripts/build_neurips_draft_artifacts.py`.

- Final run JSONs analysed: 647
- Behavioural cells: 42 (42 complete, 0 incomplete)
- Error checkpoint files discovered: 86

## Game-Only Behavioural Regimes

| Model | Condition | n | Mean final dyad reward | SD | Avg sent | Avg returned | Return ratio |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.5 | No noise | 15 | 122.40 | 14.11 | 3.62 | 6.53 | 0.59 |
| Claude Sonnet 4.5 | Neutral framing | 18 | 113.56 | 8.45 | 3.18 | 5.34 | 0.56 |
| Claude Sonnet 4.5 | Positive k=5 | 15 | 147.01 | 3.03 | 4.85 | 13.72 | 0.94 |
| Claude Sonnet 4.5 | Positive k=5 informed | 15 | 148.41 | 1.35 | 4.92 | 13.79 | 0.93 |
| Claude Sonnet 4.5 | Negative k=5 | 15 | 58.53 | 8.56 | 0.43 | 0.47 | 0.22 |
| Claude Sonnet 4.5 | Negative k=5 informed | 15 | 60.76 | 7.98 | 0.54 | 0.58 | 0.23 |
| GPT-5-Nano | No noise | 15 | 138.40 | 9.11 | 4.42 | 7.58 | 0.57 |
| GPT-5-Nano | Neutral framing | 18 | 112.33 | 26.18 | 3.12 | 4.47 | 0.47 |
| GPT-5-Nano | Positive k=5 | 15 | 144.91 | 8.65 | 4.75 | 10.37 | 0.73 |
| GPT-5-Nano | Positive k=5 informed | 15 | 144.73 | 4.99 | 4.74 | 10.53 | 0.75 |
| GPT-5-Nano | Negative k=5 | 15 | 75.73 | 17.15 | 1.29 | 1.31 | 0.31 |
| GPT-5-Nano | Negative k=5 informed | 15 | 76.72 | 10.43 | 1.34 | 1.26 | 0.29 |
| GPT-5-Nano | Bootstrap | 15 | 133.20 | 15.21 | 4.16 | 12.48 | 1.00 |
| GPT-5-Nano | Bootstrap informed | 15 | 135.87 | 6.95 | 4.29 | 12.88 | 1.00 |

## Supported Positive Myth Deltas

| Model | Condition | Myth order | Delta median | Delta SD |
|---|---|---|---|---|
| GPT-5-Nano | Neutral framing | Game -> myth | 37.00 [4.00, 48.00] | -13.07 [-18.84, -6.12] |
| GPT-5-Nano | Negative k=5 | Game -> myth | 10.08 [1.20, 26.04] | -6.21 [-12.55, 1.00] |
| GPT-5-Nano | Positive k=5 informed | Game -> myth | 4.86 [0.54, 9.22] | -3.44 [-4.82, -1.50] |
| GPT-5-Nano | Positive k=5 informed | Myth -> game | 4.86 [0.54, 9.26] | -1.40 [-4.65, 1.43] |
| Claude Sonnet 4.5 | Positive k=5 | Game -> myth | 2.38 [0.28, 3.56] | -1.42 [-2.97, 0.54] |
| GPT-5-Nano | No noise | Myth -> game | 2.00 [2.00, 10.00] | -6.68 [-9.69, -2.53] |

## Supported Negative Myth Deltas

| Model | Condition | Myth order | Delta median | Delta SD |
|---|---|---|---|---|
| GPT-5-Nano | Bootstrap informed | Game -> myth | -20.00 [-42.00, -4.00] | 11.67 [6.51, 15.47] |
| GPT-5-Nano | Bootstrap | Myth -> game | -16.00 [-32.00, -4.00] | 3.03 [-5.21, 12.86] |
| GPT-5-Nano | Bootstrap informed | Myth -> game | -12.00 [-24.00, -2.00] | 7.94 [1.58, 12.66] |

## Supported Variance Reductions

| Model | Condition | Myth order | Delta median | Delta SD |
|---|---|---|---|---|
| GPT-5-Nano | Neutral framing | Game -> myth | 37.00 [4.00, 48.00] | -13.07 [-18.84, -6.12] |
| GPT-5-Nano | Positive k=5 informed | Game -> myth | 4.86 [0.54, 9.22] | -3.44 [-4.82, -1.50] |
| Claude Sonnet 4.5 | No noise | Game -> myth | 4.00 [-24.00, 12.00] | -3.76 [-8.30, -0.11] |
| GPT-5-Nano | No noise | Myth -> game | 2.00 [2.00, 10.00] | -6.68 [-9.69, -2.53] |
| Claude Sonnet 4.5 | Positive k=5 informed | Game -> myth | 1.90 [0.00, 2.70] | -0.58 [-1.05, -0.18] |
| GPT-5-Nano | Positive k=5 | Myth -> game | 1.18 [-0.10, 8.48] | -6.44 [-10.50, -0.60] |
| GPT-5-Nano | Positive k=5 | Game -> myth | -0.12 [-2.04, 3.38] | -6.19 [-10.00, -0.49] |

## Linguistic Convergence Proxy

| Model | Condition | Task order | round obs. | Early Jaccard | Final Jaccard | Slope | Coop words/100 |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.5 | No noise | Game -> myth | 150 | 0.097 | 0.206 | 0.0120 | 2.27 |
| Claude Sonnet 4.5 | No noise | Myth -> game | 150 | 0.095 | 0.187 | 0.0108 | 2.11 |
| Claude Sonnet 4.5 | Neutral framing | Game -> myth | 180 | 0.106 | 0.222 | 0.0152 | 1.75 |
| Claude Sonnet 4.5 | Neutral framing | Myth -> game | 170 | 0.098 | 0.195 | 0.0111 | 2.11 |
| Claude Sonnet 4.5 | Positive k=5 | Game -> myth | 150 | 0.097 | 0.192 | 0.0120 | 1.73 |
| Claude Sonnet 4.5 | Positive k=5 | Myth -> game | 150 | 0.111 | 0.197 | 0.0116 | 1.85 |
| Claude Sonnet 4.5 | Positive k=5 informed | Game -> myth | 150 | 0.120 | 0.210 | 0.0110 | 2.34 |
| Claude Sonnet 4.5 | Positive k=5 informed | Myth -> game | 150 | 0.098 | 0.206 | 0.0130 | 2.27 |
| Claude Sonnet 4.5 | Negative k=5 | Game -> myth | 150 | 0.121 | 0.170 | 0.0060 | 2.12 |
| Claude Sonnet 4.5 | Negative k=5 | Myth -> game | 150 | 0.110 | 0.170 | 0.0077 | 1.81 |
| Claude Sonnet 4.5 | Negative k=5 informed | Game -> myth | 150 | 0.096 | 0.157 | 0.0085 | 1.67 |
| Claude Sonnet 4.5 | Negative k=5 informed | Myth -> game | 150 | 0.104 | 0.142 | 0.0065 | 1.78 |
| GPT-5-Nano | No noise | Game -> myth | 150 | 0.112 | 0.146 | 0.0041 | 3.72 |
| GPT-5-Nano | No noise | Myth -> game | 150 | 0.095 | 0.158 | 0.0050 | 4.66 |
| GPT-5-Nano | Neutral framing | Game -> myth | 180 | 0.120 | 0.150 | 0.0035 | 5.20 |
| GPT-5-Nano | Neutral framing | Myth -> game | 180 | 0.105 | 0.144 | 0.0045 | 5.09 |
| GPT-5-Nano | Positive k=5 | Game -> myth | 150 | 0.116 | 0.144 | 0.0047 | 3.81 |
| GPT-5-Nano | Positive k=5 | Myth -> game | 150 | 0.087 | 0.140 | 0.0049 | 4.06 |
| GPT-5-Nano | Positive k=5 informed | Game -> myth | 150 | 0.105 | 0.185 | 0.0084 | 3.86 |
| GPT-5-Nano | Positive k=5 informed | Myth -> game | 150 | 0.105 | 0.155 | 0.0059 | 3.80 |
| GPT-5-Nano | Negative k=5 | Game -> myth | 150 | 0.111 | 0.151 | 0.0046 | 3.84 |
| GPT-5-Nano | Negative k=5 | Myth -> game | 150 | 0.068 | 0.147 | 0.0064 | 5.44 |
| GPT-5-Nano | Negative k=5 informed | Game -> myth | 150 | 0.113 | 0.143 | 0.0034 | 3.65 |
| GPT-5-Nano | Negative k=5 informed | Myth -> game | 150 | 0.113 | 0.152 | 0.0047 | 4.20 |
| GPT-5-Nano | Bootstrap | Game -> myth | 150 | 0.116 | 0.169 | 0.0047 | 4.00 |
| GPT-5-Nano | Bootstrap | Myth -> game | 150 | 0.088 | 0.155 | 0.0069 | 4.81 |
| GPT-5-Nano | Bootstrap informed | Game -> myth | 150 | 0.118 | 0.161 | 0.0048 | 3.88 |
| GPT-5-Nano | Bootstrap informed | Myth -> game | 150 | 0.100 | 0.164 | 0.0062 | 3.40 |

