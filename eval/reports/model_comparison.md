# Model Comparison — AC-10

| Metric | openai/gpt-oss-120b | openai/gpt-oss-20b |
|---|---|---|
| Avg confidence | 0.656 | 0.676 |
| Abstention rate | 0.348 | 0.304 |
| Avg citations/answer | 0.91 | 0.87 |
| Avg latency (s/query) | 6.21 | 4.26 |

## Selection rationale

Fill this in after running against your live Gemini API results — in general, prefer the
lighter model (openai/gpt-oss-120b) for production if its confidence and citation coverage are
within a small margin of the larger model, since latency and cost scale with query volume in a
customer-facing assistant. Fall back to openai/gpt-oss-20b only if the lite model's abstention
rate is materially higher on eligibility/fee questions, since those are the highest-stakes
categories in this corpus.
