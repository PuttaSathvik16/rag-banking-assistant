# Sample Outputs — Demonstrating AC-02, AC-05, AC-06

This document shows real JSON responses from the RAG pipeline, demonstrating:
- **AC-02**: Clause-level citations in grounded answers
- **AC-05**: Abstention when corpus doesn't support a confident answer
- **AC-06**: Validated Pydantic schema with answer, citations, product, fee, eligibility, and confidence fields

## Example 1: Successful Lookup with Citations (AC-02, AC-06)

**Query:** `"What is the overdraft fee and how many times per day can it apply?"`

**Expected Response Type:** Direct factual lookup with clause-level citation.

```json
{
  "answer": "The overdraft fee is $34 per item, and it can apply up to 3 times per day.",
  "citations": [
    {
      "doc_id": "FEE-DEPOSIT-01",
      "clause_id": "FEE-DEPOSIT-01-2",
      "source_file": "fee-schedule-deposit-accounts.md"
    }
  ],
  "applicable_product": "Deposit Accounts",
  "relevant_fee_or_charge": "Overdraft fee: $34 per item, max 3 per day",
  "eligibility_criteria": null,
  "confidence": 0.95,
  "abstained": false,
  "abstention_reason": null
}
```

**What this demonstrates:**
- ✅ **Clause-level citation**: Response includes doc_id, clause_id, and source_file pointing to the exact section in fee-schedule-deposit-accounts.md
- ✅ **Confidence score**: 0.95 reflects high confidence (the question directly asks for a fee from the corpus)
- ✅ **Applicable product**: Populated with the relevant account type
- ✅ **Relevant fee/charge**: Extracted and summarized
- ✅ **Abstained = false**: System is confident and providing an answer

---

## Example 2: Multi-Part Query Requiring Decomposition (AC-07, AC-02, AC-06)

**Query:** `"Compare the annual fees and APRs of the Rewards Visa and the Travel Elite card"`

**Expected Response Type:** Multi-part answer synthesizing two different documents (AC-07 query transformation).

```json
{
  "answer": "Rewards Visa: $0 annual fee, 21.99% APR. Travel Elite: $150 annual fee, 18.99% APR.",
  "citations": [
    {
      "doc_id": "CARD-REWARDS-01",
      "clause_id": "CARD-REWARDS-01-1",
      "source_file": "rewards-visa-disclosure.md"
    },
    {
      "doc_id": "CARD-REWARDS-01",
      "clause_id": "CARD-REWARDS-01-2",
      "source_file": "rewards-visa-disclosure.md"
    },
    {
      "doc_id": "CARD-ELITE-01",
      "clause_id": "CARD-ELITE-01-1",
      "source_file": "travel-elite-disclosure.md"
    },
    {
      "doc_id": "CARD-ELITE-01",
      "clause_id": "CARD-ELITE-01-2",
      "source_file": "travel-elite-disclosure.md"
    }
  ],
  "applicable_product": "Credit Cards (Rewards Visa, Travel Elite)",
  "relevant_fee_or_charge": "Annual Fees: Rewards Visa $0, Travel Elite $150",
  "eligibility_criteria": null,
  "confidence": 0.88,
  "abstained": false,
  "abstention_reason": null
}
```

**What this demonstrates:**
- ✅ **Multiple citations across sources**: Four clause-level citations from two different documents (AC-02, AC-03 retrieval fusion)
- ✅ **Multi-part synthesis**: Answer correctly compares both products
- ✅ **Structured extraction**: applicable_product and relevant_fee_or_charge both populated with card-specific data
- ✅ **AC-07 in action**: Query transformation handled the two-card comparison before retrieval

---

## Example 3: Correct Abstention — Rate-by-Tier Question (AC-05)

**Query:** `"What is the interest rate on a personal loan for someone with excellent credit?"`

**Expected Response Type:** Abstention — the corpus provides a range (8.99%-25.99%) but does NOT map rates to credit tiers.

```json
{
  "answer": "I cannot confirm the interest rate for your specific credit profile from the available documentation.",
  "citations": [],
  "applicable_product": "Personal Loan",
  "relevant_fee_or_charge": null,
  "eligibility_criteria": null,
  "confidence": 0.0,
  "abstained": true,
  "abstention_reason": "The corpus provides a rate range (8.99%-25.99%) but does not specify rate-by-credit-tier mapping; cannot give a definitive rate for 'excellent credit' without that breakdown."
}
```

**What this demonstrates:**
- ✅ **Correct abstention (AC-05)**: System refuses to guess or extrapolate credit-tier-specific rates
- ✅ **Grounding rule honored**: No definitive financial advice ("you will get X% rate")
- ✅ **Confidence threshold enforced**: confidence = 0.0, below the 0.35 threshold in config.yaml
- ✅ **Abstention reason**: Explains why the system abstained (corpus limitation, not system error)
- ✅ **Safe degradation (NFR-05)**: User gets a helpful explanation rather than a fabricated answer

---

## Implementation Notes

All three examples follow the [AnswerResponse Pydantic schema](../src/schemas.py):
- `answer`: Natural-language summary grounded in corpus
- `citations`: List of `Citation` objects with doc_id, clause_id, source_file
- `applicable_product`, `relevant_fee_or_charge`, `eligibility_criteria`: Structured extraction fields
- `confidence`: 0.0–1.0 scalar, enforced at code level in [src/generation.py](../src/generation.py#L108)
- `abstained`: Boolean flag; if true, `abstention_reason` explains why
- `abstention_reason`: Only populated if abstained=true or on error

These samples demonstrate that the pipeline correctly:
1. **Retrieves** relevant documents via BM25 + semantic search fusion (AC-03)
2. **Reranks** candidates before generation (AC-04)
3. **Generates** grounded answers with citations (AC-02)
4. **Transforms** multi-part queries (AC-07)
5. **Abstains** when grounding is weak (AC-05)
6. **Validates** structured output against Pydantic schema (AC-06)
