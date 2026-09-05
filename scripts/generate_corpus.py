"""
Generates a synthetic retail-banking corpus (>=30 documents) with clause-level
IDs so the generation layer can cite document + section/clause per AC-02.
Re-runnable and idempotent: always overwrites the same file set deterministically.
No real customer, account, or card data is used anywhere in this corpus.
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")


def doc(doc_id, doc_type, title, clauses):
    """clauses: list of (clause_id, heading, body) tuples"""
    lines = [f"# {title}", f"doc_id: {doc_id}", f"doc_type: {doc_type}", ""]
    for cid, heading, body in clauses:
        lines.append(f"## [{cid}] {heading}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)


def build_account_products():
    docs = {}
    products = [
        ("Everyday Checking", "everyday-checking", 5.00, 500),
        ("Premium Savings", "premium-savings", 0.00, 2500),
        ("Student Checking", "student-checking", 0.00, 0),
        ("Senior Advantage Savings", "senior-advantage", 0.00, 1000),
        ("High-Yield Money Market", "high-yield-mm", 10.00, 10000),
        ("Basic Savings", "basic-savings", 3.00, 300),
        ("Business Essentials Checking", "business-essentials", 12.00, 1500),
        ("Youth Savings", "youth-savings", 0.00, 0),
    ]
    for i, (name, slug, fee, min_bal) in enumerate(products, start=1):
        doc_id = f"ACCT-{i:02d}"
        clauses = [
            (f"{doc_id}-1", "Product Overview",
             f"The {name} account is a retail deposit product offered for personal banking customers."),
            (f"{doc_id}-2", "Minimum Balance Requirement",
             f"A minimum daily balance of ${min_bal} must be maintained to avoid a maintenance fee."),
            (f"{doc_id}-3", "Monthly Maintenance Fee",
             f"A monthly maintenance fee of ${fee:.2f} applies if the minimum balance requirement is not met. "
             f"The fee is waived for customers enrolled in direct deposit of at least $500/month."),
            (f"{doc_id}-4", "Eligibility",
             f"Applicants must be at least 18 years of age (or a co-signer for {name} if under 18), "
             f"a U.S. resident, and provide valid government-issued identification."),
            (f"{doc_id}-5", "Interest / APY",
             f"Where applicable, interest accrues daily and is credited monthly. Rates are variable and "
             f"disclosed at account opening."),
        ]
        docs[f"{slug}-disclosure.md"] = doc(doc_id, "account_product", f"{name} — Product Disclosure", clauses)
    return docs


def build_card_products():
    docs = {}
    cards = [
        ("Everyday Rewards Visa", "rewards-visa", 0, 21.99, 35),
        ("Secured Starter Card", "secured-starter", 0, 24.99, 25),
        ("Platinum Cashback Mastercard", "platinum-cashback", 95, 19.99, 40),
        ("Student Rewards Card", "student-rewards", 0, 22.99, 30),
        ("Travel Elite Visa Signature", "travel-elite", 150, 18.99, 40),
        ("Business Cashback Card", "business-cashback", 0, 20.99, 35),
    ]
    for i, (name, slug, annual_fee, apr, late_fee) in enumerate(cards, start=1):
        doc_id = f"CARD-{i:02d}"
        clauses = [
            (f"{doc_id}-1", "Card Overview",
             f"The {name} is a consumer credit card product."),
            (f"{doc_id}-2", "Annual Fee",
             f"An annual fee of ${annual_fee} applies, billed on the account anniversary."),
            (f"{doc_id}-3", "Standard APR",
             f"The standard purchase APR is {apr}%, variable, based on the Prime Rate plus a margin "
             f"disclosed at account opening."),
            (f"{doc_id}-4", "Late Payment Fee",
             f"A late payment fee of up to ${late_fee} applies if the minimum payment is not received "
             f"by the due date."),
            (f"{doc_id}-5", "Eligibility",
             f"Applicants must be at least 18 years of age, have a valid Social Security Number or ITIN, "
             f"and meet minimum credit criteria at underwriting."),
            (f"{doc_id}-6", "Foreign Transaction Fee",
             f"A foreign transaction fee of 3% of the transaction amount applies to purchases made outside "
             f"the United States, except where the card explicitly waives this fee."),
        ]
        docs[f"{slug}-disclosure.md"] = doc(doc_id, "card_product", f"{name} — Product Disclosure", clauses)
    return docs


def build_loan_products():
    docs = {}
    loans = [
        ("Personal Loan", "personal-loan", "unsecured", 250, 8.99, 25.99),
        ("Auto Loan — New Vehicle", "auto-loan-new", "secured", 0, 5.49, 14.99),
        ("Auto Loan — Used Vehicle", "auto-loan-used", "secured", 0, 6.99, 16.99),
        ("Home Improvement Loan", "home-improvement-loan", "unsecured", 150, 9.49, 22.99),
        ("Debt Consolidation Loan", "debt-consolidation-loan", "unsecured", 200, 9.99, 24.99),
        ("Auto Refinance Loan", "auto-refinance-loan", "secured", 0, 5.99, 15.99),
    ]
    for i, (name, slug, kind, orig_fee, rate_low, rate_high) in enumerate(loans, start=1):
        doc_id = f"LOAN-{i:02d}"
        clauses = [
            (f"{doc_id}-1", "Product Overview",
             f"The {name} is a {kind} consumer lending product for personal use."),
            (f"{doc_id}-2", "Interest Rate Range",
             f"APR ranges from {rate_low}% to {rate_high}%, based on creditworthiness, loan term, and "
             f"(for auto loans) vehicle age."),
            (f"{doc_id}-3", "Origination Fee",
             f"An origination fee of ${orig_fee} applies at loan disbursement." if orig_fee
             else "No origination fee applies to this loan product."),
            (f"{doc_id}-4", "Foreclosure / Early Payoff Charge",
             f"Loans may be paid off early at any time with no prepayment penalty. A foreclosure "
             f"processing charge of $75 applies only if the loan is closed via foreclosure proceedings "
             f"following default."),
            (f"{doc_id}-5", "Eligibility",
             f"Applicants must be at least 18 years of age, a U.S. resident, and meet the bank's minimum "
             f"credit score and debt-to-income thresholds published in the Eligibility Policy Manual."),
            (f"{doc_id}-6", "Term Length",
             f"Terms are available from 24 to 72 months, subject to loan type and amount."),
        ]
        docs[f"{slug}-terms.md"] = doc(doc_id, "loan_product", f"{name} — Product Terms", clauses)
    return docs


def build_fee_schedule():
    docs = {}
    docs["fee-schedule-deposit-accounts.md"] = doc("FEE-01", "fee_schedule", "Fee Schedule — Deposit Accounts", [
        ("FEE-01-1", "Overdraft Fee", "An overdraft fee of $34 applies per item that overdraws the account, up to 3 items per day."),
        ("FEE-01-2", "Non-Sufficient Funds (NSF) Fee", "An NSF fee of $34 applies when a transaction is declined due to insufficient funds."),
        ("FEE-01-3", "Paper Statement Fee", "A fee of $3 per month applies for paper statements unless the customer is enrolled in e-statements."),
        ("FEE-01-4", "Stop Payment Fee", "A fee of $30 applies per stop-payment request on a check."),
    ])
    docs["fee-schedule-wires-and-transfers.md"] = doc("FEE-02", "fee_schedule", "Fee Schedule — Wires & Transfers", [
        ("FEE-02-1", "Wire Transfer — Domestic Outgoing", "A fee of $25 applies to domestic outgoing wire transfers."),
        ("FEE-02-2", "Wire Transfer — International Outgoing", "A fee of $45 applies to international outgoing wire transfers."),
        ("FEE-02-3", "ATM Fee — Out of Network", "A fee of $3.00 applies per withdrawal at an out-of-network ATM, in addition to any fee charged by the ATM owner."),
    ])
    docs["fee-schedule-cards.md"] = doc("FEE-03", "fee_schedule", "Fee Schedule — Cards", [
        ("FEE-03-1", "Card Replacement Fee", "A fee of $10 applies for expedited replacement of a lost or stolen debit or credit card; standard replacement is free."),
        ("FEE-03-2", "Cash Advance Fee", "A cash advance fee of 5% of the amount advanced (minimum $10) applies to credit card cash advances."),
    ])
    return docs


def build_eligibility_policy():
    docs = {}
    docs["eligibility-policy-general.md"] = doc("ELIG-01", "policy_manual", "Eligibility Policy Manual — General Requirements", [
        ("ELIG-01-1", "Minimum Age", "All personal banking products require the applicant to be at least 18 years old, except Student Checking, which permits a co-signed account for applicants aged 16-17."),
        ("ELIG-01-2", "Identification Requirements", "Applicants must provide one primary government-issued photo ID and one secondary form of identification (e.g., Social Security card, utility bill)."),
        ("ELIG-01-3", "Residency Requirement", "Applicants must be U.S. residents with a verifiable U.S. mailing address."),
    ])
    docs["eligibility-policy-credit.md"] = doc("ELIG-02", "policy_manual", "Eligibility Policy Manual — Credit Underwriting", [
        ("ELIG-02-1", "Credit Score Thresholds", "Unsecured credit products (credit cards, personal loans) generally require a minimum credit score of 640. Secured products (Secured Starter Card, auto loans) have no minimum score but require income verification."),
        ("ELIG-02-2", "Debt-to-Income Threshold", "Loan applications are generally declined if the applicant's debt-to-income ratio exceeds 45%, subject to underwriter discretion."),
        ("ELIG-02-3", "Adverse Action Notice", "Applicants who are declined will receive a written adverse action notice within 30 days, per Regulation B, stating the principal reasons for denial."),
    ])
    return docs


def build_faq():
    faqs = [
        ("How do I waive my checking account fee?", "Enroll in direct deposit of at least $500/month, or maintain the published minimum daily balance for your account type."),
        ("Can I have more than one savings account?", "Yes, customers may hold multiple deposit accounts subject to standard eligibility and identification checks on each account."),
        ("What happens if I miss a loan payment?", "A late fee may apply as disclosed in the relevant product terms, and missed payments may be reported to credit bureaus after 30 days past due."),
        ("Is there a penalty for paying off my loan early?", "No. Personal, auto, and home improvement loans described in this corpus carry no prepayment penalty."),
        ("How long does a credit card application take?", "Most decisions are returned instantly online; some applications require manual underwriting review, typically within 5-7 business days."),
        ("Can I apply for a joint account?", "Yes, joint accounts are available for checking and savings products; both applicants must independently meet identification requirements."),
        ("What is the difference between a secured and unsecured card?", "A secured card requires a refundable cash deposit that sets the credit limit; an unsecured card is extended based on creditworthiness alone."),
        ("How do I dispute a fee?", "Fee disputes can be submitted through the customer service channel within 60 days of the fee posting; each case is reviewed individually."),
    ]
    docs = {}
    for i, (q, a) in enumerate(faqs, start=1):
        doc_id = f"FAQ-{i:02d}"
        docs[f"customer-faq-{i:02d}.md"] = doc(doc_id, "faq", f"Customer FAQ — {q}", [
            (f"{doc_id}-1", q, a)
        ])
    return docs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_docs = {}
    for builder in (build_account_products, build_card_products, build_loan_products,
                    build_fee_schedule, build_eligibility_policy, build_faq):
        all_docs.update(builder())

    manifest = []
    for filename, content in all_docs.items():
        path = os.path.join(OUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        manifest.append(filename)

    with open(os.path.join(OUT_DIR, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"document_count": len(manifest), "files": sorted(manifest)}, f, indent=2)

    print(f"Generated {len(manifest)} documents into {OUT_DIR}")


if __name__ == "__main__":
    main()
