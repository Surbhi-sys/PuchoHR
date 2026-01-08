# guardrail while user query
BLOCKED_KEYWORDS = [
    "rape", "sexual assault", "molestation",
    "murder", "kill", "crime", "drugs",
    "weapon", "terrorist", "bomb"
]

# guradrail during semantic search if user query is not related to that
HR_KEYWORDS = [
    "policy", "leave", "salary", "payroll",
    "attendance", "performance", "promotion",
    "competency", "probation", "termination",
    "disciplinary", "grievance", "appraisal"
]

#guardrail while giving response dont use this type of word
SENSITIVE_PHRASES = [
    "sue", "court", "lawyer",
    "legal action", "medical advice"
]


def violates_safety_policy(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in BLOCKED_KEYWORDS)


def is_hr_question(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in HR_KEYWORDS)

#Checks if the LLM-generated answer is actually based on the retrieved documents (prevents hallucination).

def is_answer_grounded(answer: str, docs, min_overlap=20) -> bool:
    context_text = " ".join(doc.page_content for doc in docs).lower()
    answer_words = answer.lower().split()

    overlap = sum(1 for word in answer_words if word in context_text)
    return overlap >= min_overlap


def contains_sensitive_advice(answer: str) -> bool:
    a = answer.lower()
    return any(phrase in a for phrase in SENSITIVE_PHRASES)


# for long user query count
MAX_WORDS = 300

def is_query_too_long(question: str) -> bool:
    """Return False if the query exceeds 300 words."""
    return len(question.strip().split()) > MAX_WORDS
