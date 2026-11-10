"""
Removes personal details from a candidate's text BEFORE any AI model sees it,
so names, age, family situation, gender, origin and contact details cannot
influence extraction, embeddings or ranking.

Rules are simple patterns, so they are fast, repeatable and testable.
They catch the common ways people write these details; the extraction prompt
also tells the model never to copy such details (a second line of defence).
In production, Azure AI Language PII detection could replace or extend this.
"""

import re

_NAME = r"[A-ZÅÄÖÉ][\w'’-]+(?:\s+[A-ZÅÄÖÉ][\w'’-]+){0,2}"

# (pattern, replacement); applied in this order
PATTERNS = [
    # Contact details and ID numbers
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[contact]"),
    (re.compile(r"(?<!\w)(?:\+46|0)\s?7\d(?:[\s-]?\d){7}(?!\w)"), "[contact]"),
    (re.compile(r"\b(?:19|20)?\d{6}[-+]\d{4}\b"), "[id]"),
    # Names after an introduction
    (re.compile(r"(?i:\b(my name is|my name's|i am called|i'm called|call me|jag heter)\s+)" + _NAME), r"\1 [name]"),
    (re.compile(r"(?i:\b(hi|hello|hej),?\s+(i'm|i am|jag är)\s+)" + _NAME), r"\1 \2 [name]"),
    # Age
    (re.compile(r"(?i)\b(i am|i'm|jag är)\s+\d{1,2}(?=\s*(?:years?\s+old|år\b|,|\.|;|$|\s+and\b|\s+och\b))"), r"\1 [age]"),
    (re.compile(r"(?i)\b\d{1,2}[\s-]years?[\s-]old\b"), "[age]"),
    (re.compile(r"(?i)\baged\s+\d{1,2}\b"), "[age]"),
    # Family situation
    (re.compile(r"(?i)\b(?:a\s+)?(?:single\s+)?(?:mother|father|mum|mom|dad|parent)\s+of\s+(?:\w+|\d+)(?:\s+(?:children|kids|sons|daughters))?"), "[family details]"),
    (re.compile(r"(?i)\bi have\s+(?:\w+|\d+)\s+(?:children|kids|sons|daughters)\b"), "[family details]"),
    (re.compile(r"(?i)\bmy\s+(?:husband|wife|partner|children|kids|son|daughter)\b"), "[family member]"),
    # Gender and origin stated about oneself
    (re.compile(r"(?i)\b(i am|i'm|as)\s+a\s+(?:woman|man|girl|boy|mother|father|female|male)\b"), r"\1 [gender]"),
    (re.compile(r"(?i)\bi\s+(?:am|come|came|moved)\s+(?:originally\s+)?from\s+" + r"[A-ZÅÄÖ][\w-]+"), "[origin]"),
]

# Words that carry no job information once the details are removed
_FILLER = {"my", "name", "is", "name's", "i", "am", "i'm", "and", "a", "an", "the", "years",
           "old", "of", "called", "call", "me", "hi", "hello", "hej", "jag", "heter", "är",
           "år", "gammal", "och", "with", "originally", "from", "aged"}
_PLACEHOLDER = re.compile(r"\[(?:name|age|family details|family member|gender|origin|contact|id)\]")


def anonymise(text):
    """Returns (clean_text, removed): removed lists the kinds of details found.
    A sentence left with only placeholders and filler words is dropped."""
    removed = []
    for pattern, replacement in PATTERNS:
        text, count = pattern.subn(replacement, text)
        if count:
            removed.extend(_PLACEHOLDER.findall(replacement))

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []
    for sentence in sentences:
        words = re.findall(r"[\w'’]+", _PLACEHOLDER.sub(" ", sentence).lower())
        if _PLACEHOLDER.search(sentence) and len([w for w in words if w not in _FILLER]) < 2:
            continue  # nothing about the job left in this sentence
        kept.append(sentence)
    clean = re.sub(r"\s+", " ", " ".join(kept)).strip()
    return clean, sorted(set(removed))
