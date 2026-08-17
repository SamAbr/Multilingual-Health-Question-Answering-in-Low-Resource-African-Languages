"""
Shared ROUGE scoring utilities.

rouge_score's default tokenizer replaces every character outside [a-z0-9] with a
space (see rouge_score.tokenize.NON_ALPHANUM_PATTERN) and then Porter-stems what's
left. That's tuned for English: it silently drops all Ge'ez-script text (Amharic
tokenizes to an empty token list) and shreds Akan's IPA-derived letters (e.g. "wɔn"
becomes ["w", "n"]), corrupting ROUGE scores for exactly the low-resource subsets
this project cares about most. WhitespaceTokenizer avoids both problems and is
safe across all 8 subsets' scripts.
"""

from rouge_score import rouge_scorer


class WhitespaceTokenizer:
    """Whitespace tokeniser — script-agnostic, safe for Ge'ez/Latin+IPA text."""

    def tokenize(self, text):
        if text is None:
            return []
        return str(text).strip().split()


def make_rouge_scorer(rouge_types=('rouge1', 'rougeL')):
    """Build a RougeScorer that tokenizes on whitespace instead of ASCII-only regex.

    use_stemmer is intentionally omitted: Porter stemming is English-specific and
    would mis-stem non-English tokens if applied.
    """
    return rouge_scorer.RougeScorer(list(rouge_types), tokenizer=WhitespaceTokenizer())
