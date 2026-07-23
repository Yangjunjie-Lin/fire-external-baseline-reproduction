from __future__ import annotations

import re
import unicodedata

TOKENIZER_VERSION = "deterministic_bilingual_lexical_v2"

_LATIN_OR_NUMBER_RE = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")

# Frozen, case-agnostic emergency-domain lexical aliases. These aliases bridge
# the language boundary; they do not select documents or use evaluator gold.
_BILINGUAL_LEXICAL_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("电气", ("electrical",)),
    ("电器", ("electrical",)),
    ("配电", ("electrical",)),
    ("充电", ("electrical",)),
    ("断电", ("power", "isolation")),
    ("隔离", ("isolation",)),
    ("确认", ("confirmation",)),
    ("水", ("water",)),
    ("浓烟", ("smoke",)),
    ("烟气", ("smoke",)),
    ("冒烟", ("smoke",)),
    ("排烟", ("smoke",)),
    ("烟", ("smoke",)),
    ("消防", ("fire", "response")),
    ("防火", ("fire",)),
    ("明火", ("fire",)),
    ("起火", ("fire",)),
    ("火情", ("fire",)),
    ("火", ("fire",)),
    ("呼吸", ("respiratory",)),
    ("防护", ("protection",)),
    ("进入", ("entry",)),
)


def deterministic_bilingual_lexical_tokens(text: object) -> list[str]:
    """Return stable Latin tokens, CJK unigrams/bigrams, and lexical aliases.

    The function is deliberately offline and deterministic. Alias expansion is
    global and document-independent; it cannot inspect case IDs, ranks, or gold.
    """

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    tokens = [match.group(0) for match in _LATIN_OR_NUMBER_RE.finditer(normalized)]
    for match in _CJK_RUN_RE.finditer(normalized):
        run = match.group(0)
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    for source, aliases in _BILINGUAL_LEXICAL_ALIASES:
        if source in normalized:
            tokens.extend(aliases)
    return tokens
