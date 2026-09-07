"""Spelling correction applied to recognized text before it is encoded.

An OCR engine reading a shop front at 30-50% CER returns strings a subword
tokenizer cannot handle. Correcting them first is one of the options the
project set out to compare, and it sits between recognition and encoding
rather than inside either.

The risk is specific and worth stating: a corrector pulls a word toward the
nearest dictionary entry, and the strings that identify a place are proper
nouns -- shop names, streets, brands -- which are exactly the ones absent from
a general dictionary. Correction can therefore destroy the signal it is meant
to clean. Building the dictionary from the dataset's own recognized text, as
``from_corpus`` does, keeps those names in it.

Both dictionaries have a defect, which is why this stage is off by default and
unused in the reported experiments. A language dictionary erases the proper
nouns that identify a place. A corpus dictionary assumes the corpus is right,
but the corpus is OCR output at 30-50% CER: when a sign is misread the same way
in most frames -- and OCR errors are systematic rather than random -- the
frequent form is the error, and correction then pulls correct readings toward
it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


def _deletes(word: str, distance: int) -> set[str]:
    """Every string reachable by deleting up to ``distance`` characters.

    This is the SymSpell idea: index deletions once, then a candidate and a
    dictionary entry match if their deletion sets intersect. It replaces the
    quadratic scan over the vocabulary that plain edit distance would need.
    """

    result = {word}
    frontier = {word}
    for _ in range(distance):
        nxt: set[str] = set()
        for value in frontier:
            for index in range(len(value)):
                nxt.add(value[:index] + value[index + 1:])
        result |= nxt
        frontier = nxt
    return result


def edit_distance(left: str, right: str) -> int:
    """Damerau-Levenshtein distance, restricted to adjacent transpositions.

    Transpositions matter here: OCR swaps neighbouring characters often enough
    that treating a swap as two edits would push real corrections out of range.
    """

    if left == right:
        return 0
    previous_previous: list[int] = []
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i] + [0] * len(right)
        for j, b in enumerate(right, start=1):
            cost = 0 if a == b else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if (
                i > 1
                and j > 1
                and a == right[j - 2]
                and left[i - 2] == b
            ):
                current[j] = min(current[j], previous_previous[j - 2] + cost)
        previous_previous, previous = previous, current
    return previous[len(right)]


@dataclass(slots=True)
class SymSpell:
    """Dictionary-based correction with a deletion index.

    ``min_length`` leaves very short strings alone: at two or three characters
    almost every dictionary word is within the edit radius, so correction there
    is close to guessing.
    """

    max_edit_distance: int = 2
    min_length: int = 4
    min_frequency: int = 1
    _frequency: Counter = field(default_factory=Counter, init=False, repr=False)
    _index: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "backend": "symspell",
            "max_edit_distance": self.max_edit_distance,
            "min_length": self.min_length,
            "min_frequency": self.min_frequency,
            "vocabulary": len(self._frequency),
            "checksum": _vocabulary_checksum(self._frequency),
        }

    def add(self, word: str, frequency: int = 1) -> None:
        word = word.strip().casefold()
        if not word:
            return
        self._frequency[word] += frequency
        for variant in _deletes(word, self.max_edit_distance):
            self._index.setdefault(variant, set()).add(word)

    def build(self, words: Iterable[tuple[str, int]]) -> SymSpell:
        for word, frequency in words:
            self.add(word, frequency)
        return self

    def correct(self, word: str) -> str:
        """Return the best dictionary match, or the word unchanged.

        Ties are broken by corpus frequency and then alphabetically, so the
        result never depends on set iteration order.
        """

        stripped = word.strip()
        if len(stripped) < self.min_length:
            return word
        lowered = stripped.casefold()
        if self._frequency.get(lowered, 0) >= self.min_frequency:
            return word

        candidates: set[str] = set()
        for variant in _deletes(lowered, self.max_edit_distance):
            candidates |= self._index.get(variant, set())
        if not candidates:
            return word

        best: tuple[int, int, str] | None = None
        for candidate in candidates:
            distance = edit_distance(lowered, candidate)
            if distance > self.max_edit_distance:
                continue
            key = (distance, -self._frequency[candidate], candidate)
            if best is None or key < best:
                best = key
        if best is None:
            return word
        return _match_case(stripped, best[2])

    def correct_all(self, texts: Sequence[str]) -> list[str]:
        return [self.correct(text) for text in texts]


def _match_case(original: str, corrected: str) -> str:
    """Restore the original capitalization pattern where it is unambiguous."""

    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected.capitalize()
    return corrected


def _vocabulary_checksum(frequency: Counter) -> str:
    import hashlib

    digest = hashlib.sha256()
    for word in sorted(frequency):
        digest.update(f"{word}:{frequency[word]}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


def from_corpus(
    texts: Iterable[str], *, min_count: int = 2, **params: Any
) -> SymSpell:
    """Build the dictionary from the dataset's own recognized text.

    A general language dictionary would not contain PECLARD or BOUCHERIE, and
    correcting them toward common words would remove exactly the strings that
    identify the place. Words the recognizer produced repeatedly are far more
    likely to be real than a one-off misreading, so ``min_count`` separates the
    vocabulary from the noise.
    """

    counts: Counter = Counter()
    for text in texts:
        value = text.strip().casefold()
        if value:
            counts[value] += 1
    vocabulary = [(word, count) for word, count in counts.items() if count >= min_count]
    return SymSpell(**params).build(vocabulary)


def build_symspell(
    texts: Iterable[str] | None = None,
    *,
    min_count: int = 2,
    max_edit_distance: int = 2,
    min_length: int = 4,
    **params: Any,
) -> SymSpell:
    """Registry factory for the ``symspell`` text corrector.

    ``texts`` is injected by preparation, which collects the dataset's own
    recognized strings first: a general dictionary would not hold the shop
    names that identify these places.
    """

    if params:
        raise ValueError(
            f"unknown symspell params: {', '.join(sorted(params))}; "
            "supported: min_count, max_edit_distance, min_length"
        )
    settings = {"max_edit_distance": max_edit_distance, "min_length": min_length}
    if texts is None:
        return SymSpell(**settings)
    return from_corpus(texts, min_count=min_count, **settings)
