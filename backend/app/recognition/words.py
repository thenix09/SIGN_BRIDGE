"""Word builder and autocomplete suggestion engine for fingerspelling Spell mode."""

from collections.abc import Sequence
from pathlib import Path


class _TrieNode:
    """Node in autocomplete prefix Trie."""

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.is_word: bool = False
        self.word: str | None = None


class SuggestionEngine:
    """Trie-based autocomplete suggestion engine for English dictionary words."""

    def __init__(self, words: Sequence[str] | Path | str | None = None) -> None:
        self.root = _TrieNode()
        self._total_words = 0

        if isinstance(words, (Path, str)):
            path = Path(words)
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    word_list = [line.strip() for line in f if line.strip()]
                self._load_words(word_list)
        elif words is not None:
            self._load_words(words)

    def _load_words(self, words: Sequence[str]) -> None:
        """Insert sequence of words into Trie."""
        for w in words:
            word = w.strip().upper()
            if word and word.isalpha():
                self.insert(word)

    def insert(self, word: str) -> None:
        """Insert a word into the Trie."""
        node = self.root
        for char in word.upper():
            if char not in node.children:
                node.children[char] = _TrieNode()
            node = node.children[char]
        if not node.is_word:
            node.is_word = True
            node.word = word.upper()
            self._total_words += 1

    def suggest(self, prefix: str, limit: int = 3) -> list[str]:
        """Find up to `limit` words starting with the given prefix.

        Args:
            prefix: Starting letter sequence.
            limit: Maximum count of suggestions to return (default 3).

        Returns:
            List of uppercase matching words.
        """
        prefix_clean = prefix.strip().upper()
        if not prefix_clean:
            return []

        node = self.root
        for char in prefix_clean:
            if char not in node.children:
                return []
            node = node.children[char]

        results: list[str] = []

        def dfs(current: _TrieNode) -> None:
            if len(results) >= limit:
                return
            if current.is_word and current.word:
                results.append(current.word)
            for char in sorted(current.children.keys()):
                dfs(current.children[char])

        dfs(node)
        return results


class WordBuilder:
    """Stateful word accumulator and sentence compositor for fingerspelling."""

    def __init__(self) -> None:
        self._current_word: list[str] = []
        self._committed_words: list[str] = []

    @property
    def current_word(self) -> str:
        """Return currently uncommitted word being spelled."""
        return "".join(self._current_word)

    @property
    def sentence(self) -> str:
        """Return full sentence of committed words plus any active word."""
        parts = list(self._committed_words)
        if self._current_word:
            parts.append("".join(self._current_word))
        return " ".join(parts)

    def add_letter(self, letter: str) -> str:
        """Add a single letter to the current word.

        Args:
            letter: Single alphabetic character.

        Returns:
            Updated current word string.
        """
        clean_char = letter.strip().upper()
        if clean_char and len(clean_char) == 1 and clean_char.isalpha():
            self._current_word.append(clean_char)
        return self.current_word

    def backspace(self) -> str:
        """Remove the last letter from current word or last word from sentence.

        Returns:
            Updated current word.
        """
        if self._current_word:
            self._current_word.pop()
        elif self._committed_words:
            # Reopen the last committed word for editing
            last_word = self._committed_words.pop()
            self._current_word = list(last_word)
        return self.current_word

    def commit_word(self, word: str | None = None) -> str:
        """Commit the current word (or explicit override word) to the sentence.

        Args:
            word: Optional replacement word (e.g. from suggestion autocomplete).

        Returns:
            Committed word string.
        """
        committed = (word or self.current_word).strip().upper()
        if committed:
            self._committed_words.append(committed)
        self._current_word = []
        return committed

    def clear(self) -> None:
        """Clear both active word and committed sentence."""
        self._current_word.clear()
        self._committed_words.clear()
