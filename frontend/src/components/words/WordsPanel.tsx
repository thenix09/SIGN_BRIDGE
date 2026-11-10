import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { DetectedHandData, DetectedPoseData, UPPER_BODY_POSE_INDICES } from '../../types/landmarks';
import { apiClient } from '../../services/api.client';
import {
  VocabularyWord,
  BackendVocabularyResponse,
  RecognizeWordsResponse
} from '../../types/api';

interface WordsPanelProps {
  detectedHands: DetectedHandData[];
  detectedPose: DetectedPoseData | null;
  cameraActive: boolean;
}

export const WordsPanel: React.FC<WordsPanelProps> = ({
  detectedHands,
  detectedPose,
  cameraActive
}) => {
  // Vocabulary state
  const [vocabStatus, setVocabStatus] = useState<string>('loading');
  const [words, setWords] = useState<VocabularyWord[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedWord, setSelectedWord] = useState<VocabularyWord | null>(null);

  // Words I Know (User Knowledge Preferences) state
  const KNOWN_WORDS_STORAGE_KEY = 'signbridge_known_words';
  const [knownWords, setKnownWords] = useState<Set<string>>(() => {
    try {
      const saved = localStorage.getItem('signbridge_known_words');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) return new Set(parsed);
      }
    } catch {
      // fallback empty
    }
    return new Set<string>();
  });
  const [knowledgeFilter, setKnowledgeFilter] = useState<'all' | 'known' | 'learning'>('all');

  // Toggle word in "Words I Know" list (persisted locally)
  const toggleKnownWord = (wordLabel: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const key = wordLabel.toLowerCase();
    setKnownWords((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      try {
        localStorage.setItem(KNOWN_WORDS_STORAGE_KEY, JSON.stringify(Array.from(next)));
      } catch {
        // ignore storage errors
      }
      return next;
    });
  };

  // Recognition state
  const [isRecognizing, setIsRecognizing] = useState<boolean>(false);
  const [recognitionResult, setRecognitionResult] = useState<RecognizeWordsResponse | null>(null);
  const [recognitionError, setRecognitionError] = useState<string | null>(null);

  // Load vocabulary on mount
  useEffect(() => {
    let isCancelled = false;
    setVocabStatus('loading');

    apiClient
      .getWordVocabulary('ASL')
      .then((res: BackendVocabularyResponse) => {
        if (isCancelled) return;
        setWords(res.words);
        setVocabStatus(res.status);
        if (res.words.length > 0) {
          setSelectedWord(res.words[0]);
        }
      })
      .catch((err) => {
        if (isCancelled) return;
        setVocabStatus('error');
        setRecognitionError(`Failed to load vocabulary: ${err.message}`);
      });

    return () => {
      isCancelled = true;
    };
  }, []);

  // Compute unique categories
  const categories = useMemo(() => {
    const set = new Set<string>();
    words.forEach((w) => set.add(w.category));
    return ['all', ...Array.from(set)];
  }, [words]);

  // Filter words by category, search term, and knowledge preference
  const filteredWords = useMemo(() => {
    return words.filter((w) => {
      const wKey = w.label.toLowerCase();
      const isKnown = knownWords.has(wKey);
      if (knowledgeFilter === 'known' && !isKnown) return false;
      if (knowledgeFilter === 'learning' && isKnown) return false;
      const matchesCat = selectedCategory === 'all' || w.category.toLowerCase() === selectedCategory.toLowerCase();
      const matchesSearch = wKey.includes(searchQuery.toLowerCase().trim());
      return matchesCat && matchesSearch;
    });
  }, [words, selectedCategory, searchQuery, knowledgeFilter, knownWords]);

  const knownCount = useMemo(() => {
    return words.filter((w) => knownWords.has(w.label.toLowerCase())).length;
  }, [words, knownWords]);

  const masteryPct = words.length > 0 ? Math.round((knownCount / words.length) * 100) : 0;

  // Trigger single-frame recognition using real detected pose and hand landmarks
  const handleRecognizeWord = useCallback(async () => {
    if (!cameraActive) {
      setRecognitionError('Start the camera before evaluating words.');
      return;
    }

    if (!detectedPose && detectedHands.length === 0) {
      setRecognitionError('No pose or hand detected in the camera frame.');
      return;
    }

    setIsRecognizing(true);
    setRecognitionError(null);

    try {
      // Build 9-pose coordinate array required by sequence.py
      let posePoints: number[][] | null = null;
      if (detectedPose && detectedPose.landmarks.length >= 17) {
        posePoints = UPPER_BODY_POSE_INDICES.map((idx) => {
          const pt = detectedPose.landmarks[idx];
          return [pt.x, pt.y];
        });
      }

      // Format up to 2 hands (each 21 points)
      const handArrays: number[][][] = detectedHands.slice(0, 2).map((h) =>
        h.landmarks.map((lm) => [lm.x, lm.y])
      );

      const response = await apiClient.recognizeWord({
        session_id: 'words-session-frontend',
        language: 'ASL',
        frame: {
          pose: posePoints,
          hands: handArrays
        }
      });

      setRecognitionResult(response);
    } catch (err: unknown) {
      const errorMsg = (err as Error).message || 'Word recognition failed';
      setRecognitionError(errorMsg);
      setRecognitionResult(null);
    } finally {
      setIsRecognizing(false);
    }
  }, [cameraActive, detectedPose, detectedHands]);

  return (
    <div className="words-panel">
      {/* Panel header */}
      <div className="card-title">
        <h3>Word Signs (Words)</h3>
        <span className="brand-badge">Isolated Signs (30 Words)</span>
      </div>

      {/* Model availability banner */}
      {vocabStatus === 'withheld' && (
        <div className="words-withheld-banner">
          ℹ Word model status: <strong>Withheld</strong> (Offline vocabulary browsing enabled. Real-time sequence model integration in progress by Person B).
        </div>
      )}

      {/* Target Word Practice Card */}
      {selectedWord && (
        <div className="words-target-card">
          <div className="words-target-header">
            <span className="words-target-prefix">Target Sign:</span>
            <span className="words-target-name">{selectedWord.label.toUpperCase()}</span>
            <span className="words-category-badge">{selectedWord.category}</span>
          </div>

          <div className="words-target-action-row">
            <button
              className="button words-eval-btn"
              onClick={handleRecognizeWord}
              disabled={isRecognizing || !cameraActive}
            >
              {isRecognizing ? '⏳ Evaluating Sign…' : '🔍 Evaluate Current Frame'}
            </button>
            <span className="words-tracking-indicator">
              {detectedPose ? '✓ Pose tracked' : '○ No pose'}{' • '}
              {detectedHands.length > 0 ? `✓ ${detectedHands.length} hand(s)` : '○ No hands'}
            </span>
          </div>
        </div>
      )}

      {/* Recognition Results / Prediction Output */}
      {recognitionError && (
        <div className="words-error-banner" role="alert">
          ⚠ {recognitionError}
        </div>
      )}

      {recognitionResult && (
        <div className="words-results-container">
          <div className="words-result-row">
            <span className="words-state-pill">
              State: {recognitionResult.state}
            </span>
            {recognitionResult.accepted_word && (
              <span className="words-accepted-badge">
                Accepted: {recognitionResult.accepted_word.toUpperCase()}
              </span>
            )}
          </div>

          {recognitionResult.sentence && (
            <div className="words-sentence-box">
              <label>Sentence:</label>
              <p>{recognitionResult.sentence}</p>
            </div>
          )}

          {recognitionResult.top_k.length > 0 && (
            <div className="words-candidates-list">
              <label>Predictions (Top Candidates):</label>
              {recognitionResult.top_k.map((cand) => {
                const pct = Math.round(cand.confidence * 100);
                return (
                  <div key={cand.label} className="words-candidate-item">
                    <span className="words-cand-label">{cand.label}</span>
                    <div className="words-cand-track">
                      <div className="words-cand-bar" style={{ width: `${pct}%` }} />
                    </div>
                    <span className="words-cand-pct">{pct}%</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Words I Know - Mastery Progress & Filter */}
      <div className="words-knowledge-box">
        <div className="words-knowledge-header">
          <span className="words-knowledge-title">
            ⭐ <strong>Words I Know:</strong> {knownCount} / {words.length} ({masteryPct}%)
          </span>
          <div className="words-knowledge-pills" role="tablist" aria-label="Knowledge Filter">
            <button
              className={`words-knowledge-pill ${knowledgeFilter === 'all' ? 'active' : ''}`}
              onClick={() => setKnowledgeFilter('all')}
            >
              All ({words.length})
            </button>
            <button
              className={`words-knowledge-pill ${knowledgeFilter === 'known' ? 'active' : ''}`}
              onClick={() => setKnowledgeFilter('known')}
            >
              ✓ Known ({knownCount})
            </button>
            <button
              className={`words-knowledge-pill ${knowledgeFilter === 'learning' ? 'active' : ''}`}
              onClick={() => setKnowledgeFilter('learning')}
            >
              To Learn ({words.length - knownCount})
            </button>
          </div>
        </div>
        <div className="words-mastery-track">
          <div className="words-mastery-bar" style={{ width: `${masteryPct}%` }} />
        </div>
      </div>

      {/* Vocabulary Search & Category Filters */}
      <div className="words-filter-section">
        <input
          type="text"
          className="words-search-input"
          placeholder="Search vocabulary words…"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          aria-label="Search vocabulary"
        />

        <div className="words-category-pills" role="tablist" aria-label="Vocabulary Categories">
          {categories.map((cat) => (
            <button
              key={cat}
              className={`words-cat-pill ${selectedCategory === cat ? 'active' : ''}`}
              onClick={() => setSelectedCategory(cat)}
            >
              {cat.charAt(0).toUpperCase() + cat.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Vocabulary Word Grid */}
      <div className="words-grid-wrapper">
        <label className="words-grid-label">
          Available Vocabulary ({filteredWords.length} words):
        </label>

        {vocabStatus === 'loading' && (
          <div className="words-loading-box">
            <div className="spinner" />
            <p>Loading sign vocabulary…</p>
          </div>
        )}

        {vocabStatus !== 'loading' && filteredWords.length === 0 && (
          <div className="words-empty-box">
            <p>No vocabulary words match "{searchQuery}"</p>
          </div>
        )}

        {vocabStatus !== 'loading' && filteredWords.length > 0 && (
          <div className="words-grid">
            {filteredWords.map((word) => {
              const isSelected = selectedWord?.label === word.label;
              const isKnown = knownWords.has(word.label.toLowerCase());
              return (
                <div
                  key={word.label}
                  className={`words-card-item ${isSelected ? 'selected' : ''} ${isKnown ? 'is-known' : ''}`}
                  onClick={() => {
                    setSelectedWord(word);
                    setRecognitionResult(null);
                    setRecognitionError(null);
                  }}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      setSelectedWord(word);
                    }
                  }}
                  aria-pressed={isSelected}
                >
                  <div className="words-card-top-row">
                    <span className="words-card-label">{word.label}</span>
                    <button
                      type="button"
                      className={`words-known-toggle ${isKnown ? 'active' : ''}`}
                      onClick={(e) => toggleKnownWord(word.label, e)}
                      title={isKnown ? 'Click to mark as learning' : 'Click to mark as known'}
                      aria-label={`Mark ${word.label} as known`}
                    >
                      {isKnown ? '✓ Known' : '+ Learn'}
                    </button>
                  </div>
                  <span className="words-card-cat">{word.category}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
