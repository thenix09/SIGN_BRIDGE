import React, { useState, useEffect, useRef, useCallback } from 'react';
import { DetectedHandData } from '../../types/landmarks';
import { apiClient } from '../../services/api.client';
import { RecognizeResponse } from '../../types/api';

/** Recognition state machine stages, mirroring backend RecognizeResponse.state */
type RecognitionState = 'IDLE' | 'DETECTING' | 'STABILISING' | 'CONFIRMED';

type SpellMode = 'mock' | 'live';

interface WordSuggestion {
  word: string;
}

interface SpellPanelProps {
  /** Current detected hands from the MediaPipe pipeline. */
  detectedHands: DetectedHandData[];
  /** Whether the camera feed is currently active. */
  cameraActive: boolean;
}

const MOCK_LETTERS = ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D'];
const DETECTING_DURATION_MS = 1500;
const STABILISING_DURATION_MS = 1000;

export const SpellPanel: React.FC<SpellPanelProps> = ({ detectedHands, cameraActive }) => {
  // Mode selection: Simulated Demo vs Live Backend Model
  const [spellMode, setSpellMode] = useState<SpellMode>('mock');

  // Recognition state
  const [currentLetter, setCurrentLetter] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [recognitionState, setRecognitionState] = useState<RecognitionState>('DETECTING');

  // Live recognition telemetry / error
  const [liveError, setLiveError] = useState<string | null>(null);
  const [isCallingLive, setIsCallingLive] = useState<boolean>(false);

  // Word accumulation
  const [wordBuffer, setWordBuffer] = useState<string[]>([]);
  const [sentence, setSentence] = useState<string>('');

  // Suggestions
  const [suggestions, setSuggestions] = useState<WordSuggestion[]>([]);

  // Mock cycle state
  const mockIndexRef = useRef(0);
  const cycleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isActive = cameraActive && detectedHands.length > 0;

  // Load vocabulary suggestions on mount
  useEffect(() => {
    apiClient
      .getVocabulary()
      .then((vocab) => {
        setSuggestions(vocab.supported_words.map((w) => ({ word: w })));
      })
      .catch(() => {});
  }, []);

  // -------------------------------------------------------------
  // MOCK SIMULATION CYCLE
  // -------------------------------------------------------------
  const clearCycleTimer = () => {
    if (cycleTimerRef.current) {
      clearTimeout(cycleTimerRef.current);
      cycleTimerRef.current = null;
    }
  };

  const confirmLetter = useCallback((letter: string) => {
    setWordBuffer((prev) => [...prev, letter]);
  }, []);

  const runMockCycle = useCallback(() => {
    if (!isActive || spellMode !== 'mock') return;

    const letter = MOCK_LETTERS[mockIndexRef.current % MOCK_LETTERS.length];
    setCurrentLetter(letter);
    setConfidence(0.6 + Math.random() * 0.3);
    setRecognitionState('DETECTING');

    cycleTimerRef.current = setTimeout(() => {
      if (!isActive || spellMode !== 'mock') return;
      setRecognitionState('STABILISING');
      setConfidence(0.8 + Math.random() * 0.18);

      cycleTimerRef.current = setTimeout(() => {
        if (!isActive || spellMode !== 'mock') return;
        setRecognitionState('CONFIRMED');
        setConfidence(0.92 + Math.random() * 0.07);
        confirmLetter(letter);
        mockIndexRef.current += 1;

        cycleTimerRef.current = setTimeout(() => {
          runMockCycle();
        }, 800);
      }, STABILISING_DURATION_MS);
    }, DETECTING_DURATION_MS);
  }, [isActive, spellMode, confirmLetter]);

  // Handle mock mode lifecycle
  useEffect(() => {
    clearCycleTimer();

    if (spellMode === 'mock') {
      if (isActive) {
        runMockCycle();
      } else {
        setCurrentLetter(null);
        setConfidence(0);
        setRecognitionState('DETECTING');
      }
    }

    return () => clearCycleTimer();
  }, [isActive, spellMode, runMockCycle]);

  // -------------------------------------------------------------
  // LIVE RECOGNITION (POST /api/recognize)
  // -------------------------------------------------------------
  const lastLiveCallRef = useRef<number>(0);

  const runLiveRecognition = useCallback(async () => {
    if (!isActive || spellMode !== 'live' || isCallingLive) return;

    // Rate limit to once every 250ms
    const now = Date.now();
    if (now - lastLiveCallRef.current < 250) return;
    lastLiveCallRef.current = now;

    const hand = detectedHands[0];
    const landmarks = hand.landmarks.map((lm) => [lm.x, lm.y, lm.z]);

    setIsCallingLive(true);

    try {
      const res: RecognizeResponse = await apiClient.recognizeLetter({
        session_id: 'spell-live-session',
        language: 'ASL',
        handedness: hand.handedness,
        landmarks
      });

      setLiveError(null);
      setCurrentLetter(res.label || null);
      setConfidence(res.confidence);
      setRecognitionState(res.state as RecognitionState);

      if (res.confirmed_letter) {
        setWordBuffer((prev) => [...prev, res.confirmed_letter!]);
      }
      if (res.sentence) {
        setSentence(res.sentence);
      }
      if (res.suggestions && res.suggestions.length > 0) {
        setSuggestions(res.suggestions.map((w) => ({ word: w })));
      }
    } catch (err: unknown) {
      const errMsg = (err as Error).message || 'Recognition error';
      setLiveError(errMsg);
      setRecognitionState('IDLE');
    } finally {
      setIsCallingLive(false);
    }
  }, [isActive, spellMode, isCallingLive, detectedHands]);

  // Trigger live recognition when hand landmarks change
  useEffect(() => {
    if (spellMode === 'live' && isActive) {
      runLiveRecognition();
    }
  }, [spellMode, isActive, detectedHands, runLiveRecognition]);

  // Controls
  const handleBackspace = () => {
    setWordBuffer((prev) => prev.slice(0, -1));
  };

  const handleCommitWord = () => {
    if (wordBuffer.length === 0) return;
    const word = wordBuffer.join('');
    setSentence((prev) => (prev ? prev + ' ' + word : word));
    setWordBuffer([]);
  };

  const handleClear = () => {
    setWordBuffer([]);
    setSentence('');
    mockIndexRef.current = 0;
  };

  const handleSuggestionClick = (word: string) => {
    setSentence((prev) => (prev ? prev + ' ' + word.toUpperCase() : word.toUpperCase()));
    setWordBuffer([]);
  };

  const confidencePct = Math.round(confidence * 100);

  return (
    <div className="spell-panel">
      {/* Panel heading */}
      <div className="card-title">
        <h3>Spell Mode</h3>
        <span className="brand-badge">Sign → Text</span>
      </div>

      {/* Mode Switcher */}
      <div className="spell-mode-toggle-row">
        <button
          className={`spell-mode-toggle-btn ${spellMode === 'mock' ? 'active' : ''}`}
          onClick={() => {
            setSpellMode('mock');
            setLiveError(null);
          }}
        >
          🎭 Simulated Demo (Mock)
        </button>
        <button
          className={`spell-mode-toggle-btn ${spellMode === 'live' ? 'active' : ''}`}
          onClick={() => {
            setSpellMode('live');
            setCurrentLetter(null);
            setConfidence(0);
          }}
        >
          ⚡ Live Backend Model
        </button>
      </div>

      {/* Mock data warning banner */}
      {spellMode === 'mock' && (
        <div className="spell-mock-banner" role="alert">
          ⚠ MOCK — Recognition not yet connected. Data is simulated.
        </div>
      )}

      {/* Live backend status / error banner */}
      {spellMode === 'live' && liveError && (
        <div className="spell-live-error-banner" role="alert">
          <strong>⚠ Live Recognition Blocker:</strong> {liveError}
          <div className="spell-live-error-sub">
            (Person B ASL model file <code>backend/models/asl_model.joblib</code> is pending training/export. Switch to Simulated Demo to test frontend interaction).
          </div>
        </div>
      )}

      {spellMode === 'live' && !liveError && (
        <div className="spell-live-active-banner">
          ✓ Connected to <code>POST /api/recognize</code> using real MediaPipe landmarks.
        </div>
      )}

      {/* Current letter + confidence */}
      <div className="spell-letter-display">
        <span className="spell-big-letter" aria-live="polite">
          {currentLetter ?? '—'}
        </span>

        <div className="spell-confidence">
          <div className="spell-confidence-track">
            <div
              className="spell-confidence-bar"
              style={{ width: isActive ? `${confidencePct}%` : '0%' }}
            />
          </div>
          <span className="spell-confidence-label">
            {isActive ? `${confidencePct}% confidence` : 'Waiting for camera & hand…'}
          </span>
        </div>
      </div>

      {/* Recognition state pill */}
      {isActive && (
        <div className="spell-state-row">
          <span className={`spell-state-pill spell-state-${recognitionState.toLowerCase()}`}>
            {recognitionState}
          </span>
        </div>
      )}

      {/* Word buffer — confirmed letters in current word */}
      <div className="spell-word-section">
        <label className="spell-section-label">Current word</label>
        <div className="spell-word-buffer" aria-live="polite" aria-label="Current word">
          {wordBuffer.length > 0 ? (
            wordBuffer.map((ch, i) => (
              <span key={i} className="spell-letter-chip">
                {ch}
              </span>
            ))
          ) : (
            <span className="spell-word-placeholder">—</span>
          )}
        </div>

        <div className="spell-word-controls">
          <button
            className="button spell-btn-sm"
            onClick={handleBackspace}
            disabled={wordBuffer.length === 0}
            aria-label="Delete last letter"
          >
            ⌫ Backspace
          </button>
          <button
            className="button spell-btn-sm"
            onClick={handleCommitWord}
            disabled={wordBuffer.length === 0}
            aria-label="Commit word to sentence"
          >
            ↵ Add Word
          </button>
          <button
            className="button spell-btn-sm spell-btn-clear"
            onClick={handleClear}
            aria-label="Clear everything"
          >
            ✕ Clear
          </button>
        </div>
      </div>

      {/* Sentence output */}
      <div className="spell-sentence-section">
        <label className="spell-section-label">Sentence</label>
        <div className="spell-sentence-output" aria-live="polite" aria-label="Sentence output">
          {sentence || <span className="spell-word-placeholder">Your sentence will appear here…</span>}
        </div>
      </div>

      {/* Word suggestions */}
      {suggestions.length > 0 && (
        <div className="spell-suggestions-section">
          <label className="spell-section-label">Suggestions</label>
          <div className="spell-suggestions">
            {suggestions.map((s) => (
              <button
                key={s.word}
                className="spell-suggestion-chip"
                onClick={() => handleSuggestionClick(s.word)}
                aria-label={`Use suggestion: ${s.word}`}
              >
                {s.word}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Idle hint */}
      {!cameraActive && <p className="collect-hint">Start the camera to begin recognition.</p>}
      {cameraActive && detectedHands.length === 0 && (
        <p className="collect-hint collect-hint-warn">Show your hand to start spelling.</p>
      )}
    </div>
  );
};
