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
const DETECTING_DURATION_MS = 600;
const STABILISING_DURATION_MS = 400;

export const SpellPanel: React.FC<SpellPanelProps> = ({ detectedHands, cameraActive }) => {
  // Mode selection: Live Backend Model by default
  const [spellMode, setSpellMode] = useState<SpellMode>('live');

  // Responsiveness settings: 60ms = snappy real-time, 100ms = balanced, 200ms = relaxed
  const [responseSpeed, setResponseSpeed] = useState<number>(60);
  const [requiredStreak, setRequiredStreak] = useState<number>(4);

  // Recognition state
  const [currentLetter, setCurrentLetter] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [recognitionState, setRecognitionState] = useState<RecognitionState>('DETECTING');
  const [streakCount, setStreakCount] = useState<number>(0);

  // Live recognition telemetry / error
  const [liveError, setLiveError] = useState<string | null>(null);
  const [isCallingLive, setIsCallingLive] = useState<boolean>(false);

  // Word accumulation
  const [wordBuffer, setWordBuffer] = useState<string[]>([]);
  const [sentence, setSentence] = useState<string>('');

  // Suggestions
  const [suggestions, setSuggestions] = useState<WordSuggestion[]>([]);

  // Streak tracking ref across frames
  const streakRef = useRef<{ letter: string; count: number; confirmed: boolean }>({
    letter: '',
    count: 0,
    confirmed: false,
  });

  // Mock cycle state
  const mockIndexRef = useRef(0);
  const cycleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isActive = cameraActive && detectedHands.length > 0;

  // Load vocabulary suggestions on mount
  useEffect(() => {
    apiClient
      .getVocabulary()
      .then((vocab) => {
        if (vocab && vocab.supported_words) {
          setSuggestions(vocab.supported_words.map((w) => ({ word: w })));
        }
      })
      .catch(() => {});
  }, []);

  const confirmLetter = useCallback((letter: string) => {
    if (!letter) return;
    setWordBuffer((prev) => [...prev, letter]);
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

  const runMockCycle = useCallback(() => {
    if (!isActive || spellMode !== 'mock') return;

    const letter = MOCK_LETTERS[mockIndexRef.current % MOCK_LETTERS.length];
    setCurrentLetter(letter);
    setConfidence(0.6 + Math.random() * 0.3);
    setRecognitionState('DETECTING');
    setStreakCount(1);

    cycleTimerRef.current = setTimeout(() => {
      if (!isActive || spellMode !== 'mock') return;
      setRecognitionState('STABILISING');
      setConfidence(0.85 + Math.random() * 0.12);
      setStreakCount(3);

      cycleTimerRef.current = setTimeout(() => {
        if (!isActive || spellMode !== 'mock') return;
        setRecognitionState('CONFIRMED');
        setConfidence(0.95 + Math.random() * 0.04);
        setStreakCount(requiredStreak);
        confirmLetter(letter);
        mockIndexRef.current += 1;

        cycleTimerRef.current = setTimeout(() => {
          runMockCycle();
        }, 500);
      }, STABILISING_DURATION_MS);
    }, DETECTING_DURATION_MS);
  }, [isActive, spellMode, confirmLetter, requiredStreak]);

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
        setStreakCount(0);
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

    // Responsive user-configurable throttle (default 60ms = ~16 fps)
    const now = Date.now();
    if (now - lastLiveCallRef.current < responseSpeed) return;
    lastLiveCallRef.current = now;

    const hand = detectedHands[0];
    const landmarks = hand.landmarks.map((lm) => [lm.x, lm.y, lm.z]);

    setIsCallingLive(true);

    try {
      const res: RecognizeResponse = await apiClient.recognizeLetter({
        session_id: 'spell-live-session',
        language: 'ASL',
        handedness: hand.handedness,
        landmarks,
      });

      setLiveError(null);
      const recognized = res.label || null;
      setCurrentLetter(recognized);
      setConfidence(res.confidence);

      // Fast-streak tracking to confirm without holding hands too long
      if (recognized && res.confidence >= 0.55) {
        if (streakRef.current.letter === recognized) {
          streakRef.current.count += 1;
        } else {
          streakRef.current = { letter: recognized, count: 1, confirmed: false };
        }

        const count = streakRef.current.count;
        setStreakCount(count);

        if (count >= requiredStreak && !streakRef.current.confirmed) {
          streakRef.current.confirmed = true;
          setRecognitionState('CONFIRMED');
          confirmLetter(recognized);
        } else if (count >= 2) {
          setRecognitionState('STABILISING');
        } else {
          setRecognitionState('DETECTING');
        }
      } else {
        streakRef.current = { letter: '', count: 0, confirmed: false };
        setStreakCount(0);
        setRecognitionState('IDLE');
      }

      // Backend confirmed letter hook fallback
      if (res.confirmed_letter && !streakRef.current.confirmed) {
        confirmLetter(res.confirmed_letter);
        streakRef.current.confirmed = true;
        setRecognitionState('CONFIRMED');
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
  }, [isActive, spellMode, isCallingLive, detectedHands, responseSpeed, requiredStreak, confirmLetter]);

  // Reset streak when hands disappear
  useEffect(() => {
    if (!isActive) {
      streakRef.current = { letter: '', count: 0, confirmed: false };
      setStreakCount(0);
      setCurrentLetter(null);
      setConfidence(0);
      setRecognitionState('IDLE');
    }
  }, [isActive]);

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
    streakRef.current = { letter: '', count: 0, confirmed: false };
    setStreakCount(0);
  };

  const handleClear = () => {
    setWordBuffer([]);
    setSentence('');
    streakRef.current = { letter: '', count: 0, confirmed: false };
    setStreakCount(0);
    mockIndexRef.current = 0;
  };

  const handleSuggestionClick = (word: string) => {
    setSentence((prev) => (prev ? prev + ' ' + word.toUpperCase() : word.toUpperCase()));
    setWordBuffer([]);
    streakRef.current = { letter: '', count: 0, confirmed: false };
    setStreakCount(0);
  };

  const confidencePct = Math.round(confidence * 100);
  const streakPct = Math.min(100, Math.round((streakCount / requiredStreak) * 100));

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
          className={`spell-mode-toggle-btn ${spellMode === 'live' ? 'active' : ''}`}
          onClick={() => {
            setSpellMode('live');
            setCurrentLetter(null);
            setConfidence(0);
            streakRef.current = { letter: '', count: 0, confirmed: false };
            setStreakCount(0);
          }}
        >
          ⚡ Live Backend Model
        </button>
        <button
          className={`spell-mode-toggle-btn ${spellMode === 'mock' ? 'active' : ''}`}
          onClick={() => {
            setSpellMode('mock');
            setLiveError(null);
          }}
        >
          🎭 Simulated Demo (Mock)
        </button>
      </div>

      {/* Speed Preset Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          margin: '8px 0 12px',
          padding: '6px 10px',
          borderRadius: '8px',
          background: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
        }}
      >
        <span style={{ fontSize: '0.78rem', color: '#94a3b8', fontWeight: 600 }}>Hold Time:</span>
        <div style={{ display: 'flex', gap: '4px' }}>
          {[
            { label: '⚡ Snappy (0.25s)', speed: 60, frames: 3 },
            { label: '🏎 Fast (0.4s)', speed: 80, frames: 5 },
            { label: '🛡 Stable (0.8s)', speed: 120, frames: 7 },
          ].map((mode) => (
            <button
              key={mode.speed}
              onClick={() => {
                setResponseSpeed(mode.speed);
                setRequiredStreak(mode.frames);
              }}
              style={{
                background: responseSpeed === mode.speed ? '#0284c7' : 'rgba(255, 255, 255, 0.05)',
                border: `1px solid ${responseSpeed === mode.speed ? '#38bdf8' : 'rgba(255, 255, 255, 0.1)'}`,
                color: responseSpeed === mode.speed ? '#ffffff' : '#94a3b8',
                borderRadius: '6px',
                padding: '3px 8px',
                fontSize: '0.74rem',
                cursor: 'pointer',
                fontWeight: responseSpeed === mode.speed ? 700 : 400,
              }}
            >
              {mode.label}
            </button>
          ))}
        </div>
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
          <strong>⚠ Live Recognition Notice:</strong> {liveError}
        </div>
      )}

      {spellMode === 'live' && !liveError && (
        <div className="spell-live-active-banner">
          ✓ Real-time inference at <strong>{Math.round(1000 / responseSpeed)} fps</strong> with Person B's ASL model.
        </div>
      )}

      {/* Current letter + confidence + Instant Add Button */}
      <div className="spell-letter-display">
        <span className="spell-big-letter" aria-live="polite">
          {currentLetter ?? '—'}
        </span>

        {/* Instant Add Button — No waiting necessary if hand hurts */}
        {currentLetter && (
          <div style={{ display: 'flex', justifyContent: 'center', marginTop: '6px' }}>
            <button
              onClick={() => {
                confirmLetter(currentLetter);
                streakRef.current.confirmed = true;
              }}
              style={{
                background: 'linear-gradient(135deg, #0284c7, #0369a1)',
                border: '1px solid rgba(56, 189, 248, 0.5)',
                color: '#ffffff',
                borderRadius: '8px',
                padding: '6px 16px',
                fontSize: '0.9rem',
                fontWeight: 700,
                cursor: 'pointer',
                boxShadow: '0 4px 12px rgba(2, 132, 199, 0.35)',
                transition: 'transform 0.1s ease',
              }}
              title={`Instantly add letter "${currentLetter}" without holding`}
            >
              + Add "{currentLetter}" Now
            </button>
          </div>
        )}

        {/* Real-time confidence bar */}
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

        {/* Live Stability / Confirmation Meter */}
        {isActive && currentLetter && (
          <div style={{ width: '100%', marginTop: '8px' }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '0.72rem',
                color: '#94a3b8',
                marginBottom: '3px',
              }}
            >
              <span>Auto-Confirm:</span>
              <span style={{ color: streakCount >= requiredStreak ? '#34d399' : '#38bdf8', fontWeight: 600 }}>
                {streakCount >= requiredStreak ? 'CONFIRMED ✓' : `${streakCount}/${requiredStreak} frames`}
              </span>
            </div>
            <div
              style={{
                height: '5px',
                background: 'rgba(255, 255, 255, 0.08)',
                borderRadius: '3px',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${streakPct}%`,
                  background: streakCount >= requiredStreak ? '#34d399' : '#38bdf8',
                  transition: 'width 60ms linear',
                }}
              />
            </div>
          </div>
        )}
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
