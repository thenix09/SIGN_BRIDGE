import React, { useState, useRef, useEffect, useCallback } from 'react';
import { apiClient } from '../../services/api.client';
import { TextToSignResponse, SignFrameResponse } from '../../types/api';
import { HAND_CONNECTIONS } from '../../types/landmarks';

const QUICK_PHRASES = ['HELLO', 'THANK YOU', 'WELCOME', 'ASL', 'GOOD MORNING', 'HELP'];

export const TextToSignPlayer: React.FC = () => {
  const [inputText, setInputText] = useState('HELLO');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [signData, setSignData] = useState<TextToSignResponse | null>(null);

  // Playback state
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const timerRef = useRef<number | null>(null);

  // Translate text via backend API
  const handleTranslate = useCallback(async (textToTranslate?: string) => {
    const query = (textToTranslate !== undefined ? textToTranslate : inputText).trim();
    if (!query) return;

    setLoading(true);
    setError(null);
    setIsPlaying(false);
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
    }

    try {
      const res = await apiClient.textToSign({ text: query, language: 'ASL' });
      if (!res.frames || res.frames.length === 0) {
        setError('No sign animation frames generated for this text.');
        setSignData(null);
      } else {
        setSignData(res);
        setCurrentFrameIndex(0);
        setIsPlaying(true);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to translate text to sign language.');
      setSignData(null);
    } finally {
      setLoading(false);
    }
  }, [inputText]);

  // Initial translation on component mount
  useEffect(() => {
    handleTranslate('HELLO');
    return () => {
      if (timerRef.current) {
        window.clearTimeout(timerRef.current);
      }
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Frame animation timer loop
  useEffect(() => {
    if (!isPlaying || !signData || signData.frames.length === 0) {
      if (timerRef.current) window.clearTimeout(timerRef.current);
      return;
    }

    const currentFrame = signData.frames[currentFrameIndex];
    const duration = (currentFrame?.duration_ms || 600) / playbackSpeed;

    timerRef.current = window.setTimeout(() => {
      setCurrentFrameIndex((prev) => {
        if (prev + 1 >= signData.frames.length) {
          setIsPlaying(false);
          return prev; // stop at last frame
        }
        return prev + 1;
      });
    }, duration);

    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, [isPlaying, currentFrameIndex, signData, playbackSpeed]);

  // Render hand on canvas
  const renderFrameOnCanvas = useCallback((frame: SignFrameResponse) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Dark sleek background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Subtle grid background lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const points = frame.right_hand || frame.left_hand;
    if (!points || points.length < 21) {
      // Pause or resting frame
      ctx.fillStyle = '#64748b';
      ctx.font = '600 16px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('[ Resting / Word Pause ]', width / 2, height / 2);
      return;
    }

    // Auto-center and scale hand to fit canvas comfortably
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    points.forEach((pt) => {
      if (pt[0] < minX) minX = pt[0];
      if (pt[0] > maxX) maxX = pt[0];
      if (pt[1] < minY) minY = pt[1];
      if (pt[1] > maxY) maxY = pt[1];
    });

    const handW = Math.max(maxX - minX, 0.1);
    const handH = Math.max(maxY - minY, 0.1);

    const targetH = height * 0.72;
    const targetW = width * 0.72;
    const scale = Math.min(targetW / handW, targetH / handH);

    const handCenterX = (minX + maxX) / 2;
    const handCenterY = (minY + maxY) / 2;

    const canvasCenterX = width / 2;
    const canvasCenterY = height / 2;

    const toScreen = (pt: number[]): { x: number; y: number } => ({
      x: canvasCenterX + (pt[0] - handCenterX) * scale,
      y: canvasCenterY + (pt[1] - handCenterY) * scale,
    });

    // 1. Draw Bones / Connections
    ctx.lineWidth = 5;
    ctx.strokeStyle = '#38bdf8'; // Sky blue
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    HAND_CONNECTIONS.forEach(({ start, end }) => {
      if (start < points.length && end < points.length) {
        const p1 = toScreen(points[start]);
        const p2 = toScreen(points[end]);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    });

    // 2. Draw 21 Joints
    const fingertipIndices = [4, 8, 12, 16, 20];
    points.forEach((pt, idx) => {
      const scr = toScreen(pt);
      const isTip = fingertipIndices.includes(idx);
      const radius = isTip ? 7.5 : 5;

      // Outer glow / border
      ctx.beginPath();
      ctx.arc(scr.x, scr.y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = isTip ? '#fbbf24' : '#ffffff'; // Amber tip, white border
      ctx.fill();

      // Inner core
      ctx.beginPath();
      ctx.arc(scr.x, scr.y, radius - 2, 0, 2 * Math.PI);
      ctx.fillStyle = isTip ? '#d97706' : '#0284c7';
      ctx.fill();
    });

    // 3. Draw Watermark / Frame Badge inside canvas
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.beginPath();
    ctx.roundRect(16, 16, 150, 40, 8);
    ctx.fill();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = '#38bdf8';
    ctx.font = '700 18px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(frame.kind === 'pause' ? 'PAUSE' : `SIGN: ${frame.label}`, 28, 42);
  }, []);

  // Update canvas on frame change
  useEffect(() => {
    if (signData && signData.frames.length > 0) {
      const frame = signData.frames[currentFrameIndex];
      if (frame) {
        renderFrameOnCanvas(frame);
      }
    }
  }, [currentFrameIndex, signData, renderFrameOnCanvas]);

  const activeFrame = signData?.frames[currentFrameIndex];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.6fr) minmax(320px, 1fr)', gap: '24px' }}>
      {/* Visual Canvas Player Section */}
      <section className="card" style={{ padding: '24px' }}>
        <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 style={{ margin: 0, fontSize: '1.25rem', color: '#f8fafc' }}>Text-to-Sign Demonstration</h2>
            <p style={{ margin: '4px 0 0', fontSize: '0.85rem', color: '#94a3b8' }}>
              Synthesizes 3D skeletal ASL animations from English text.
            </p>
          </div>
          {activeFrame && (
            <span
              style={{
                background: activeFrame.kind === 'word' ? 'rgba(52, 211, 153, 0.15)' : 'rgba(56, 189, 248, 0.15)',
                border: `1px solid ${activeFrame.kind === 'word' ? 'rgba(52, 211, 153, 0.4)' : 'rgba(56, 189, 248, 0.4)'}`,
                color: activeFrame.kind === 'word' ? '#34d399' : '#38bdf8',
                borderRadius: '12px',
                padding: '4px 12px',
                fontSize: '0.85rem',
                fontWeight: 700,
              }}
            >
              {activeFrame.kind === 'word' ? 'WORD SIGN' : activeFrame.kind === 'letter' ? 'LETTER' : 'PAUSE'}
            </span>
          )}
        </div>

        {/* Canvas Display */}
        <div
          style={{
            marginTop: '16px',
            borderRadius: '12px',
            overflow: 'hidden',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.35)',
            position: 'relative',
          }}
        >
          <canvas
            ref={canvasRef}
            width={640}
            height={460}
            style={{ width: '100%', height: 'auto', display: 'block', aspectRatio: '640 / 460' }}
          />

          {/* Active letter banner overlay */}
          {activeFrame && (
            <div
              style={{
                position: 'absolute',
                bottom: '16px',
                left: '50%',
                transform: 'translateX(-50%)',
                background: 'rgba(15, 23, 42, 0.9)',
                border: '1px solid rgba(56, 189, 248, 0.3)',
                borderRadius: '24px',
                padding: '6px 20px',
                color: '#ffffff',
                fontSize: '1rem',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
              }}
            >
              <span style={{ color: '#94a3b8' }}>Signing:</span>
              <span style={{ color: '#38bdf8', fontSize: '1.25rem', fontWeight: 800 }}>
                {activeFrame.label || '—'}
              </span>
            </div>
          )}
        </div>

        {/* Timeline Scrubber & Progress */}
        {signData && signData.frames.length > 0 && (
          <div style={{ marginTop: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '6px' }}>
              <span>Frame {currentFrameIndex + 1} of {signData.frames.length}</span>
              <span>{Math.round(((currentFrameIndex + 1) / signData.frames.length) * 100)}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={signData.frames.length - 1}
              value={currentFrameIndex}
              onChange={(e) => {
                setIsPlaying(false);
                setCurrentFrameIndex(Number(e.target.value));
              }}
              style={{
                width: '100%',
                accentColor: '#38bdf8',
                cursor: 'pointer',
              }}
            />
          </div>
        )}

        {/* Playback Controls Toolbar */}
        <div
          style={{
            marginTop: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '12px',
          }}
        >
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={() => {
                if (currentFrameIndex >= (signData?.frames.length || 1) - 1) {
                  setCurrentFrameIndex(0);
                }
                setIsPlaying(!isPlaying);
              }}
              className="btn btn-primary"
              style={{ minWidth: '100px' }}
              disabled={!signData || signData.frames.length === 0}
            >
              {isPlaying ? '⏸ Pause' : currentFrameIndex >= (signData?.frames.length || 1) - 1 ? '↺ Replay' : '▶ Play'}
            </button>
            <button
              onClick={() => {
                setIsPlaying(false);
                setCurrentFrameIndex(0);
              }}
              className="btn btn-secondary"
              disabled={!signData || signData.frames.length === 0}
            >
              Reset
            </button>
          </div>

          {/* Speed Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Speed:</span>
            {[0.5, 1.0, 1.5].map((spd) => (
              <button
                key={spd}
                onClick={() => setPlaybackSpeed(spd)}
                style={{
                  background: playbackSpeed === spd ? '#0284c7' : 'rgba(255, 255, 255, 0.05)',
                  border: `1px solid ${playbackSpeed === spd ? '#38bdf8' : 'rgba(255, 255, 255, 0.1)'}`,
                  color: playbackSpeed === spd ? '#ffffff' : '#94a3b8',
                  borderRadius: '6px',
                  padding: '4px 8px',
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                }}
              >
                {spd}x
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Controls & Sentence Input Sidebar */}
      <aside className="card" style={{ padding: '24px' }}>
        <h3 style={{ margin: '0 0 16px', fontSize: '1.1rem', color: '#f8fafc' }}>Translate English to Sign</h3>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleTranslate();
          }}
        >
          <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: '8px' }}>
            Enter Word or Sentence:
          </label>
          <div style={{ display: 'flex', gap: '8px' }}>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="e.g. HELLO, THANK YOU..."
              style={{
                flex: 1,
                background: 'rgba(15, 23, 42, 0.6)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                borderRadius: '8px',
                padding: '10px 14px',
                color: '#ffffff',
                fontSize: '0.95rem',
                outline: 'none',
              }}
            />
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !inputText.trim()}
              style={{ padding: '10px 18px', whiteSpace: 'nowrap' }}
            >
              {loading ? 'Translating...' : 'Translate'}
            </button>
          </div>
        </form>

        {error && (
          <div
            style={{
              marginTop: '12px',
              padding: '10px 14px',
              borderRadius: '8px',
              background: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              color: '#f87171',
              fontSize: '0.85rem',
            }}
          >
            {error}
          </div>
        )}

        {/* Quick Example Chips */}
        <div style={{ marginTop: '24px' }}>
          <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: '8px' }}>
            Try Quick Examples:
          </label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {QUICK_PHRASES.map((phrase) => (
              <button
                key={phrase}
                onClick={() => {
                  setInputText(phrase);
                  handleTranslate(phrase);
                }}
                style={{
                  background: 'rgba(56, 189, 248, 0.08)',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                  borderRadius: '16px',
                  padding: '6px 12px',
                  color: '#38bdf8',
                  fontSize: '0.85rem',
                  cursor: 'pointer',
                  fontWeight: 500,
                }}
              >
                {phrase}
              </button>
            ))}
          </div>
        </div>

        {/* Translated Token Segments Summary */}
        {signData && signData.segments && signData.segments.length > 0 && (
          <div style={{ marginTop: '28px', borderTop: '1px solid rgba(255, 255, 255, 0.08)', paddingTop: '16px' }}>
            <h4 style={{ margin: '0 0 10px', fontSize: '0.9rem', color: '#94a3b8' }}>Translation Breakdown</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {signData.segments.map((seg, i) => (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    background: 'rgba(255, 255, 255, 0.03)',
                    borderRadius: '8px',
                    padding: '8px 12px',
                    fontSize: '0.85rem',
                  }}
                >
                  <span style={{ fontWeight: 600, color: '#f8fafc', textTransform: 'uppercase' }}>{seg.token}</span>
                  <span
                    style={{
                      fontSize: '0.75rem',
                      color: seg.kind === 'word' ? '#34d399' : '#38bdf8',
                      background: seg.kind === 'word' ? 'rgba(52, 211, 153, 0.1)' : 'rgba(56, 189, 248, 0.1)',
                      padding: '2px 8px',
                      borderRadius: '10px',
                    }}
                  >
                    {seg.kind === 'word' ? 'Lexical Word' : `Spelled (${seg.end_index - seg.start_index} letters)`}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </aside>
    </div>
  );
};
