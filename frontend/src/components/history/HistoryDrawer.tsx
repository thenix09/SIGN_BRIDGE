import React, { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../../services/api.client';
import { HistoryItemResponse } from '../../types/api';

const LOCAL_STORAGE_KEY = 'signbridge_history_cache';

interface HistoryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectPhrase?: (phrase: string, direction: string) => void;
}

export const HistoryDrawer: React.FC<HistoryDrawerProps> = ({
  isOpen,
  onClose,
  onSelectPhrase
}) => {
  const [items, setItems] = useState<HistoryItemResponse[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isOffline, setIsOffline] = useState<boolean>(false);

  // Load history from backend, falling back to local storage
  const loadHistory = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await apiClient.getHistory(50);
      setItems(res.items);
      setIsOffline(false);
      try {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(res.items));
      } catch {
        // Ignore storage quotas
      }
    } catch {
      // Backend unavailable; fall back to local storage cache
      setIsOffline(true);
      try {
        const cached = localStorage.getItem(LOCAL_STORAGE_KEY);
        if (cached) {
          const parsed = JSON.parse(cached);
          if (Array.isArray(parsed)) {
            setItems(parsed);
          }
        }
      } catch {
        setItems([]);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      loadHistory();
    }
  }, [isOpen, loadHistory]);

  const handleClear = async () => {
    try {
      await apiClient.clearHistory();
    } catch {
      // Offline fallback
    }
    localStorage.removeItem(LOCAL_STORAGE_KEY);
    setItems([]);
  };

  const formatTimestamp = (raw: string): string => {
    try {
      const date = new Date(raw);
      if (isNaN(date.getTime())) return raw;
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) +
        ' • ' +
        date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } catch {
      return raw;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="history-drawer-overlay" onClick={onClose} aria-label="History Drawer Overlay">
      <div
        className="history-drawer-panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Translation History"
      >
        {/* Drawer Header */}
        <div className="history-drawer-header">
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h2 style={{ margin: 0, fontSize: '1.2rem', color: '#f8fafc' }}>Translation History</h2>
              <span className="brand-badge" style={{ fontSize: '0.75rem' }}>
                {items.length} {items.length === 1 ? 'record' : 'records'}
              </span>
            </div>
            <p style={{ margin: '4px 0 0', fontSize: '0.8rem', color: '#94a3b8' }}>
              {isOffline
                ? 'ℹ Showing local cached history (backend offline)'
                : '✓ Synchronized with backend (GET /api/history)'}
            </p>
          </div>
          <button
            onClick={onClose}
            className="history-close-btn"
            title="Close Drawer"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* Drawer Controls Bar */}
        <div className="history-drawer-actions">
          <button
            onClick={loadHistory}
            className="btn btn-secondary"
            style={{ padding: '6px 12px', fontSize: '0.8rem' }}
            disabled={loading}
          >
            {loading ? 'Refreshing...' : '↻ Refresh'}
          </button>
          {items.length > 0 && (
            <button
              onClick={handleClear}
              className="btn btn-secondary"
              style={{
                padding: '6px 12px',
                fontSize: '0.8rem',
                color: '#f87171',
                borderColor: 'rgba(239, 68, 68, 0.3)'
              }}
            >
              Clear All History
            </button>
          )}
        </div>

        {error && (
          <div
            style={{
              margin: '12px 16px',
              padding: '8px 12px',
              borderRadius: '6px',
              background: 'rgba(239, 68, 68, 0.15)',
              color: '#f87171',
              fontSize: '0.82rem'
            }}
          >
            {error}
          </div>
        )}

        {/* List of History Items */}
        <div className="history-list-container">
          {loading && items.length === 0 ? (
            <div className="history-empty-state">
              <div className="spinner" style={{ width: "24px", height: "24px", margin: "0 auto 8px" }} />
              <p>Loading history records...</p>
            </div>
          ) : items.length === 0 ? (
            <div className="history-empty-state">
              <div style={{ display: 'flex', justifyContent: 'center', opacity: 0.35, marginBottom: '8px' }}>
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                  <path d="M3 3v5h5" />
                  <path d="M12 7v5l4 2" />
                </svg>
              </div>
              <p style={{ fontWeight: 600, color: '#f8fafc', margin: '8px 0 4px' }}>
                No translation history yet
              </p>
              <p style={{ fontSize: '0.82rem', color: '#94a3b8', maxWidth: '280px' }}>
                Phrases converted in <strong>Text to Sign</strong> or recognized in <strong>Fingerspelling</strong> and <strong>Word Signs</strong> will appear here automatically.
              </p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {items.map((item, idx) => {
                const isTts = item.direction === 'text_to_sign';
                return (
                  <div key={item.id ?? idx} className="history-item-card">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
                      <span
                        style={{
                          fontSize: '0.72rem',
                          fontWeight: 700,
                          padding: '2px 8px',
                          borderRadius: '8px',
                          background: isTts ? 'rgba(56, 189, 248, 0.15)' : 'rgba(52, 211, 153, 0.15)',
                          color: isTts ? '#38bdf8' : '#34d399',
                          border: `1px solid ${isTts ? 'rgba(56, 189, 248, 0.3)' : 'rgba(52, 211, 153, 0.3)'}`
                        }}
                      >
                        {isTts ? 'Text → Sign' : 'Sign → Text'}
                      </span>
                      <span style={{ fontSize: '0.72rem', color: '#64748b' }}>
                        {formatTimestamp(item.created_at)}
                      </span>
                    </div>

                    <div style={{ marginTop: '8px', fontSize: '1rem', fontWeight: 600, color: '#f8fafc', wordBreak: 'break-word' }}>
                      {item.content}
                    </div>

                    <div style={{ marginTop: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                        Language: {item.language || 'ASL'}
                      </span>
                      {onSelectPhrase && (
                        <button
                          onClick={() => {
                            onSelectPhrase(item.content, item.direction);
                            onClose();
                          }}
                          className="btn btn-secondary"
                          style={{
                            padding: '3px 10px',
                            fontSize: '0.75rem',
                            color: '#38bdf8',
                            borderColor: 'rgba(56, 189, 248, 0.3)'
                          }}
                        >
                          {isTts ? '↺ Replay in Player' : 'Revisit'}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
