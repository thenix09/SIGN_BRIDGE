import React, { useState, useEffect } from 'react';
import { StatusBadge, BadgeStatus } from '../common/StatusBadge';

interface HeaderProps {
  systemStatus?: BadgeStatus;
  statusLabel?: string;
  onOpenHistory?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  systemStatus = 'idle',
  statusLabel,
  onOpenHistory
}) => {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    return (localStorage.getItem('signbridge-theme') as 'dark' | 'light') || 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('signbridge-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  const defaultLabel = systemStatus === 'online' ? 'Camera Live' : 'Standby';

  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-logo-badge" aria-hidden="true">
          <svg
            className="brand-logo-svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M18 11V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2v3" />
            <path d="M14 9V4a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7" />
            <path d="M10 10.5V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2v8" />
            <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
          </svg>
        </div>
        <div className="brand-text-block">
          <h1 className="brand-title">
            <span className="brand-title-sign">Sign</span>
            <span className="brand-title-bridge">Bridge</span>
          </h1>
          <span className="brand-subtitle">AI Sign Language System</span>
        </div>
      </div>
      <div className="header-meta">
        {onOpenHistory && (
          <button
            type="button"
            className="btn btn-secondary header-history-btn"
            onClick={onOpenHistory}
            title="Open Translation History Drawer"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              fontSize: '0.85rem',
              fontWeight: 600,
              cursor: 'pointer'
            }}
          >
            <span>📜</span>
            <span>History</span>
          </button>
        )}
        <button
          type="button"
          className="theme-toggle-btn"
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          aria-label="Toggle theme"
        >
          <span>{theme === 'dark' ? '☀️' : '🌙'}</span>
        </button>
        <StatusBadge label={statusLabel || defaultLabel} status={systemStatus} />
      </div>
    </header>
  );
};
