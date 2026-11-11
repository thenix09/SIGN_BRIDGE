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
            className="header-history-btn"
            onClick={onOpenHistory}
            title="Open Translation History Drawer"
            aria-label="Translation History"
          >
            <svg
              className="header-history-icon"
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
              <path d="M12 7v5l4 2" />
            </svg>
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
          {theme === 'dark' ? (
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <circle cx="12" cy="12" r="4" />
              <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
            </svg>
          ) : (
            <svg
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
            </svg>
          )}
        </button>
        <StatusBadge label={statusLabel || defaultLabel} status={systemStatus} />
      </div>
    </header>
  );
};
