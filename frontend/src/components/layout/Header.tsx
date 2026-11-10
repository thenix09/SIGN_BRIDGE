import React from 'react';
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
  const defaultLabel = systemStatus === 'online' ? 'Camera Live' : 'Standby';

  return (
    <header className="app-header">
      <div className="brand-section">
        <h1 className="brand-title">SignBridge</h1>
        <span className="brand-badge">Day 4 • Multimodal Sign Platform</span>
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
        <StatusBadge label={statusLabel || defaultLabel} status={systemStatus} />
      </div>
    </header>
  );
};
