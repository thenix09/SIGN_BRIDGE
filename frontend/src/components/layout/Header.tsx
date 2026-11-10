import React from 'react';
import { StatusBadge, BadgeStatus } from '../common/StatusBadge';

interface HeaderProps {
  systemStatus?: BadgeStatus;
  statusLabel?: string;
}

export const Header: React.FC<HeaderProps> = ({
  systemStatus = 'idle',
  statusLabel
}) => {
  const defaultLabel = systemStatus === 'online' ? 'Camera Live' : 'Standby';

  return (
    <header className="app-header">
      <div className="brand-section">
        <h1 className="brand-title">SignBridge</h1>
        <span className="brand-badge">Day 1 • Webcam Integration</span>
      </div>
      <div className="header-meta">
        <StatusBadge label={statusLabel || defaultLabel} status={systemStatus} />
      </div>
    </header>
  );
};
