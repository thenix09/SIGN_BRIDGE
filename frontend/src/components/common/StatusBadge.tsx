import React from 'react';

export type BadgeStatus = 'online' | 'idle' | 'warning' | 'error' | 'requesting';

interface StatusBadgeProps {
  label: string;
  status: BadgeStatus;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ label, status }) => {
  return (
    <span className={`status-pill status-${status}`}>
      <span className="status-dot" />
      {label}
    </span>
  );
};
