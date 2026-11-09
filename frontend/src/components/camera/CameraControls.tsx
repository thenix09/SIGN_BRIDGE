import React from 'react';
import { CameraStatus } from '../../types/camera';
import { StatusBadge } from '../common/StatusBadge';

interface CameraControlsProps {
  status: CameraStatus;
  onStart: () => void;
  onStop: () => void;
}

export const CameraControls: React.FC<CameraControlsProps> = ({
  status,
  onStart,
  onStop
}) => {
  const getBadgeConfig = (
    currentStatus: CameraStatus
  ): { label: string; badgeStatus: 'online' | 'idle' | 'warning' | 'error' | 'requesting' } => {
    switch (currentStatus) {
      case 'active':
        return { label: 'Camera Active', badgeStatus: 'online' };
      case 'requesting':
        return { label: 'Requesting...', badgeStatus: 'requesting' };
      case 'denied':
        return { label: 'Permission Denied', badgeStatus: 'warning' };
      case 'error':
        return { label: 'Error', badgeStatus: 'error' };
      case 'idle':
      default:
        return { label: 'Camera Off', badgeStatus: 'idle' };
    }
  };

  const { label, badgeStatus } = getBadgeConfig(status);

  return (
    <div className="camera-controls-container">
      <div className="controls-status-row">
        <span className="control-label">Status:</span>
        <StatusBadge label={label} status={badgeStatus} />
      </div>

      <div className="controls-actions-row">
        {status !== 'active' ? (
          <button
            className="button"
            onClick={onStart}
            disabled={status === 'requesting'}
            style={{ width: '100%' }}
          >
            {status === 'requesting' ? 'Requesting Permission...' : 'Start Camera'}
          </button>
        ) : (
          <button
            className="button button-danger"
            onClick={onStop}
            style={{ width: '100%' }}
          >
            Stop Camera
          </button>
        )}
      </div>
    </div>
  );
};
