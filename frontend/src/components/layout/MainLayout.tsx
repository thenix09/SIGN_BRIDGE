import React, { useState } from 'react';
import { Header } from './Header';
import { Navigation, ActiveTab } from './Navigation';
import { BadgeStatus } from '../common/StatusBadge';

interface MainLayoutProps {
  headerStatus?: BadgeStatus;
  headerLabel?: string;
  activeTab?: ActiveTab;
  onSelectTab?: (tab: ActiveTab) => void;
  onOpenHistory?: () => void;
  children: (activeTab: ActiveTab) => React.ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({
  headerStatus,
  headerLabel,
  activeTab: controlledActiveTab,
  onSelectTab: controlledOnSelectTab,
  onOpenHistory,
  children
}) => {
  const [internalActiveTab, setInternalActiveTab] = useState<ActiveTab>('spell');

  const currentTab = controlledActiveTab !== undefined ? controlledActiveTab : internalActiveTab;
  const handleTabChange = controlledOnSelectTab || setInternalActiveTab;

  return (
    <div className="app-container">
      <Header
        systemStatus={headerStatus}
        statusLabel={headerLabel}
        onOpenHistory={onOpenHistory}
      />
      <Navigation activeTab={currentTab} onSelectTab={handleTabChange} />
      <main className={`main-content ${currentTab === 'text_to_sign' ? 'main-content-wide' : ''}`}>
        {children(currentTab)}
      </main>
      <footer className="app-footer">
        <div className="footer-content">
          <p className="footer-description">
            <strong>SignBridge</strong> is a bidirectional assistive communication system bridging American Sign Language (ASL) and spoken text through real-time MediaPipe vision tracking, fingerspelling recognition, and animated sign synthesis.
          </p>
          <span className="footer-tagline">AI-Powered Accessibility Platform</span>
        </div>
      </footer>
    </div>
  );
};
