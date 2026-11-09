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
        <span>SignBridge Accessibility Project • Team of 3</span>
        <span>Person C (Frontend & Docs)</span>
      </footer>
    </div>
  );
};
