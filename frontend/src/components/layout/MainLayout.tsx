import React, { useState } from 'react';
import { Header } from './Header';
import { Navigation, ActiveTab } from './Navigation';
import { BadgeStatus } from '../common/StatusBadge';

interface MainLayoutProps {
  headerStatus?: BadgeStatus;
  headerLabel?: string;
  children: (activeTab: ActiveTab) => React.ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({
  headerStatus,
  headerLabel,
  children
}) => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('spell');

  return (
    <div className="app-container">
      <Header systemStatus={headerStatus} statusLabel={headerLabel} />
      <Navigation activeTab={activeTab} onSelectTab={setActiveTab} />
      <main className="main-content">
        {children(activeTab)}
      </main>
      <footer className="app-footer">
        <span>SignBridge Accessibility Project • Team of 3</span>
        <span>Person C (Frontend & Docs)</span>
      </footer>
    </div>
  );
};
