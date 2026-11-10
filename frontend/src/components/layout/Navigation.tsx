import React from 'react';

export type ActiveTab = 'spell' | 'words' | 'text_to_sign' | 'collect' | 'settings';

interface NavigationProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
}

export const Navigation: React.FC<NavigationProps> = ({ activeTab, onSelectTab }) => {
  return (
    <nav className="nav-tabs" aria-label="Main Navigation">
      <button
        className={`nav-tab ${activeTab === 'spell' ? 'active' : ''}`}
        onClick={() => onSelectTab('spell')}
      >
        <span aria-hidden="true">🔤</span>
        <span>Fingerspell</span>
      </button>
      <button
        className={`nav-tab ${activeTab === 'words' ? 'active' : ''}`}
        onClick={() => onSelectTab('words')}
      >
        <span aria-hidden="true">💬</span>
        <span>Word Signs</span>
      </button>
      <button
        className={`nav-tab ${activeTab === 'text_to_sign' ? 'active' : ''}`}
        onClick={() => onSelectTab('text_to_sign')}
      >
        <span aria-hidden="true">🤖</span>
        <span>Text to Sign</span>
      </button>
      <button
        className={`nav-tab ${activeTab === 'collect' ? 'active' : ''}`}
        onClick={() => onSelectTab('collect')}
      >
        <span aria-hidden="true">🎯</span>
        <span>Dataset</span>
      </button>
    </nav>
  );
};
