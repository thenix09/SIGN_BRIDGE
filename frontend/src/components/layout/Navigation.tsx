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
        Sign to Text (Spell)
      </button>
      <button
        className={`nav-tab ${activeTab === 'text_to_sign' ? 'active' : ''}`}
        onClick={() => onSelectTab('text_to_sign')}
      >
        Text to Sign (Animation)
      </button>
      <button
        className={`nav-tab ${activeTab === 'words' ? 'active' : ''}`}
        onClick={() => onSelectTab('words')}
      >
        Word Signs (Words)
      </button>
      <button
        className={`nav-tab ${activeTab === 'collect' ? 'active' : ''}`}
        onClick={() => onSelectTab('collect')}
      >
        Dataset Collection
      </button>
    </nav>
  );
};
