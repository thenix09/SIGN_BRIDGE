import React from 'react';

export type ActiveTab = 'spell' | 'words' | 'text_to_sign' | 'collect' | 'settings';

interface NavigationProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
}

export const Navigation: React.FC<NavigationProps> = ({ activeTab, onSelectTab }) => {
  return (
    <div className="nav-container">
      <nav className="nav-tabs" aria-label="Main Navigation">
        <button
          className={`nav-tab ${activeTab === 'spell' ? 'active' : ''}`}
          onClick={() => onSelectTab('spell')}
        >
          <svg
            className="nav-tab-icon"
            viewBox="0 0 24 24"
            width="14"
            height="14"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M18 11V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2v3" />
            <path d="M14 9V4a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7" />
            <path d="M10 10.5V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2v8" />
            <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15" />
          </svg>
          <span>Fingerspell</span>
        </button>

        <button
          className={`nav-tab ${activeTab === 'words' ? 'active' : ''}`}
          onClick={() => onSelectTab('words')}
        >
          <svg
            className="nav-tab-icon"
            viewBox="0 0 24 24"
            width="14"
            height="14"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M2 12h5l3 7 4-14 3 7h5" />
          </svg>
          <span>Word Signs</span>
        </button>

        <button
          className={`nav-tab ${activeTab === 'text_to_sign' ? 'active' : ''}`}
          onClick={() => onSelectTab('text_to_sign')}
        >
          <svg
            className="nav-tab-icon"
            viewBox="0 0 24 24"
            width="14"
            height="14"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
          <span>Text to Sign</span>
        </button>

        <button
          className={`nav-tab ${activeTab === 'collect' ? 'active' : ''}`}
          onClick={() => onSelectTab('collect')}
        >
          <svg
            className="nav-tab-icon"
            viewBox="0 0 24 24"
            width="14"
            height="14"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <ellipse cx="12" cy="5" rx="9" ry="3" />
            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
          </svg>
          <span>Dataset</span>
        </button>
      </nav>
    </div>
  );
};
