import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './app/App';
import { AppStoreProvider } from './app/store';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppStoreProvider>
      <App />
    </AppStoreProvider>
  </React.StrictMode>
);
