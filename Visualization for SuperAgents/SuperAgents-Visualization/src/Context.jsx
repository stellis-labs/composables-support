import React, { createContext, useState, useEffect } from 'react';

// Create the App Context
export const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [jsonData, setJsonData] = useState(null);

  // Context value to be shared across the app
  const value = {
    jsonData,
    setJsonData
  };

  return (
    <AppContext.Provider value={value}>
      {children} {/* Render child components */}
    </AppContext.Provider>
  );
};