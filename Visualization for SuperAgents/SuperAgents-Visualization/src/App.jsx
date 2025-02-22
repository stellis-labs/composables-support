import React from "react";
import AgentsVisualization from "./AgentsVisualization";
import JSONImportExport from "./JSONImportExport";
import "./App.css"

function App() {
  return (
    <div className="app-container">
      <AgentsVisualization />
      <JSONImportExport />
    </div>
  );
}

export default App;
