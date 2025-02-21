import React from "react";
import AgentsVisualization from "./AgentsVisualization";
import JSONImportExport from "./JSONImportExport";
import "./App.css"

function App() {
  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      <AgentsVisualization />
      <JSONImportExport />
    </div>
  );
}

export default App;
