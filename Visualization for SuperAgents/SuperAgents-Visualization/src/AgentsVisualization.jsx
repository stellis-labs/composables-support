import React, { useState, useCallback, useContext } from "react";
import ReactFlow, { MiniMap, Controls, Handle } from "reactflow";
import { AppContext } from "./Context";
import "reactflow/dist/style.css";

// Function to generate nodes and edges from JSON
const generateGraph = (data) => {
  if (!data) {
    const nodes = [];
    const edges = [];
    return { nodes, edges };
  }
  const nodes = data.agents.map((agent, index) => ({
    id: agent.agent_id,
    position: { x: index * 250, y: agent.parent_id ? 200 : 50 },
    data: {
      label: (
        <div>
          <strong>{agent.agent_id}</strong>
          <p>{agent.role_name}</p>
          <p>{agent.system_prompt}</p>
          <p>{agent.task_prompt}</p>
        </div>
      )
    },
    style: { width: 200, padding: 10, borderRadius: 10, background: "#f3f3f3", border: "1px solid #ccc" }
  }));

  const edges = data.agents
    .filter(agent => agent.parent_id)
    .map(agent => ({
      id: `edge-${agent.parent_id}-${agent.agent_id}`,
      source: agent.parent_id,
      target: agent.agent_id,
      animated: true
    }));

  return { nodes, edges };
};

// Component for the React FLow visualization panel and the agent detail panel
const AgentsVisualization = () => {
    const {jsonData, setJsonData} = useContext(AppContext);
    const { nodes, edges } = generateGraph(jsonData);
    const [selectedAgent, setSelectedAgent] = useState(null);
  
    const onNodeClick = useCallback((event, node) => {
      const agent = jsonData.agents.find(a => a.agent_id === node.id);
      setSelectedAgent(agent);
    }, [jsonData]);
  
    return (
      <div style={{ width: "100%", height: "500px", display: "flex" }}>
        {/* React Flow Graph */}
        <div style={{ width: "70%", height: "100%" }}>
          <ReactFlow nodes={nodes} edges={edges} onNodeClick={onNodeClick}>
            <MiniMap />
            <Controls />
          </ReactFlow>
        </div>
  
        {/* Agent Details Panel */}
        {selectedAgent && (
          <div style={{ width: "30%", padding: 20, borderLeft: "1px solid #ddd", background: "#f9f9f9" }}>
            <h3>{selectedAgent.role_name}</h3>
            <p><strong>System Prompt:</strong> {selectedAgent.system_prompt}</p>
            <p><strong>Task Prompt:</strong> {selectedAgent.task_prompt}</p>
            <p><strong>Created At:</strong> {selectedAgent.metadata.creation_timestamp}</p>
            <p><strong>LLM Used:</strong> {selectedAgent.metadata.llm_used}</p>
          </div>
        )}
      </div>
    );
};

export default AgentsVisualization;