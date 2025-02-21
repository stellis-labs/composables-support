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
  const { jsonData, setJsonData } = useContext(AppContext);
  const { nodes, edges } = generateGraph(jsonData);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [modify, setModify] = useState(false);
  const [editedData, setEditedData] = useState({}); // Store changes before saving

  const onNodeClick = useCallback((event, node) => {
    const agent = jsonData.agents.find((a) => a.agent_id === node.id);
    setModify(false);
    setSelectedAgent(agent);
    setEditedData(agent); // Initialize with existing data
  }, [jsonData]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setEditedData((prev) => ({ ...prev, [name]: value }));
  };

  const saveChange = () => {
    if (!selectedAgent) return;

    // Update jsonData with new changes
    setJsonData((prevData) => ({
      ...prevData,
      agents: prevData.agents.map((agent) =>
        agent.agent_id === selectedAgent.agent_id ? { ...agent, ...editedData } : agent
      ),
    }));

    setModify(false);
  };

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
        modify ? (
          <div style={{ width: "30%", padding: 20, borderLeft: "1px solid #ddd", background: "#f9f9f9" }}>
            <h3>
              <label><strong>Role Name:</strong>
                <input name="role_name" value={editedData.role_name} onChange={handleChange} />
              </label>
            </h3>
            <p>
              <label><strong>System Prompt:</strong>
                <input name="system_prompt" value={editedData.system_prompt} onChange={handleChange} />
              </label>
            </p>
            <p>
              <label><strong>Task Prompt:</strong>
                <input name="task_prompt" value={editedData.task_prompt} onChange={handleChange} />
              </label>
            </p>
            <p>
              <label><strong>Created At:</strong>
                <input name="metadata.creation_timestamp" value={editedData.metadata.creation_timestamp} readOnly />
              </label>
            </p>
            <p>
              <label><strong>LLM Used:</strong>
                <input name="metadata.llm_used" value={editedData.metadata.llm_used} readOnly />
              </label>
            </p>
            <button onClick={saveChange}>Save</button>
            <button onClick={() => setModify(false)}>Cancel</button>
          </div>
        ) : (
          <div style={{ width: "30%", padding: 20, borderLeft: "1px solid #ddd", background: "#f9f9f9" }}>
            <h3>{selectedAgent.role_name}</h3>
            <p><strong>System Prompt:</strong> {selectedAgent.system_prompt}</p>
            <p><strong>Task Prompt:</strong> {selectedAgent.task_prompt}</p>
            <p><strong>Created At:</strong> {selectedAgent.metadata.creation_timestamp}</p>
            <p><strong>LLM Used:</strong> {selectedAgent.metadata.llm_used}</p>
            <button onClick={() => setModify(true)}>Edit</button>
          </div>
        )
      )}
    </div>
  );
};

export default AgentsVisualization;