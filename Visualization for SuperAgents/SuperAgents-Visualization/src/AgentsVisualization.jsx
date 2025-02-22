import React, { useState, useCallback, useContext } from "react";
import ReactFlow, { MiniMap, Controls, Handle, useReactFlow, useEdgesState, useNodesState } from "reactflow";
import { MdModeEdit, MdOutlineCancel } from "react-icons/md";
import { FaSave } from "react-icons/fa";
import { AppContext } from "./Context";
import "reactflow/dist/style.css";
import "./AgentsVisualization.css"

// Function to generate nodes and edges from JSON
// const generateGraph = (data) => {
//   if (!data) {
//     const nodes = [];
//     const edges = [];
//     return { nodes, edges };
//   }
//   const nodes = data.agents.map((agent, index) => ({
//     id: agent.agent_id,
//     position: { x: index * 250, y: agent.parent_id ? 200 : 50 },
//     data: {
//       label: (
//         <div>
//           <strong>{agent.agent_id}</strong>
//           <p>{agent.role_name}</p>
//           <p>{agent.system_prompt}</p>
//           <p>{agent.task_prompt}</p>
//         </div>
//       )
//     },
//     style: { width: 200, padding: 10, borderRadius: 10, background: "#f3f3f3", border: "1px solid #ccc" }
//   }));

//   const edges = data.agents
//     .filter(agent => agent.parent_id)
//     .map(agent => ({
//       id: `edge-${agent.parent_id}-${agent.agent_id}`,
//       source: agent.parent_id,
//       target: agent.agent_id,
//       animated: true
//     }));

//   return { nodes, edges };
// };

const generateGraph = (data) => {
  if (!data || !data.agents) {
    return { nodes: [], edges: [] };
  }

  // Create a map of agent_id -> agent for easy lookup
  const agentMap = new Map(data.agents.map(agent => [agent.agent_id, agent]));

  // Determine levels of nodes
  const levels = new Map();
  const setLevel = (agent_id, level) => {
    if (levels.has(agent_id)) return; // Prevent re-processing
    levels.set(agent_id, level);
    const children = data.agents.filter(a => a.parent_id === agent_id);
    children.forEach(child => setLevel(child.agent_id, level + 1));
  };

  // Find the root node (parent_id === null) and start the level calculation
  data.agents.filter(a => a.parent_id === null).forEach(root => setLevel(root.agent_id, 0));

  // Group nodes by their levels
  const groupedLevels = {};
  levels.forEach((level, id) => {
    if (!groupedLevels[level]) groupedLevels[level] = [];
    groupedLevels[level].push(id);
  });

  // Generate nodes with computed positions
  const nodes = data.agents.map(agent => {
    const level = levels.get(agent.agent_id);
    const index = groupedLevels[level].indexOf(agent.agent_id); // Get index within the level
    const x = index * 250;  // Space nodes evenly in X direction
    const y = level * 200;  // Y position based on level

    return {
      id: agent.agent_id,
      position: { x, y },
      data: {
        label: (
          <div>
            <strong>{agent.agent_id}</strong>
            <p>{agent.role_name}</p>
            <p>{agent.system_prompt}</p>
            <p>{agent.task_prompt}</p>
          </div>
        ),
      },
      draggable: true, // Nodes are still draggable
      style: { 
        width: 200, 
        padding: 10, 
        borderRadius: 10, 
        background: "#f3f3f3", 
        border: "1px solid #ccc" 
      }
    };
  });

  // Generate edges (connections between parent and child nodes)
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
    <div className="react-flow-container">
      {/* React Flow Graph */}
      <div style={{ width: "100%", height: "100%" }} >
        <ReactFlow 
        nodes={nodes} 
        edges={edges} 
        onNodeClick={onNodeClick}
        fitView
        >
          <MiniMap />
          <Controls />
        </ReactFlow>
      </div>

      {/* Agent Details Panel */}
      {selectedAgent && (
        modify ? (
          <div className="agent-detail-panel">
            <h3>
              <label><strong>Role Name:</strong>
                <input name="role_name" className="edit-detail" value={editedData.role_name} onChange={handleChange} />
              </label>
            </h3>
            <p>
              <label><strong>System Prompt:</strong>
                <input name="system_prompt" className="edit-detail" value={editedData.system_prompt} onChange={handleChange} />
              </label>
            </p>
            <p>
              <label><strong>Task Prompt:</strong>
                <input name="task_prompt" className="edit-detail" value={editedData.task_prompt} onChange={handleChange} />
              </label>
            </p>
            <button className="save-btn" onClick={saveChange}>
              <FaSave />
              Save
            </button>
            <button className="cancel-btn" onClick={() => setModify(false)}>
              <MdOutlineCancel />
              Cancel
            </button>
          </div>
        ) : (
          <div className="agent-detail-panel">
            <h3>{selectedAgent.role_name}</h3>
            <p><strong>System Prompt:</strong> {selectedAgent.system_prompt}</p>
            <p><strong>Task Prompt:</strong> {selectedAgent.task_prompt}</p>
            <button className="edit-btn" onClick={() => setModify(true)}> 
              <MdModeEdit /> Edit
            </button>
          </div>
        )
      )}
    </div>
  );
};

export default AgentsVisualization;