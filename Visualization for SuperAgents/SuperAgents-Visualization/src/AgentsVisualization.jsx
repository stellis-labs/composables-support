import React, { useState, useCallback } from "react";
import ReactFlow, { MiniMap, Controls, Handle } from "reactflow";
import "reactflow/dist/style.css";

const agentData = {
  agents: [
    {
      agent_id: "agent_1",
      parent_id: null,
      related_agents: ["agent_2", "agent_3"],
      role_name: "Market Research Agent",
      system_prompt: "Conduct market research for new product launches.",
      task_prompt: "Analyze market trends and customer preferences.",
      metadata: {
        creation_timestamp: "2023-10-01T12:00:00Z",
        llm_used: "Ollama"
      }
    },
    {
      agent_id: "agent_2",
      parent_id: "agent_1",
      related_agents: ["agent_4"],
      role_name: "Content Creation Agent",
      system_prompt: "Create marketing content for campaigns.",
      task_prompt: "Develop blog posts, social media content, and ads.",
      metadata: {
        creation_timestamp: "2023-10-01T12:05:00Z",
        llm_used: "Ollama"
      }
    },
    {
      agent_id: "agent_3",
      parent_id: "agent_1",
      related_agents: [],
      role_name: "Budget Planning Agent",
      system_prompt: "Plan marketing budgets for campaigns.",
      task_prompt: "Allocate budget for different marketing channels.",
      metadata: {
        "creation_timestamp": "2023-10-01T12:10:00Z",
        "llm_used": "Ollama"
      }
    }
  ]
};

// Function to generate nodes and edges from JSON
const generateGraph = (data) => {
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

const AgentsVisualization = () => {
    const { nodes, edges } = generateGraph(agentData);
    const [selectedAgent, setSelectedAgent] = useState(null);
  
    const onNodeClick = useCallback((event, node) => {
      const agent = agentData.agents.find(a => a.agent_id === node.id);
      setSelectedAgent(agent);
    }, []);
  
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