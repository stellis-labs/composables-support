import React, { useState, useEffect, useCallback } from "react";
import ReactFlow, {
  Controls,
  Background,
  Node,
  Edge,
  applyNodeChanges,
  applyEdgeChanges,
} from "reactflow";
import "reactflow/dist/style.css";
import { Agent } from "@/types"; // Your Agent interface
import AgentSidebar from "@/components/AgentSidebar";
import styles from "../style.module.css";
import "@/app/globals.css";

import nodeLayout from "@/utils/nodeLayout"; // Updated nodeLayout import

interface Props {
  data: { agents: Agent[] };
  onUpdatedData: (data: { agents: Agent[] }) => void;
}

const AgentGraph: React.FC<Props> = ({ data, onUpdatedData }) => {
  const defaultViewport = { x: 0, y: 0, zoom: 1.5 };
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [updatedData, setUpdatedData] = useState<{ agents: Agent[] }>({
    agents: data.agents,
  });

  useEffect(() => {
    if (!updatedData?.agents) return;

    const newNodes: Node[] = updatedData.agents.map((agent) => ({
      id: agent.agent_id,
      data: {
        label: (
          <div className={`${styles.node}`}>
            <div className={`${styles.node_id}`}>ID: {agent.agent_id}</div>
            <div className={`${styles.node_head}`}>{agent.role_name}</div>
            <div className={`${styles.node_body}`}>
              <div>System: {agent.system_prompt}</div>
              <div>Task: {agent.task_prompt}</div>
            </div>
          </div>
        ),
        role_name: agent.role_name,
        system_prompt: agent.system_prompt,
        task_prompt: agent.task_prompt,
      },
      position: { x: 0, y: 0 }, // Temporary placeholder, will be updated
    }));

    const newEdges: Edge[] = updatedData.agents.flatMap((agent) =>
      agent.parent_id
        ? [
            {
              id: `${agent.parent_id}-${agent.agent_id}`,
              source: agent.parent_id,
              target: agent.agent_id,
              type: 'smoothstep',
              animated: true,
            },
          ]
        : []
    );

    // Asynchronous layout calculation inside useEffect
    nodeLayout(newNodes, newEdges).then((layoutedNodes) => {
      setNodes(layoutedNodes);
    });

    setEdges(newEdges);
    onUpdatedData(updatedData);
  }, [updatedData, onUpdatedData]);

  const onNodesChange = useCallback(
    (changes: any) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  const onEdgesChange = useCallback(
    (changes: any) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  const updateNodeData = useCallback(
    (updatedRoleName: string, updatedSystemPrompt: string, updatedTaskPrompt: string) => {
      if (selectedNode) {
        setNodes((nds) =>
          nds.map((node) =>
            node.id === selectedNode.id
              ? {
                  ...node,
                  data: {
                    ...node.data,
                    role_name: updatedRoleName,
                    system_prompt: updatedSystemPrompt,
                    task_prompt: updatedTaskPrompt,
                    label: (
                      <div className={`${styles.nodes}`}>
                        <strong>{updatedRoleName}</strong>
                        <small>ID: {selectedNode.id}</small>
                        <em>System: {updatedSystemPrompt}</em>
                        <em>Task: {updatedTaskPrompt}</em>
                      </div>
                    ),
                  },
                }
              : node
          )
        );

        setUpdatedData((prevData) => ({
          agents: prevData.agents.map((agent) =>
            agent.agent_id === selectedNode.id
              ? {
                  ...agent,
                  role_name: updatedRoleName,
                  system_prompt: updatedSystemPrompt,
                  task_prompt: updatedTaskPrompt,
                }
              : agent
          ),
        }));
      }
    },
    [selectedNode]
  );

  return (
    <div className="relative flex h-screen items-center justify-center">
      <div className="w-full h-full">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={(_, node) => setSelectedNode(node)}
          defaultViewport={defaultViewport}
          minZoom={0.2}
          maxZoom={4}
          attributionPosition="bottom-left"
          fitView
          fitViewOptions={{ padding: 0.5 }}
        >
          <Controls />
          <Background />
        </ReactFlow>
      </div>
      <div
        className={`absolute top-[250px] right-[50px] w-[400px] ${selectedNode ? "" : "hidden"} p-4 items-center bg-white text-[#3D3D3D] rounded-[10px] z-10`}
      >
        <AgentSidebar node={selectedNode} updateNodeData={updateNodeData} />
      </div>
    </div>
  );
};

export default AgentGraph;
