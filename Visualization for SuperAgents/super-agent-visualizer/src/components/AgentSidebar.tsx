import React, { useState, useEffect } from "react";
import { Node } from "reactflow";
import styles from "../style.module.css";

interface AgentSidebarProps {
  node?: Node | null;
  updateNodeData: (roleName: string, systemPrompt: string, taskPrompt: string) => void;
}

const AgentSidebar: React.FC<AgentSidebarProps> = ({ node, updateNodeData }) => {
  const [roleName, setRoleName] = useState<string>("");
  const [systemPrompt, setSystemPrompt] = useState<string>("");
  const [taskPrompt, setTaskPrompt] = useState<string>("");

  useEffect(() => {
    if (node) {
      setRoleName(node.data.role_name);
      setSystemPrompt(node.data.system_prompt);
      setTaskPrompt(node.data.task_prompt);
    }
  }, [node]); // This will run only when `node` changes

  useEffect(() => {
    if (node) {
      updateNodeData(roleName, systemPrompt, taskPrompt);
    }
  }, [roleName, systemPrompt, taskPrompt, node, updateNodeData]);

  if (!node) return <div className="w-1/4 p-4 border-l">Select an agent to edit</div>;

  return (
    <div className="w-full flex flex-col gap-4 items-center">
      <h1 className="text-xl font-bold mb-2">Edit Agent</h1>
      <div className="w-full flex flex-col gap-1">
        <label>Role Name</label>
        <input
          name="role_name"
          value={roleName}
          onChange={(e) => setRoleName(e.target.value)} // Update role name
          className={`${styles.input}`}
        />
      </div>
      <div className="w-full flex flex-col gap-1">
        <label>System Prompt</label>
        <textarea
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)} // Update system prompt
          className={`${styles.input}`}
        />
      </div>
      <div className="w-full flex flex-col gap-1">
        <label>Task Prompt</label>
        <textarea
          value={taskPrompt}
          onChange={(e) => setTaskPrompt(e.target.value)} // Update task prompt
          className={`${styles.input}`}
        />
      </div>
    </div>
  );
};

export default AgentSidebar;
