import React from "react";

export interface Agent {
  agent_id: string;
  parent_id: string | null;
  related_agents: string[];
  role_name: string;
  system_prompt: string;
  task_prompt: string;
}

export interface NodeData {
  id: string;
  position: { x: number; y: number };
  data: {
    label: React.ReactNode;
  };
}

export interface EdgeData {
  id: string;
  source: string;
  target: string;
}
