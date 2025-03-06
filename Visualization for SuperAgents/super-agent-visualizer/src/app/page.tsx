'use client';
import JsonUploader from "@/components/JsonUploader";
import AgentGraph from "@/components/AgentGraph";
import { useState } from "react";
import { Agent } from "@/types";
import styles from "../style.module.css";

export default function Home() {
  const [jsonData, setJsonData] = useState<{ agents: Agent[] }>();
  const [updatedData, setUpdatedData] = useState<{ agents: Agent[] }>();

  return (
    <div className={`relative flex flex-col h-screen justify-center items-center ${styles.background}`}>
      {/* Sidebar */}
      <div className={`flex flex-col gap-6 ${jsonData ? 'w-[400px] absolute top-[50px] right-[50px] p-4' : `w-1/3 p-12 ${styles.shadow}`} items-center bg-white text-[#3D3D3D] rounded-[10px] z-10`}>
        <div className={`flex flex-col gap-1 items-center`}>
          <h1 className="text-xl font-bold">AI Agent Visualizer</h1>
          {jsonData ? <p className="text-xs text-center">Change JSON file or Export current JSON file</p> : <p className="text-center">Upload a JSON file to visualize</p>}
        </div>
        <JsonUploader onUpload={setJsonData} updatedData={updatedData} />
      </div>

      {/* Graph Visualization */}
      {jsonData ?
        <div className={`w-full`}>
          <AgentGraph data={jsonData} onUpdatedData={setUpdatedData} />
        </div> :
        <div className="hidden"></div>
      }
    </div>
  );
}
