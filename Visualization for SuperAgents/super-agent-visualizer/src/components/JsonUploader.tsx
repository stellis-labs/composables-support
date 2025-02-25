import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import styles from "../style.module.css"

interface JsonUploaderProps {
  onUpload: (data: any) => void;
  updatedData: any;
}


const JsonUploader: React.FC<JsonUploaderProps> = ({ onUpload, updatedData }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    const reader = new FileReader();
    
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target?.result as string);
        onUpload(json);
      } catch (error) {
        console.error("Invalid JSON file:", error);
      }
    };

    reader.readAsText(file);
  }, [onUpload]);

  const { getRootProps, getInputProps } = useDropzone({ onDrop, accept: { 'application/json': ['.json'] } });

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(updatedData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "updated_agents.json";
    a.click();
  };
  

  return (
    <div className={`flex ${updatedData ? 'flex-row' : 'flex-col'} gap-6 items-center`}>
      <div {...getRootProps()} className={`${updatedData ? 'px-6 py-2' : 'px-12 py-6'} text-center ${styles.upload_border} !cursor-pointer`}>
        <input {...getInputProps()} />
        <p>Drag files here or <span className="text-blue-600">browse</span></p>
      </div>
      <button onClick={handleExport} className={`${updatedData ? '' : 'hidden'} px-6 py-2 bg-blue-600 text-white rounded-[5px]`}>Export JSON</button>
    </div>
  );
};

export default JsonUploader;
