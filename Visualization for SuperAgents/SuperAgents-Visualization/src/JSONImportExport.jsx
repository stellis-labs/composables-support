import React, { useState } from "react";

const JSONImportExport = () => {
  const [jsonData, setJsonData] = useState(null);
  const [error, setError] = useState(null);

  // Handle file drop
  const handleDrop = (event) => {
    event.preventDefault();
    setError(null);
    const file = event.dataTransfer.files[0];
    processFile(file);
  };

  // Handle file selection through click
  const handleFileInputChange = (event) => {
    const file = event.target.files[0];
    processFile(file);
  };

  // Process JSON file
  const processFile = (file) => {
    if (file && file.type === "application/json") {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const data = JSON.parse(reader.result);
          setJsonData(data);
          setError(null);
        } catch (e) {
          setError("Invalid JSON file.");
        }
      };
      reader.readAsText(file);
    } else {
      setError("Please upload a valid JSON file.");
    }
  };

  // Prevent default drag behaviors
  const handleDragOver = (event) => {
    event.preventDefault();
  };

  // Export JSON data
  const handleExport = () => {
    if (!jsonData) return;

    const blob = new Blob([JSON.stringify(jsonData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "export.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">JSON Import/Export</h2>
      <div
        className="border-dashed border-2 border-gray-400 rounded-lg p-4 text-center"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <p className="text-gray-600">Drag and drop a JSON file here</p>
        <p className="text-gray-500">or</p>
        {/* File Input for Click Upload */}
        <label
          htmlFor="fileInput"
          className="text-blue-500 cursor-pointer underline"
        >
          Click to upload
        </label>
        <input
          id="fileInput"
          type="file"
          accept="application/json"
          className="hidden"
          onChange={handleFileInputChange}
        />
      </div>
      {error && <p className="text-red-500 mt-2">{error}</p>}
      {jsonData && (
        <div className="mt-4">
          <h3 className="text-lg font-bold">JSON Preview:</h3>
          <pre className="bg-gray-100 p-2 rounded-lg overflow-auto max-h-64">
            {JSON.stringify(jsonData, null, 2)}
          </pre>
          <button
            className="mt-4 bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600"
            onClick={handleExport}
          >
            Export JSON
          </button>
        </div>
      )}
    </div>
  );
};

export default JSONImportExport;
