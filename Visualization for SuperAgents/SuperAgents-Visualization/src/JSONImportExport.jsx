import React, { useState, useContext } from "react";
import { AppContext } from "./Context";
import { FiUpload, FiDownload, FiChevronDown, FiChevronUp } from "react-icons/fi";
import "./JSONImportExport.css";

const JSONImportExport = () => {
  const { jsonData, setJsonData } = useContext(AppContext);
  const [error, setError] = useState(null);
  const [menuOpen, setMenuOpen] = useState(true);

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
    if (!file) {
      return;
    }
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
      setMenuOpen(false);
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
    <div className="dropdown-container">
      {/* Dropdown Toggle Button */}
      <button className="dropdown-toggle" onClick={() => setMenuOpen(!menuOpen)}>
        JSON Import/Export {menuOpen ? <FiChevronUp /> : <FiChevronDown />}
      </button>

      {/* Dropdown Menu */}
      {menuOpen && (
        <div className="dropdown-menu">
          {/* Drag and Drop Zone */}
          <div
            className="drop-zone"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <p>Drag & Drop a JSON file here</p>
            <p>or</p>
            <label htmlFor="fileInput" className="file-upload">
              <FiUpload /> Click to upload
            </label>
            <input
              id="fileInput"
              type="file"
              accept="application/json"
              className="hidden"
              onChange={handleFileInputChange}
            />
          </div>

          {/* Error Message */}
          {error && <p className="error-text">{error}</p>}

          {/* Export Button */}
          {jsonData && (
            <button className="export-btn" onClick={handleExport}>
              <FiDownload /> Export JSON
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default JSONImportExport;
