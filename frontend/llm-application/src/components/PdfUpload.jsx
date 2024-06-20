import React from "react";

function PdfUpload({ onFileUpload }) {
  const handleChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      onFileUpload(file);
    }
  };

  return (
    <div>
      <input type="file" accept=".pdf" onChange={handleChange} />
    </div>
  );
}

export default PdfUpload;
