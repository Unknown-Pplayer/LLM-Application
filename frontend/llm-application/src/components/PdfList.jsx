import React from "react";

function pdfList({ pdfs, onPdfSelect }) {
  return (
    <div>
      <h3>Available PDFs</h3>
      <ul>
        {pdfs.map((pdf, index) => (
          <li
            key={index}
            onClick={() => onPdfSelect(pdf)}
            style={{ cursor: "pointer" }}
          >
            {pdf}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default pdfList;
