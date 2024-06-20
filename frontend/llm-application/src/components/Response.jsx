import React from "react";

const Response = ({ response, sourceDocuments, inputTokens, outputTokens }) => {
  return (
    <div>
      {response && (
        <div>
          {response.question && (
            <p>
              <strong>Question:</strong> {response.question}
            </p>
          )}
          {response.answer && (
            <p>
              <strong>Answer:</strong> {response.answer}
            </p>
          )}
        </div>
      )}
      {sourceDocuments && sourceDocuments.length > 0 && (
        <div>
          <p>
            <strong>Source Documents:</strong>
          </p>
          <ul>
            {sourceDocuments.map((doc, index) => (
              <li key={index}>
                <p>
                  <strong>Page {doc.page}:</strong> {doc.quote}
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}
      {inputTokens && (
        <div>
          <p>
            <strong>Input Tokens:</strong> {inputTokens}
          </p>
        </div>
      )}
      {outputTokens && (
        <div>
          <p>
            <strong>Output Tokens:</strong> {outputTokens}
          </p>
        </div>
      )}
    </div>
  );
};

export default Response;
