import React from "react";
import "./ChatBoxResponse.css";

const ChatBoxResponse = ({ messages }) => {
  return (
    <div className="chat-container">
      {messages.map((message, index) => (
        <div key={index} className={`message-row ${message.type}`}>
          <div className={`message ${message.type}`}>{message.content}</div>
        </div>
      ))}
    </div>
  );
};

export default ChatBoxResponse;
