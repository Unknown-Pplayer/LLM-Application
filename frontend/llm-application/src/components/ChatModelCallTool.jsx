import React, { useState } from "react";
import axios from "axios";

const ChatModelCallTool = () => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const handleInputChange = (event) => {
    setQuery(event.target.value);
  };

  const handleChatModelCall = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post(
        "https://7c69-69-119-209-231.ngrok-free.app/chatModelCallTool",
        { question: query }
      );
      console.log("Response:", response.data);
      if (response.data.tool_calls[0]) {
        setResponse(response.data.tool_calls[0]);
      } else {
        setResponse(response.data.content);
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const renderResponse = () => {
    if (!response) return null;

    // if (response.name === "Joke") {
    //   // Render Joke
    //   return (
    //     <div>
    //       <h3>Joke</h3>
    //       <p>Setup: {response.args.setup}</p>
    //       <p>Punchline: {response.args.punchline}</p>
    //       <p>Rating: {response.args.rating}</p>
    //     </div>
    //   );
    // } else if (response.name === "ConversationalResponse") {
    //   // Render ConversationalResponse
    //   return (
    //     <div>
    //       <h3>Response</h3>
    //       <p>{response.args.response}</p>
    //     </div>
    //   );
    // } else if (response.name === "AddEvent") {
    //   // Render AddEvent
    //   return (
    //     <div>
    //       <h3>Event</h3>
    //       <p>Name: {response.args.event_name}</p>
    //       <p>Date: {response.args.event_date}</p>
    //       <p>Time: {response.args.event_time}</p>
    //       <p>Location: {response.args.event_location}</p>
    //     </div>
    //   );
    // } else if (response.name === "SendingEmail") {
    //   // Render SendingEmail
    //   return (
    //     <div>
    //       <h3>Email</h3>
    //       <p>To: {response.args.recipient}</p>
    //       <p>Subject: {response.args.subject}</p>
    //       <p>Body: {response.args.body}</p>
    //     </div>
    //   );
    // } else if (response.name === "CodeDivision") {
    //   return (
    //     <div>
    //       <h3>CodeDivision</h3>
    //       <p>Code: {response.args.code}</p>
    //       <p>Part: {response.args.part}</p>
    //     </div>
    //   );
    // } else {
    //   return <p>{response}</p>;
    // }
  };
  const formatJSON = (obj) => {
    return JSON.stringify(obj, null, 2);
  };

  return (
    <div>
      <p>Chat Model to Call Tool</p>
      <form onSubmit={handleChatModelCall}>
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          placeholder="Query"
        />
        <button type="submit">Submit</button>
      </form>
      {formatJSON(response)}
    </div>
  );
};

export default ChatModelCallTool;
