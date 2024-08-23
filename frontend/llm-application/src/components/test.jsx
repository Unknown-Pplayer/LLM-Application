import React, { useState } from 'react';
import axios from 'axios';

const Test = () => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
        const response = await axios.post(
          "https://13b3-69-119-209-231.ngrok-free.app/test",
          {
            question: question,
          }
        );
        const result = await response.data;
        setResponse(result);
        console.log("Response:", result);
    } catch (err) {
      console.error("Failed to fetch data:", err);
    }
  };

  return (
    <div>
      <h1>Test API</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Question:
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            required
          />
        </label>
        <button type="submit">Submit</button>
      </form>
      {response && (
        <div>
          <h2>Response:</h2>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default Test;