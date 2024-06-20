import React, { useState } from "react";

function TextArea({ onSubmit, placeholder }) {
  const [textAreaValue, setTextAreaValue] = useState("");
  const [option, setOption] = useState("1");
  const [language, setLanguage] = useState("en");
  const [tone, setTone] = useState("formal");

  const handleTextAreaChange = (e) => {
    setTextAreaValue(e.target.value);
  };

  const handleOptionChange = (e) => {
    setOption(e.target.value);
  };

  const handleFormSubmit = (e) => {
    e.preventDefault();
    onSubmit(textAreaValue, option, language, tone);
    setTextAreaValue("");
  };

  const handleLanguageChange = (e) => {
    setLanguage(e.target.value);
  };

  const handleTone = (e) => {
    setTone(e.target.value);
  };

  return (
    <form onSubmit={handleFormSubmit}>
      <textarea
        value={textAreaValue}
        onChange={handleTextAreaChange}
        placeholder={placeholder}
      />
      <select value={option} onChange={handleOptionChange}>
        <option value="1">Translate</option>
        <option value="2">Summarize</option>
        <option value="3">Transformation to email</option>
        <option value="4">Grammar correction</option>
        <option value="5">Sentiment</option>
      </select>
      {option === "1" && (
        <input
          type="text"
          placeholder="Language"
          onChange={handleLanguageChange}
        />
      )}
      {option === "3" && (
        <input type="text" placeholder="Tone" onChange={handleTone} />
      )}
      <button type="submit">Submit</button>
    </form>
  );
}

export default TextArea;
