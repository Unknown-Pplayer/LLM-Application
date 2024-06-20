import React from "react";

const Question = ({
  value,
  onChange,
  onSubmit,
  placeholder,
  temperature,
  onTemperatureChange,
  showTemperature,
}) => {
  const handleInputChange = (event) => {
    onChange(event.target.value);
  };

  const handleFormSubmit = (event) => {
    event.preventDefault();
    onSubmit();
  };

  const handleTemperatureChange = (event) => {
    if (onTemperatureChange) {
      onTemperatureChange(event.target.value);
    }
  };

  return (
    <form onSubmit={handleFormSubmit}>
      <input
        type="text"
        value={value}
        onChange={handleInputChange}
        placeholder={placeholder}
      />
      {showTemperature && (
        <input
          type="number"
          value={temperature}
          step="0.1"
          min="0"
          max="1"
          onChange={handleTemperatureChange}
          placeholder="Temperature"
        />
      )}
      <button type="submit">Ask</button>
    </form>
  );
};

export default Question;
