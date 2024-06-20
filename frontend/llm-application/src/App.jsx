import React, { useState, useEffect } from "react";
import axios from "axios";
import PdfUpload from "./components/PdfUpload";
import Question from "./components/Question";
import Response from "./components/Response";
import PdfList from "./components/PdfList";
import TextArea from "./components/TextArea";
import ChatBoxResponse from "./components/ChatBoxResponse";

function App() {
  const [pdfFile, setPdfFile] = useState(null);
  const [userInput, setUserInput] = useState("");
  const [response, setResponse] = useState("");
  const [sourceDocuments, setSourceDocuments] = useState([]);
  const [inputTokens, setInputTokens] = useState("");
  const [outputTokens, setOutputTokens] = useState("");
  const [pdfs, setPdfs] = useState([]);
  const [selectedPdf, setSelectedPdf] = useState(null);
  const [gptInput, setGptInput] = useState("");
  const [temperature, setTemperature] = useState(0.5);
  const [gptResponse, setGptResponse] = useState("");
  const [textResponse, setTextResponse] = useState("");
  const [moreInformation, setMoreInformation] = useState(false);
  const [placeholder, setPlaceholder] = useState([]);
  const [additionalInfo, setAdditionalInfo] = useState("");
  const [combinedResponse, setCombinedResponse] = useState("");
  const [sentiment, setSentiment] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [amb, setAmb] = useState("");
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    handleRetreivePdf();
  }, []);
  const handleAdditionalInfo = (e) => {
    setAdditionalInfo(e.target.value);
  };

  const handleAdditional = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(
        "https://6f3f-69-119-209-231.ngrok-free.app/api/info",
        {
          input: textResponse.answer,
          additionalInfo: additionalInfo,
        }
      );
      setMoreInformation(false);
      console.log("Response:", response.data);
      setCombinedResponse({ answer: response.data });
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const handleFileUpload = async (file) => {
    setPdfFile(file);

    const formData = new FormData();
    formData.append("file", file);

    try {
      await axios.post(
        "https://6f3f-69-119-209-231.ngrok-free.app/upload",
        formData
      );
    } catch (error) {
      console.error("Error uploading PDF:", error);
    }
  };

  const handleAskQuestion = async () => {
    try {
      const response = await axios.post(
        "https://6f3f-69-119-209-231.ngrok-free.app/ask",
        {
          question: userInput,
        }
      );
      setResponse(response.data.answer);
      setSourceDocuments(response.data.source_documents_info);
      setInputTokens(response.data.input_tokens);
      setOutputTokens(response.data.output_tokens);
    } catch (error) {
      console.error("Error asking question:", error);
    }
  };

  const handleRetreivePdf = async () => {
    try {
      const response = await axios.get(
        "https://6f3f-69-119-209-231.ngrok-free.app/pdfs",
        {
          headers: {
            "ngrok-skip-browser-warning": "true",
          },
        }
      );
      console.log(response.data);
      setPdfs(response.data);
    } catch (error) {
      console.error("Error retrieving PDF:", error);
    }
  };

  const handleSelectPdf = async (pdf) => {
    setSelectedPdf(pdf);
    console.log(selectedPdf);
    const fakeFile = new File([""], pdf, {
      type: "application/pdf",
    });

    handleFileUpload(fakeFile);
  };

  const handleGptQuery = async () => {
    try {
      const response = await axios.post(
        "https://6f3f-69-119-209-231.ngrok-free.app/api/gpt",
        {
          input: gptInput,
          temperature: temperature,
        }
      );
      setGptResponse({ question: gptInput, answer: response.data.answer });
      console.log(response.data);
    } catch (error) {
      console.error("Error chatting directly with GPT:", error);
    }
  };

  const handleTextAreaSubmit = async (
    textAreaValue,
    option,
    language,
    tone
  ) => {
    try {
      const response = await axios.post(
        "https://6f3f-69-119-209-231.ngrok-free.app/api/bot",
        {
          input: textAreaValue,
          option: option,
          language: language,
          tone: tone,
        }
      );
      console.log("Response:", response.data);
      if (response.data.placeholders) {
        setMoreInformation(true);
        setPlaceholder(response.data.placeholders);
        console.log("Placeholders:", response.data.placeholders);
        setTextResponse({ answer: response.data.transformation });
      } else if (response.data.response) {
        setSentiment({ answer: response.data.sentiment });
        setTextResponse({ answer: response.data.response });
      } else {
        setTextResponse({ answer: response.data });
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const handleTextQuery = async () => {
    try {
      const response = await axios.post(
        "https://6f3f-69-119-209-231.ngrok-free.app/api/ambiguous",
        {
          input: chatInput === "" ? "reset" : chatInput,
        }
      );
      if (chatInput === "") {
        setMessages([]);
      } else {
        const updatedMessages = [
          ...response.data.map((msg) => ({
            type: msg.type,
            content: msg.content,
          })),
        ];
        setMessages(updatedMessages);
      }
      setChatInput("");
      console.log("Response:", response.data);
      setAmb({ answer: response.data });
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div>
      <PdfList pdfs={pdfs} onPdfSelect={handleSelectPdf} />
      <PdfUpload onFileUpload={handleFileUpload} />
      {/* PDF Question */}
      <Question
        value={userInput}
        onChange={(value) => setUserInput(value)}
        onSubmit={handleAskQuestion}
        placeholder={"Ask about the PDF"}
      />
      <Response
        response={response}
        sourceDocuments={sourceDocuments}
        inputTokens={inputTokens}
        outputTokens={outputTokens}
      />
      {/* GPT Question */}
      <Question
        value={gptInput}
        onChange={(value) => setGptInput(value)}
        onSubmit={handleGptQuery}
        placeholder={"Ask question for GPT"}
        showTemperature={true}
        temperature={temperature}
        onTemperatureChange={(value) => setTemperature(value)}
      />
      <Response response={gptResponse} />
      <TextArea
        placeholder={"Your text here..."}
        onSubmit={handleTextAreaSubmit}
      />
      <Response response={sentiment} />
      <Response response={textResponse} />
      {moreInformation && (
        <>
          <p>
            please provide more information in the following order separate by
            comma
          </p>
          {placeholder.map((ph) => (
            <p key={ph}>{ph}</p>
          ))}
          <form onSubmit={handleAdditional}>
            <input
              type="text"
              value={additionalInfo}
              onChange={handleAdditionalInfo}
            />
            <button type="submit">submit</button>
          </form>
        </>
      )}
      <Response response={combinedResponse} />
      <Question
        value={chatInput}
        onChange={(value) => setChatInput(value)}
        onSubmit={handleTextQuery}
        placeholder={"ambiguous text"}
      />
      <button onClick={handleTextQuery}>Clear</button>
      {/* <Response response={amb} /> */}
      <ChatBoxResponse messages={messages} />
    </div>
  );
}

export default App;
