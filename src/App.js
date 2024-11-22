// src/components/ChatInterface.js
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
//import test from './images/1.gif'

const ChatInterface = () => {
  const [messages, setMessages] = useState([{
    text: "Hello! How can I help you today?",
    sender: 'api',
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage = {
      text: input,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      });
      
      const data = await response.json();
      
      // Add API response
      const apiMessage = {
        text: data.response,
        sender: 'api',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        image: data.image,
        url: data.url
      };
      setMessages(prev => [...prev, apiMessage]);
    } catch (error) {
        console.error('Error:', error);
        // Add error message
        const errorMessage = {
          text: 'Error generating response.',
          sender: 'api',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        setMessages(prev => [...prev, errorMessage]);
    } finally {
        setIsLoading(false);
    }
    };

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto bg-gray-100">
      {/* Chat header */}
      <div className="bg-blue-800 text-white p-4 shadow-md">
        <h1 className="text-xl font-semibold">Retail store playground</h1>
      </div>

      {/* Messages container */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[70%] rounded-lg p-3 ${
                message.sender === 'user'
                  ? 'bg-blue-700 text-white rounded-br-none'
                  : 'bg-white text-gray-800 rounded-bl-none'
              } shadow`}
            >
             <ReactMarkdown
                className="break-words"
                //linkTarget="_blank"
                components={{
                  a: ({ node, ...props }) => (
                    <a
                      {...props}
                      className="text-blue-600 font-semibold underline hover:text-blue-800"
                    />
                  ),
                }}
              >
                {message.text}
              </ReactMarkdown>
              {/*<p className="break-words">{message.text}</p>*/}
              {message.image && (
                <a href={message.url} target="_blank" rel="noopener noreferrer">
                  <img
                    src={message.image}
                    alt=""
                    className="mt-2 w-48 h-auto rounded-lg transition-transform duration-300 ease-in-out transform hover:scale-105"
                  />
                </a>
                )
              }
              <p className={`text-xs mt-1 ${
                message.sender === 'user' ? 'text-blue-100' : 'text-gray-500'
              }`}>
                {message.timestamp}
              </p>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white text-gray-800 rounded-lg rounded-bl-none p-3 shadow">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="p-4 bg-white shadow-lg">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message..."
            className="flex-1 p-2 border border-gray-300 rounded-full focus:outline-none focus:border-blue-500"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-blue-500 text-white px-4 py-2 rounded-full hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;