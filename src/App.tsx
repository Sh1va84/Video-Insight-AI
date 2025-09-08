import { useState, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import { createSupabaseClient } from "./api/api";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const App = () => {
  const [url, setUrl] = useState("");
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [conversationId, setConversationId] = useState("");
  const [documentIds, setDocumentIds] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [thumbnailUrl, setThumbnailUrl] = useState<string | null>(null);

  // Extract YouTube video ID and thumbnail URL
  const extractVideoId = (url: string): string | null => {
    const regExp =
      /(?:https?:\/\/(?:www\.)?youtube\.com(?:\/(?:[^\/\n\s]+\/\S+\/?|(?:\S*\?v=|\S*\/\S+\/?v=))(\w+))|(?:youtu\.be\/(\w+)))/;
    const match = url.match(regExp);
    return match ? match[1] || match[2] : null;
  };

  useEffect(() => {
    if (url) {
      const videoId = extractVideoId(url);
      if (videoId) {
        setThumbnailUrl(`https://img.youtube.com/vi/${videoId}/maxresdefault.jpg`);
      }
    }
  }, [url]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const convId = uuidv4();
      const docId = uuidv4();

      // Create conversation and document entries
      const supabase = createSupabaseClient();
      await supabase.from("conversations").insert({ id: convId });
      await supabase.from("documents").insert({ id: docId });
      await supabase.from("conversation_documents").insert({
        conversation_id: convId,
        document_id: docId,
      });

      // Store document (simulate storing URL)
      await fetch("http://localhost:8000/store-document", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, documentId: docId }),
      });

      setConversationId(convId);
      setDocumentIds([docId]);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleSendPrompt = async (e: React.FormEvent) => {
    e.preventDefault();

    const userMessage: Message = { content: prompt, role: "user" };
    const assistantMessage: Message = { content: "", role: "assistant" };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setStreaming(true);
    setPrompt("");

    try {
      const response = await fetch("http://localhost:8000/query-document", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: prompt,
          conversationId,
          documentIds,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch response");
      }

      const data = await response.json();
      const assistantMessage: Message = { content: data, role: "assistant" };

      setMessages((prev) => [...prev.slice(0, -1), assistantMessage]);
    } catch (error) {
      console.error("Error fetching response:", error);
      const assistantMessage: Message = { content: "Sorry, something went wrong.", role: "assistant" };
      setMessages((prev) => [...prev.slice(0, -1), assistantMessage]);
    } finally {
      setStreaming(false);
    }
  };

  if (conversationId) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 px-6 py-10 flex flex-col items-center text-white">
        <h2 className="text-3xl font-bold mb-2">Chat with Any YouTube Video</h2>
        <p className="text-sm text-gray-400 mb-6">Video URL: <span className="underline text-blue-400">{url}</span></p>

        {thumbnailUrl && (
          <div className="w-full max-w-3xl mb-4">
            <iframe
              width="700"
              height="400"
              src={`https://www.youtube.com/embed/${extractVideoId(url)}`}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              className="mt-4 rounded-lg shadow-lg object-contain"
            ></iframe>
          </div>
        )}

        <div className="w-full max-w-3xl space-y-4 mb-10 overflow-y-auto max-h-[60vh] scrollbar-thin scrollbar-thumb-blue-500 pr-2">
          {messages.map((message, index) => (
            <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`rounded-2xl px-5 py-3 max-w-[70%] text-sm shadow-md ${
                  message.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-100'
                }`}>
                {message.content || "..."}
              </div>
            </div>
          ))}
        </div>

        <form onSubmit={handleSendPrompt} className="w-full max-w-2xl flex gap-3 items-center">
          <input
            type="text"
            placeholder="Ask a question about the video..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="flex-1 rounded-full px-5 py-3 text-white focus:outline-none focus:ring-2 focus:ring-white border border-white"
          />
          <button
            type="submit"
            disabled={streaming || !prompt.trim()}
            className={`px-5 py-3 rounded-full font-medium ${
              streaming || !prompt.trim() ? 'bg-gray-500 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 cursor-pointer'
            }`}>
            {streaming ? 'Processing...' : 'Send'}
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 flex flex-col items-center justify-center text-white">
      <div className="bg-gray-800 shadow-lg rounded-lg p-8 w-full max-w-md">
        <h1 className="text-4xl font-extrabold text-center mb-6 text-indigo-400">
          AI Chat with YouTube
        </h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="text"
            placeholder="Drop a YouTube URL here..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="w-full px-4 py-2 rounded-lg bg-gray-700 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button
            type="submit"
            disabled={loading}
            className={`w-full py-2 rounded-lg font-semibold text-white cursor-pointer ${loading
              ? "bg-indigo-300 cursor-not-allowed"
              : "bg-indigo-500 hover:bg-indigo-600"
            }`}>
            {loading ? "Processing..." : "Submit"}
          </button>
        </form>
        {loading && (
          <div className="mt-4 flex justify-center">
            <div className="loading-spinner border-t-4 border-indigo-500 rounded-full w-8 h-8 animate-spin"></div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
