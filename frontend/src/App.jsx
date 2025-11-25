import { useState, useRef, useEffect } from "react";
import { Send, Plus } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";


const fileToBase64 = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);

    reader.onload = () => {
      const base64 = reader.result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = reject;
  });

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [imageBase64, setImageBase64] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const messagesEndRef = useRef(null);


  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);


  useEffect(() => {
    const handlePaste = async (e) => {
      const item = [...e.clipboardData.items].find((i) =>
        i.type.startsWith("image/")
      );
      if (item) {
        const file = item.getAsFile();
        const b64 = await fileToBase64(file);
        setImageBase64(b64);
        setImagePreview(URL.createObjectURL(file));
      }
    };

    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, []);


  const sendMessage = async () => {
    if (!input.trim() && !imageBase64) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", text: input || "", image: imagePreview },
    ]);

    const payload = {
      text: input || null,
      image_base64: imageBase64 || null,
      session_id: "sap-session-1",
    };

    setInput("");
    setImageBase64(null);
    setImagePreview(null);
    setIsThinking(true);

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer, image: null },
      ]);
    } catch (err) {
      console.error("Backend error:", err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Error contacting backend.", image: null },
      ]);
    } finally {
      setIsThinking(false);
    }
  };


  const handleFile = async (file) => {
    if (file && file.type.startsWith("image/")) {
      const b64 = await fileToBase64(file);
      setImageBase64(b64);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  return (
    <div className="h-screen bg-gradient-to-br from-[#0b0b0f] via-[#141421] to-[#1e1e2e] text-white px-4 pt-16 py-12 flex flex-col items-center overflow-x-hidden">

      <motion.div
        initial={{ opacity: 0, y: -25 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="mb-15 text-center"
      >
        <h1 className="text-7xl font-extrabold tracking-wide drop-shadow-lg bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
          SAP
        </h1>

        <h2 className="text-4xl mt-5 font-semibold bg-gradient-to-r from-purple-300 to-pink-300 bg-clip-text text-transparent opacity-90 pb-2">
          A Localized Ecological Assistant
        </h2>
      </motion.div>

      <div className="w-full max-w-4xl h-[70vh] mx-auto bg-black/20 backdrop-blur-xl border border-white/15 rounded-2xl shadow-xl flex flex-col overflow-x-hidden">

        <div className="flex-1 overflow-y-auto p-5 space-y-5 flex flex-col custom-scroll">
          {messages.map((msg, i) => (
            <MessageBubble
              key={i}
              role={msg.role}
              text={msg.text}
              image={msg.image}
            />
          ))}

          <AnimatePresence>{isThinking && <ThinkingBubble />}</AnimatePresence>

          <div ref={messagesEndRef} />
        </div>

        <div className="flex items-center gap-3 p-4 bg-black/30 border-t border-white/10 backdrop-blur-xl relative">

          <input
            type="file"
            id="filePicker"
            accept="image/*"
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />

          <motion.button
            whileTap={{ scale: 0.9 }}
            onClick={() => document.getElementById("filePicker").click()}
            className="p-3 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 hover:opacity-90 transition shadow-md"
          >
            <Plus size={20} />
          </motion.button>

          <div className="flex-1 bg-gradient-to-r from-purple-500/60 to-pink-500/60 p-[1.5px] rounded-xl">
            <input
              type="text"
              placeholder="Chat..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              className="w-full px-4 py-2 rounded-xl bg-black/40 text-white placeholder-white/40 focus:outline-none"
            />
          </div>

          <motion.button
            whileTap={{ scale: 0.9 }}
            onClick={sendMessage}
            className="p-3 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 hover:opacity-90 transition shadow-lg"
          >
            <Send size={20} />
          </motion.button>
        </div>

        {imagePreview && (
          <div className="absolute bottom-24 left-1/2 transform -translate-x-1/2 bg-black/40 border border-white/20 p-3 rounded-xl backdrop-blur-xl shadow-lg">
            <div className="relative">
              <img
                src={imagePreview}
                className="h-24 w-auto rounded-lg border border-white/20"
              />
              <button
                onClick={() => {
                  setImageBase64(null);
                  setImagePreview(null);
                }}
                className="absolute -top-2 -right-2 bg-red-600 text-xs px-2 py-1 rounded-full"
              >
                ✕
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


function MessageBubble({ role, text, image }) {
  const isUser = role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`max-w-[55%] rounded-2xl px-[2px] py-[2px] shadow-xl ${
        isUser
          ? "self-end bg-gradient-to-r from-purple-500 to-pink-500"
          : "bg-gradient-to-r from-purple-500/40 to-pink-500/40"
      }`}
    >
      <div
        className={`rounded-2xl px-4 py-3 backdrop-blur-xl ${
          isUser
            ? "bg-black/40 text-white"
            : "bg-white/10 text-white border border-white/10"
        }`}
      >
        {image && (
          <img
            src={image}
            className="w-48 rounded-xl mb-2 border border-white/20"
          />
        )}
        {text}
      </div>
    </motion.div>
  );
}


function ThinkingBubble() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="bg-gradient-to-r from-purple-500/40 to-pink-500/40 p-[2px] rounded-2xl w-20"
    >
      <div className="flex items-center space-x-2 bg-black/40 px-4 py-3 rounded-2xl backdrop-blur-xl">
        <div className="w-2 h-2 bg-white/60 rounded-full animate-bounce" />
        <div className="w-2 h-2 bg-white/60 rounded-full animate-bounce delay-150" />
        <div className="w-2 h-2 bg-white/60 rounded-full animate-bounce delay-300" />
      </div>
    </motion.div>
  );
}
