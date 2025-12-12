import React from "react";
import ReactDOM from "react-dom/client";
import { Recommend } from "./pages/Recommend";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <Recommend />
    </div>
  </React.StrictMode>,
);
