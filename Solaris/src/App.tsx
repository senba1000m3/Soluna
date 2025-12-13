import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Navbar } from "./components/Navbar";
import { Soluna } from "./pages/Soluna";
import { Recommend } from "./pages/Recommend";
import { Synergy } from "./pages/Synergy";
import { Timeline } from "./pages/Timeline";
import { DropPredict } from "./pages/DropPredict";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-900 text-white font-sans selection:bg-purple-500 selection:text-white">
        <Navbar />
        <main className="p-4 md:p-8">
          <Routes>
            <Route path="/" element={<Soluna />} />
            <Route path="/recommend" element={<Recommend />} />
            <Route path="/synergy" element={<Synergy />} />
            <Route path="/timeline" element={<Timeline />} />
            <Route path="/drop-predict" element={<DropPredict />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
