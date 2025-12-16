import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { GlobalUserProvider } from "./contexts/AuthContext";
import { Navbar } from "./components/Navbar";
import { Soluna } from "./pages/Soluna";
import { Recommend } from "./pages/Recommend";
import { Synergy } from "./pages/Synergy";
import { Timeline } from "./pages/Timeline";
import { DropPredict } from "./pages/DropPredict";
import { Recap } from "./pages/Recap";
import { QuickIDSettings } from "./pages/QuickIDSettings";

function App() {
  return (
    <GlobalUserProvider>
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
              <Route path="/recap" element={<Recap />} />
              <Route path="/settings/quick-ids" element={<QuickIDSettings />} />
            </Routes>
          </main>
        </div>
      </Router>
    </GlobalUserProvider>
  );
}

export default App;
