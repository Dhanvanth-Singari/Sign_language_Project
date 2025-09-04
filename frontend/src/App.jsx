import { useState } from "react";
import Sidebar from "./components/Sidebar";
import Home from "./components/Home";
import Profile from "./components/Profile";
import About from "./components/About";
import TextToGesture from "./components/TextToGesture";
import GestureToText from "./components/GestureToText";

export default function App() {
  const [page, setPage] = useState("Home");

  const renderPage = () => {
    switch (page) {
      case "Home": return <Home />;
      case "Profile": return <Profile />;
      case "About": return <About />;
      case "TextToGesture": return <TextToGesture />;
      case "GestureToText": return <GestureToText />;
      default: return <Home />;
    }
  };

  return (
    <div className="layout">
      <Sidebar setPage={setPage} />
      <main>{renderPage()}</main>
    </div>
  );
}
