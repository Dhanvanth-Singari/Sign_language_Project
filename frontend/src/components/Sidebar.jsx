export default function Sidebar({ setPage }) {
  const menu = [
    { name: "Home" },
    { name: "Profile" },
    { name: "About" },
    { name: "TextToGesture", label: "Text → Gesture" },
    { name: "GestureToText", label: "Gesture → Text" },
  ];

  return (
    <aside className="w-60 bg-gray-900 text-white p-4">
      <h1 className="text-2xl font-bold mb-6">SignLang</h1>
      <ul className="space-y-2">
        {menu.map((item) => (
          <li key={item.name}>
            <button
              onClick={() => setPage(item.name)}
              className="w-full text-left px-3 py-2 rounded hover:bg-gray-700"
            >
              {item.label || item.name}
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}
