import React from 'react';

// Neo4j Browser embedded via iframe — shows KG structure visually during demo
const NEO4J_BROWSER_URL = 'http://localhost:7474/browser/';

export function KGViewer() {
  return (
    <div className="h-full flex flex-col">
      <h3 className="font-semibold text-gray-700 px-4 py-2 text-sm uppercase tracking-wide border-b">
        Knowledge Graph
      </h3>
      <iframe
        src={NEO4J_BROWSER_URL}
        className="flex-1 w-full border-0"
        title="Neo4j Browser"
        sandbox="allow-scripts allow-same-origin allow-forms"
      />
    </div>
  );
}
