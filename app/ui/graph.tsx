'use client';

import { useRef, useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { graphData as fullGraphData } from '@/app/data/graphData';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

export default function ClassGraph() {
  const fgRef = useRef();
  const [selectedGroups, setSelectedGroups] = useState(() => {
    const allGroups = new Set(fullGraphData.nodes.map(n => n.group));
    return Array.from(allGroups);
  });

  // Filter nodes
  const filteredNodes = fullGraphData.nodes.filter(n => selectedGroups.includes(n.group));
  const nodeIds = new Set(filteredNodes.map(n => n.id));

  // Normalize source/target and filter links
  const filteredLinks = fullGraphData.links.flatMap(link => {
    const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
    const targetId = typeof link.target === 'string' ? link.target : link.target.id;

    if (nodeIds.has(sourceId) && nodeIds.has(targetId)) {
      if (link.label === 'is equivalent to') {
        return [
          { ...link, source: sourceId, target: targetId },
          { ...link, source: targetId, target: sourceId }
        ];
      } else {
        return [{ ...link, source: sourceId, target: targetId }];
      }
    }
    return [];
  });

  const filteredData = {
    nodes: filteredNodes,
    links: filteredLinks,
  };

  useEffect(() => {
    if (fgRef.current) fgRef.current.zoomToFit(400);
  }, [filteredData]);

  const allGroups = Array.from(new Set(fullGraphData.nodes.map(n => n.group)));

  const getColorForGroup = (group: string) => {
    const hash = [...group].reduce((a, c) => a + c.charCodeAt(0), 0);
    return `hsl(${(hash * 23) % 360}, 60%, 60%)`;
  };

  return (
    <div>
      <div className="mb-4 text-left">
        <strong>Filter by Group:</strong>
        {allGroups.map(group => (
          <label key={group} className="ml-4">
            <input
              type="checkbox"
              checked={selectedGroups.includes(group)}
              onChange={() =>
                setSelectedGroups(prev =>
                  prev.includes(group)
                    ? prev.filter(g => g !== group)
                    : [...prev, group]
                )
              }
              className="mr-1"
            />
            {group}
          </label>
        ))}
      </div>

      <div style={{ width: '100%', height: '100%', position: 'relative' }}>
        <ForceGraph2D
          ref={fgRef}
          graphData={filteredData}
          nodeAutoColorBy="group"
          nodeLabel="id"
          linkDirectionalArrowLength={6}
          linkDirectionalArrowRelPos={0.95}
          linkCanvasObjectMode={() => 'none'}
          linkCanvasObject={() => {}}
        />

        {/* Legend */}
        <div style={{
          position: 'absolute',
          top: 10,
          right: 10,
          background: 'rgba(255, 255, 255, 0.9)',
          padding: '10px',
          borderRadius: '8px',
          fontSize: '0.85rem',
          boxShadow: '0 2px 6px rgba(0,0,0,0.2)'
        }}>
          <div><strong>Legend</strong></div>
          <div style={{ marginTop: '6px' }}>
            <div>→ <span style={{ marginLeft: 4 }}>is a subset of</span></div>
            <div>⇄ <span style={{ marginLeft: 4 }}>is equivalent to</span></div>
          </div>
          <div style={{ marginTop: '8px' }}><strong>Node Groups:</strong></div>
          {allGroups.map(group => (
            <div key={group} style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
              <span style={{
                width: 12,
                height: 12,
                backgroundColor: getColorForGroup(group),
                borderRadius: '50%'
              }} />
              <span>{group}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
