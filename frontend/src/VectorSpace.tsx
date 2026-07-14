import React, { useRef, useEffect, useState } from "react";
import { Sparkles, RotateCcw } from "lucide-react";
import { getRealMovieNodes } from "./webgpuEngine";

interface MovieNode {
  id: number;
  title: string;
  genre: string;
  x: number; // 3D coordinates
  y: number;
  z: number;
  px?: number; // 2D projected coordinates
  py?: number;
  pz?: number; // Depth z-value after rotation
  color: string;
}

const GENRES = ["Action", "Sci-Fi", "Drama", "Thriller", "Comedy", "Romance"];
const GENRE_COLORS = [
  "#ef4444", // Action -> Red
  "#06b6d4", // Sci-Fi -> Cyan
  "#10b981", // Drama -> Emerald
  "#a78bfa", // Thriller -> Purple
  "#f59e0b", // Comedy -> Amber
  "#ec4899", // Romance -> Pink
];

const SAMPLE_MOVIES = [
  { title: "The Dark Knight", genre: "Action" },
  { title: "Inception", genre: "Sci-Fi" },
  { title: "Interstellar", genre: "Sci-Fi" },
  { title: "Pulp Fiction", genre: "Thriller" },
  { title: "The Matrix", genre: "Sci-Fi" },
  { title: "Se7en", genre: "Thriller" },
  { title: "Gladiator", genre: "Action" },
  { title: "Avatar", genre: "Sci-Fi" },
  { title: "Superbad", genre: "Comedy" },
  { title: "La La Land", genre: "Romance" },
  { title: "Fight Club", genre: "Thriller" },
  { title: "The Departed", genre: "Thriller" },
  { title: "The Avengers", genre: "Action" },
  { title: "Forrest Gump", genre: "Drama" },
  { title: "The Godfather", genre: "Drama" },
];

export function VectorSpace() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<MovieNode[]>([]);
  const [hoveredNode, setHoveredNode] = useState<MovieNode | null>(null);
  const [selectedNode, setSelectedNode] = useState<MovieNode | null>(null);
  
  // Camera state
  const rotX = useRef(0.5);
  const rotY = useRef(0.5);
  const [autoRotate, setAutoRotate] = useState(true);

  // Mouse drag state
  const isDragging = useRef(false);
  const previousMousePosition = useRef({ x: 0, y: 0 });

  // Generate stable coordinates on mount
  useEffect(() => {
    const realNodes = getRealMovieNodes();
    if (realNodes && realNodes.length > 0) {
      setNodes(realNodes);
      return;
    }

    const generatedNodes: MovieNode[] = [];
    
    // Create 150 movie nodes in clusters
    for (let i = 0; i < 150; i++) {
      const genreIdx = i % GENRES.length;
      const genre = GENRES[genreIdx];
      const color = GENRE_COLORS[genreIdx];

      // Cluster position based on genre
      const clusterAngle = (genreIdx / GENRES.length) * Math.PI * 2;
      const clusterX = Math.cos(clusterAngle) * 120;
      const clusterZ = Math.sin(clusterAngle) * 120;

      // Random offset within cluster
      const x = clusterX + (Math.random() - 0.5) * 80;
      const y = (Math.random() - 0.5) * 100;
      const z = clusterZ + (Math.random() - 0.5) * 80;

      // Assign a real movie title if it matches sample index, otherwise generate dummy
      const title = i < SAMPLE_MOVIES.length
        ? SAMPLE_MOVIES[i].title
        : `${genre} Recommendation #${i - SAMPLE_MOVIES.length + 1}`;

      generatedNodes.push({
        id: i,
        title,
        genre,
        x,
        y,
        z,
        color,
      });
    }
    setNodes(generatedNodes);
  }, []);

  // Drawing loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;

    const render = () => {
      // Clear canvas
      ctx.fillStyle = "#050508";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      const fov = 400; // Camera perspective focal length

      // Apply rotation angles
      const cosX = Math.cos(rotX.current);
      const sinX = Math.sin(rotX.current);
      const cosY = Math.cos(rotY.current);
      const sinY = Math.sin(rotY.current);

      // 1. Project all nodes into 2D screen coordinates
      const projectedNodes = nodes.map((node) => {
        // Rotate around Y-axis (yaw)
        let x1 = node.x * cosY - node.z * sinY;
        let z1 = node.z * cosY + node.x * sinY;

        // Rotate around X-axis (pitch)
        let y2 = node.y * cosX - z1 * sinX;
        let z2 = z1 * cosX + node.y * sinX;

        // Perspective projection
        const cameraDistance = 300;
        const pz = z2 + cameraDistance;

        // Prevent divide by zero
        const scale = fov / Math.max(1, pz);
        const px = cx + x1 * scale;
        const py = cy + y2 * scale;

        return { ...node, px, py, pz };
      });

      // Sort by depth (pz descending) for painters algorithm (draw back to front)
      projectedNodes.sort((a, b) => (b.pz || 0) - (a.pz || 0));

      // 2. Draw connections (lines) for Similarity Vector Web
      ctx.lineWidth = 0.5;
      if (selectedNode) {
        // Draw strong similarity lines from selected node to 5 nearest neighbors
        const selProj = projectedNodes.find((n) => n.id === selectedNode.id);
        if (selProj) {
          projectedNodes.forEach((other) => {
            if (other.id !== selectedNode.id && other.genre === selectedNode.genre) {
              // Connect nodes in the same cluster/genre
              ctx.strokeStyle = `rgba(99, 102, 241, ${Math.max(0.1, 1 - (other.pz || 0) / 600)})`;
              ctx.beginPath();
              ctx.moveTo(selProj.px!, selProj.py!);
              ctx.lineTo(other.px!, other.py!);
              ctx.stroke();
            }
          });
        }
      } else if (hoveredNode) {
        const hoverProj = projectedNodes.find((n) => n.id === hoveredNode.id);
        if (hoverProj) {
          projectedNodes.forEach((other) => {
            const dist = Math.sqrt(
              Math.pow(other.x - hoverProj.x, 2) +
                Math.pow(other.y - hoverProj.y, 2) +
                Math.pow(other.z - hoverProj.z, 2)
            );
            if (dist < 60 && other.id !== hoverProj.id) {
              ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
              ctx.beginPath();
              ctx.moveTo(hoverProj.px!, hoverProj.py!);
              ctx.lineTo(other.px!, other.py!);
              ctx.stroke();
            }
          });
        }
      }

      // 3. Draw nodes (dots)
      projectedNodes.forEach((node) => {
        const radius = Math.max(2, (400 / Math.max(1, node.pz || 1)) * 2.5);
        const isHovered = hoveredNode && hoveredNode.id === node.id;
        const isSelected = selectedNode && selectedNode.id === node.id;

        // Glow effect
        if (isHovered || isSelected) {
          ctx.shadowBlur = 15;
          ctx.shadowColor = node.color;
        } else {
          ctx.shadowBlur = 0;
        }

        ctx.fillStyle = isHovered || isSelected ? "#ffffff" : node.color;
        ctx.beginPath();
        ctx.arc(node.px!, node.py!, radius * (isHovered || isSelected ? 1.5 : 1), 0, Math.PI * 2);
        ctx.fill();

        // Draw selection ring
        if (isSelected) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(node.px!, node.py!, radius * 2.2, 0, Math.PI * 2);
          ctx.stroke();
        }
      });

      // Reset shadow blur
      ctx.shadowBlur = 0;

      // Update automatic rotation if enabled
      if (autoRotate) {
        rotY.current += 0.002;
      }

      animationId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [nodes, autoRotate, hoveredNode, selectedNode]);

  // Handle drag to rotate camera
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDragging.current = true;
    setAutoRotate(false);
    previousMousePosition.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // 1. Mouse Drag Rotation
    if (isDragging.current) {
      const deltaX = e.clientX - previousMousePosition.current.x;
      const deltaY = e.clientY - previousMousePosition.current.y;

      rotY.current += deltaX * 0.005;
      rotX.current = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, rotX.current + deltaY * 0.005));

      previousMousePosition.current = { x: e.clientX, y: e.clientY };
      return;
    }

    // 2. Node Hover Detection
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const fov = 400;

    const cosX = Math.cos(rotX.current);
    const sinX = Math.sin(rotX.current);
    const cosY = Math.cos(rotY.current);
    const sinY = Math.sin(rotY.current);

    let match: MovieNode | null = null;
    let minDistance = 15; // Click/hover radius threshold

    nodes.forEach((node) => {
      let x1 = node.x * cosY - node.z * sinY;
      let z1 = node.z * cosY + node.x * sinY;
      let y2 = node.y * cosX - z1 * sinX;
      let z2 = z1 * cosX + node.y * sinX;

      const scale = fov / Math.max(1, z2 + 300);
      const px = cx + x1 * scale;
      const py = cy + y2 * scale;

      const dist = Math.sqrt(Math.pow(x - px, 2) + Math.pow(y - py, 2));
      if (dist < minDistance) {
        minDistance = dist;
        match = node;
      }
    });

    setHoveredNode(match);
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  const handleClick = () => {
    if (hoveredNode) {
      setSelectedNode(hoveredNode);
    } else {
      setSelectedNode(null);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "16px", background: "rgba(10, 10, 15, 0.4)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h2 style={{ fontSize: "1.3rem", margin: 0, display: "flex", alignItems: "center", gap: "8px" }}>
            <Sparkles size={18} style={{ color: "var(--accent)" }} />
            <span>APEX 3D Movie Embedding Space</span>
          </h2>
          <p style={{ fontSize: "0.8rem", color: "var(--muted)", margin: "4px 0 0 0" }}>
            Drag to rotate vectors in 3D. Hover to inspect recommendations. Click to isolate similarity paths.
          </p>
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          <button
            className="icon-button"
            type="button"
            onClick={() => {
              setRotX(0.5);
              setRotY(0.5);
              setAutoRotate(true);
              setSelectedNode(null);
            }}
            title="Reset Camera"
            aria-label="Reset Camera"
          >
            <RotateCcw size={16} />
          </button>
          <button
            className="sheet-link-btn"
            type="button"
            onClick={() => setAutoRotate(!autoRotate)}
            style={{ fontSize: "0.8rem", padding: "6px 12px", border: "1px solid var(--line)" }}
          >
            {autoRotate ? "Pause Auto-Rotate" : "Auto-Rotate"}
          </button>
        </div>
      </div>

      <div style={{ position: "relative", width: "100%", height: "450px", borderRadius: "12px", overflow: "hidden", background: "#050508", border: "1px solid var(--line)" }}>
        <canvas
          ref={canvasRef}
          width={800}
          height={450}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onClick={handleClick}
          style={{ width: "100%", height: "100%", cursor: isDragging.current ? "grabbing" : "grab" }}
        />

        {/* Hover / Selection Details Overlay */}
        {(hoveredNode || selectedNode) && (
          <div
            className="glass-panel"
            style={{
              position: "absolute",
              bottom: "16px",
              left: "16px",
              padding: "12px 16px",
              background: "rgba(5, 5, 8, 0.85)",
              border: "1px solid var(--line-strong)",
              minWidth: "220px",
              boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
              pointerEvents: "none",
            }}
          >
            <span style={{ fontSize: "0.7rem", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.05em", color: (selectedNode || hoveredNode)!.color }}>
              {(selectedNode || hoveredNode)!.genre} Cluster
            </span>
            <h3 style={{ fontSize: "1rem", margin: "4px 0 8px 0" }}>
              {(selectedNode || hoveredNode)!.title}
            </h3>
            <div style={{ fontSize: "0.75rem", color: "var(--muted)", display: "flex", flexDirection: "column", gap: "2px" }}>
              <span>X: {(selectedNode || hoveredNode)!.x.toFixed(1)}</span>
              <span>Y: {(selectedNode || hoveredNode)!.y.toFixed(1)}</span>
              <span>Z: {(selectedNode || hoveredNode)!.z.toFixed(1)}</span>
            </div>
            {selectedNode && (
              <div style={{ marginTop: "8px", borderTop: "1px solid var(--line)", paddingTop: "8px", fontSize: "0.75rem", color: "var(--accent)" }}>
                ⚡ Showing similarity connections
              </div>
            )}
          </div>
        )}

        {/* Legend */}
        <div
          style={{
            position: "absolute",
            top: "16px",
            right: "16px",
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            padding: "10px",
            background: "rgba(5, 5, 8, 0.6)",
            borderRadius: "8px",
            border: "1px solid var(--line)",
          }}
        >
          {GENRES.map((g, idx) => (
            <div key={g} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.75rem" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: GENRE_COLORS[idx] }}></span>
              <span>{g}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
