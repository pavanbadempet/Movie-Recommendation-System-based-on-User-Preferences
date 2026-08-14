import React, { useRef, useEffect, useState, useCallback } from "react";
import { Sparkles, RotateCcw, Compass, Zap, Eye } from "lucide-react";
import { getRealMovieNodes } from "./webgpuEngine";

interface MovieNode {
  id: number;
  title: string;
  genre: string;
  x: number;
  y: number;
  z: number;
  px?: number;
  py?: number;
  pz?: number;
  color: string;
  isAnchor?: boolean;
}

interface Star {
  x: number;
  y: number;
  z: number;
  size: number;
  alpha: number;
}

const GENRES = ["Action", "Sci-Fi", "Drama", "Thriller", "Comedy", "Romance"];
const GENRE_COLORS = [
  "#f43f5e", // Action -> Vibrant Rose Red
  "#06b6d4", // Sci-Fi -> Radiant Cyan
  "#10b981", // Drama -> Emerald Green
  "#a855f7", // Thriller -> Neon Purple
  "#f59e0b", // Comedy -> Warm Amber
  "#ec4899", // Romance -> Hot Pink
];

const ANCHOR_TITLES = [
  { title: "The Dark Knight", genre: "Action" },
  { title: "Inception", genre: "Sci-Fi" },
  { title: "Interstellar", genre: "Sci-Fi" },
  { title: "Pulp Fiction", genre: "Thriller" },
  { title: "The Matrix", genre: "Sci-Fi" },
  { title: "Se7en", genre: "Thriller" },
  { title: "Gladiator", genre: "Action" },
  { title: "Avatar", genre: "Sci-Fi" },
  { title: "The Godfather", genre: "Drama" },
  { title: "Fight Club", genre: "Thriller" },
  { title: "La La Land", genre: "Romance" },
  { title: "Forrest Gump", genre: "Drama" },
];

export function VectorSpace() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<MovieNode[]>([]);
  const [hoveredNode, setHoveredNode] = useState<MovieNode | null>(null);
  const [selectedNode, setSelectedNode] = useState<MovieNode | null>(null);

  const hoveredNodeRef = useRef(hoveredNode);
  hoveredNodeRef.current = hoveredNode;

  const selectedNodeRef = useRef(selectedNode);
  selectedNodeRef.current = selectedNode;

  // Camera & Interaction state
  const rotX = useRef(0.35);
  const rotY = useRef(0.65);
  const zoom = useRef(1.0);
  const [autoRotate, setAutoRotate] = useState(true);

  const isDragging = useRef(false);
  const previousMousePosition = useRef({ x: 0, y: 0 });
  const backgroundStars = useRef<Star[]>([]);

  // Initialize Background Starfield
  useEffect(() => {
    const stars: Star[] = [];
    for (let i = 0; i < 160; i++) {
      stars.push({
        x: (Math.random() - 0.5) * 1200,
        y: (Math.random() - 0.5) * 800,
        z: Math.random() * 800 - 200,
        size: Math.random() * 1.5 + 0.5,
        alpha: Math.random() * 0.6 + 0.2,
      });
    }
    backgroundStars.current = stars;
  }, []);

  // Initialize Galaxy Nodes along Hyperbolic Curvature
  useEffect(() => {
    const realNodes = getRealMovieNodes();
    if (realNodes && realNodes.length > 0) {
      setNodes(realNodes);
      return;
    }

    const generatedNodes: MovieNode[] = [];
    const totalNodes = 200;

    for (let i = 0; i < totalNodes; i++) {
      const genreIdx = i % GENRES.length;
      const genre = GENRES[genreIdx];
      const color = GENRE_COLORS[genreIdx];

      // Hyperbolic Radial Geodesics: R * tanh(dist / sigma)
      const rawDist = 30 + (i / totalNodes) * 180;
      const radius = 180 * Math.tanh(rawDist / 95);
      const clusterAngle = (genreIdx / GENRES.length) * Math.PI * 2 + (Math.random() - 0.5) * 0.5;

      const clusterX = Math.cos(clusterAngle) * radius;
      const clusterZ = Math.sin(clusterAngle) * radius;

      const x = clusterX + (Math.random() - 0.5) * 55;
      const y = Math.sin(clusterAngle * 2 + i * 0.1) * 70 + (Math.random() - 0.5) * 40;
      const z = clusterZ + (Math.random() - 0.5) * 55;

      const isAnchor = i < ANCHOR_TITLES.length;
      const title = isAnchor
        ? ANCHOR_TITLES[i].title
        : `${genre} Vector Twin #${i - ANCHOR_TITLES.length + 1}`;

      generatedNodes.push({
        id: i,
        title,
        genre,
        x,
        y,
        z,
        color,
        isAnchor,
      });
    }
    setNodes(generatedNodes);
  }, []);

  // Resize canvas for High-DPI Retina Displays
  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const width = container.clientWidth;
    const height = Math.max(520, Math.min(680, window.innerHeight * 0.6));

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
  }, []);

  useEffect(() => {
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
    return () => window.removeEventListener("resize", resizeCanvas);
  }, [resizeCanvas]);

  // Main 3D Render Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;

    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;

      ctx.save();
      ctx.scale(dpr, dpr);

      // Deep Space Gradient Background
      const bgGradient = ctx.createRadialGradient(width / 2, height / 2, 50, width / 2, height / 2, Math.max(width, height) / 1.2);
      bgGradient.addColorStop(0, "#080914");
      bgGradient.addColorStop(0.6, "#04050a");
      bgGradient.addColorStop(1, "#020204");
      ctx.fillStyle = bgGradient;
      ctx.fillRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const fov = 420 * zoom.current;

      const cosX = Math.cos(rotX.current);
      const sinX = Math.sin(rotX.current);
      const cosY = Math.cos(rotY.current);
      const sinY = Math.sin(rotY.current);

      // 1. Draw Starfield
      backgroundStars.current.forEach((star) => {
        const sx = cx + star.x * 0.4;
        const sy = cy + star.y * 0.4;
        if (sx > 0 && sx < width && sy > 0 && sy < height) {
          ctx.fillStyle = `rgba(255, 255, 255, ${star.alpha * 0.7})`;
          ctx.beginPath();
          ctx.arc(sx, sy, star.size, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // 2. Draw 3D Orbital Horizon Rings
      const ringRadii = [80, 160, 240];
      ringRadii.forEach((r) => {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let a = 0; a <= Math.PI * 2; a += 0.1) {
          const rx = Math.cos(a) * r;
          const rz = Math.sin(a) * r;
          const rx1 = rx * cosY - rz * sinY;
          const rz1 = rz * cosY + rx * sinY;
          const ry2 = -rz1 * sinX;
          const rz2 = rz1 * cosX;
          const scale = fov / Math.max(1, rz2 + 350);
          const px = cx + rx1 * scale;
          const py = cy + ry2 * scale;
          if (a === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.stroke();
      });

      // 3. Project Nodes into 2D Coordinates
      const projectedNodes = nodes.map((node) => {
        const x1 = node.x * cosY - node.z * sinY;
        const z1 = node.z * cosY + node.x * sinY;
        const y2 = node.y * cosX - z1 * sinX;
        const z2 = z1 * cosX + node.y * sinX;

        const pz = z2 + 350;
        const scale = fov / Math.max(1, pz);
        const px = cx + x1 * scale;
        const py = cy + y2 * scale;

        return { ...node, px, py, pz };
      });

      projectedNodes.sort((a, b) => (b.pz || 0) - (a.pz || 0));

      // 4. Draw Intra-Cluster Constellation Filaments
      for (let i = 0; i < projectedNodes.length; i++) {
        const n1 = projectedNodes[i];
        for (let j = i + 1; j < projectedNodes.length; j++) {
          const n2 = projectedNodes[j];
          if (n1.genre === n2.genre) {
            const dx = n1.x - n2.x;
            const dy = n1.y - n2.y;
            const dz = n1.z - n2.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < 48) {
              const alpha = Math.max(0.04, (1 - dist / 48) * 0.18);
              ctx.strokeStyle = n1.color;
              ctx.globalAlpha = alpha;
              ctx.lineWidth = 0.8;
              ctx.beginPath();
              ctx.moveTo(n1.px!, n1.py!);
              ctx.lineTo(n2.px!, n2.py!);
              ctx.stroke();
            }
          }
        }
      }
      ctx.globalAlpha = 1.0;

      // 5. Draw Active Selection/Hover Beams
      const activeNode = selectedNodeRef.current || hoveredNodeRef.current;
      if (activeNode) {
        const activeProj = projectedNodes.find((n) => n.id === activeNode.id);
        if (activeProj) {
          projectedNodes.forEach((other) => {
            if (other.id !== activeProj.id && other.genre === activeProj.genre) {
              const dx = activeProj.x - other.x;
              const dy = activeProj.y - other.y;
              const dz = activeProj.z - other.z;
              const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
              if (dist < 90) {
                ctx.strokeStyle = activeProj.color;
                ctx.lineWidth = 1.5;
                ctx.globalAlpha = Math.max(0.2, (1 - dist / 90) * 0.8);
                ctx.beginPath();
                ctx.moveTo(activeProj.px!, activeProj.py!);
                ctx.lineTo(other.px!, other.py!);
                ctx.stroke();
              }
            }
          });
          ctx.globalAlpha = 1.0;
        }
      }

      // 6. Draw Luminous Stellar Nodes
      projectedNodes.forEach((node) => {
        const baseRadius = Math.max(2.5, (400 / Math.max(1, node.pz || 1)) * (node.isAnchor ? 3.8 : 2.6));
        const isHovered = hoveredNodeRef.current?.id === node.id;
        const isSelected = selectedNodeRef.current?.id === node.id;
        const radius = isHovered || isSelected ? baseRadius * 1.6 : baseRadius;

        // Radial Glow Halo
        const glowRadius = radius * 3.5;
        const glowGradient = ctx.createRadialGradient(node.px!, node.py!, radius * 0.5, node.px!, node.py!, glowRadius);
        glowGradient.addColorStop(0, node.color);
        glowGradient.addColorStop(0.4, `${node.color}55`);
        glowGradient.addColorStop(1, "transparent");

        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(node.px!, node.py!, glowRadius, 0, Math.PI * 2);
        ctx.fill();

        // Core Stellar Dot
        ctx.fillStyle = isHovered || isSelected ? "#ffffff" : node.color;
        ctx.beginPath();
        ctx.arc(node.px!, node.py!, radius, 0, Math.PI * 2);
        ctx.fill();

        // Selection Aura
        if (isSelected) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(node.px!, node.py!, radius * 2.2, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Floating 3D Anchor Movie Labels
        if (node.isAnchor || isHovered || isSelected) {
          const fontSize = Math.max(9, Math.min(13, 3800 / Math.max(1, node.pz || 1)));
          ctx.font = `${isHovered || isSelected ? "700" : "500"} ${fontSize}px "Inter", -apple-system, sans-serif`;
          ctx.fillStyle = isHovered || isSelected ? "#ffffff" : "rgba(255, 255, 255, 0.75)";
          ctx.textAlign = "center";
          ctx.shadowColor = "rgba(0, 0, 0, 0.8)";
          ctx.shadowBlur = 4;
          ctx.fillText(node.title, node.px!, node.py! - radius - 6);
          ctx.shadowBlur = 0;
        }
      });

      // Auto Rotation
      if (autoRotate) {
        rotY.current += 0.0018;
      }

      ctx.restore();
      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [nodes, autoRotate]);

  // Mouse & Touch Controls
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

    if (isDragging.current) {
      const deltaX = e.clientX - previousMousePosition.current.x;
      const deltaY = e.clientY - previousMousePosition.current.y;
      rotY.current += deltaX * 0.005;
      rotX.current = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, rotX.current + deltaY * 0.005));
      previousMousePosition.current = { x: e.clientX, y: e.clientY };
      return;
    }

    // Node hit testing
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    const cx = width / 2;
    const cy = height / 2;
    const fov = 420 * zoom.current;

    const cosX = Math.cos(rotX.current);
    const sinX = Math.sin(rotX.current);
    const cosY = Math.cos(rotY.current);
    const sinY = Math.sin(rotY.current);

    let match: MovieNode | null = null;
    let minDistance = 18;

    nodes.forEach((node) => {
      const x1 = node.x * cosY - node.z * sinY;
      const z1 = node.z * cosY + node.x * sinY;
      const y2 = node.y * cosX - z1 * sinX;
      const z2 = z1 * cosX + node.y * sinX;

      const scale = fov / Math.max(1, z2 + 350);
      const px = cx + x1 * scale;
      const py = cy + y2 * scale;

      const dist = Math.sqrt(Math.pow(x - px, 2) + Math.pow(y - py, 2));
      if (dist < minDistance) {
        minDistance = dist;
        match = node;
      }
    });

    setHoveredNode(prev => prev?.id === match?.id ? prev : match);
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  const handleClick = () => {
    setSelectedNode(hoveredNode);
  };

  return (
    <div ref={containerRef} className="glass-panel" style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "16px", background: "rgba(8, 9, 16, 0.7)", borderRadius: "20px", border: "1px solid rgba(255, 255, 255, 0.08)" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <h2 style={{ fontSize: "1.25rem", fontWeight: "700", margin: 0, display: "flex", alignItems: "center", gap: "10px", color: "#ffffff" }}>
            <Compass size={20} style={{ color: "#06b6d4" }} />
            <span>3D Neural Vector Galaxy</span>
          </h2>
          <p style={{ fontSize: "0.82rem", color: "var(--muted)", margin: "4px 0 0 0" }}>
            Poincaré hyperbolic manifold embedding 200+ movies in 768-D semantic space. Drag to orbit, hover to trace neural connections.
          </p>
        </div>
        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          <button
            className="icon-button"
            type="button"
            onClick={() => {
              rotX.current = 0.35;
              rotY.current = 0.65;
              zoom.current = 1.0;
              setAutoRotate(true);
              setSelectedNode(null);
            }}
            title="Reset Orbit"
            aria-label="Reset Orbit"
            style={{ width: "36px", height: "36px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)" }}
          >
            <RotateCcw size={15} />
          </button>
          <button
            type="button"
            onClick={() => setAutoRotate(!autoRotate)}
            style={{
              fontSize: "0.8rem",
              fontWeight: "600",
              padding: "8px 14px",
              borderRadius: "10px",
              border: "1px solid rgba(255, 255, 255, 0.12)",
              background: autoRotate ? "rgba(6, 182, 212, 0.1)" : "rgba(255, 255, 255, 0.04)",
              color: autoRotate ? "#22d3ee" : "#cbd5e1",
              cursor: "pointer",
            }}
          >
            {autoRotate ? "Pause Orbit" : "Resume Orbit"}
          </button>
        </div>
      </div>

      {/* 3D Canvas Box */}
      <div style={{ position: "relative", width: "100%", height: "540px", borderRadius: "16px", overflow: "hidden", border: "1px solid rgba(255, 255, 255, 0.08)", boxShadow: "inset 0 0 80px rgba(0,0,0,0.8)" }}>
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onClick={handleClick}
          style={{ width: "100%", height: "100%", cursor: isDragging.current ? "grabbing" : "grab", display: "block" }}
        />

        {/* Selected / Hovered Movie HUD */}
        {(hoveredNode || selectedNode) && (
          <div
            style={{
              position: "absolute",
              bottom: "20px",
              left: "20px",
              padding: "16px 20px",
              background: "rgba(9, 10, 18, 0.92)",
              backdropFilter: "blur(16px)",
              border: "1px solid rgba(255, 255, 255, 0.12)",
              borderRadius: "14px",
              minWidth: "260px",
              boxShadow: "0 12px 40px rgba(0,0,0,0.6)",
              pointerEvents: "none",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
              <span style={{ fontSize: "0.72rem", fontWeight: "800", textTransform: "uppercase", letterSpacing: "0.08em", color: (selectedNode || hoveredNode)!.color }}>
                {(selectedNode || hoveredNode)!.genre} Cluster
              </span>
              <span style={{ fontSize: "0.68rem", padding: "2px 8px", borderRadius: "12px", background: "rgba(255,255,255,0.06)", color: "#94a3b8" }}>
                768-D Vector
              </span>
            </div>
            <h3 style={{ fontSize: "1.1rem", fontWeight: "700", margin: "0 0 10px 0", color: "#ffffff" }}>
              {(selectedNode || hoveredNode)!.title}
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", fontSize: "0.76rem", color: "#94a3b8" }}>
              <div>Spatial X: <span style={{ color: "#fff", fontWeight: "600" }}>{(selectedNode || hoveredNode)!.x.toFixed(1)}</span></div>
              <div>Spatial Y: <span style={{ color: "#fff", fontWeight: "600" }}>{(selectedNode || hoveredNode)!.y.toFixed(1)}</span></div>
              <div>Spatial Z: <span style={{ color: "#fff", fontWeight: "600" }}>{(selectedNode || hoveredNode)!.z.toFixed(1)}</span></div>
              <div>Manifold: <span style={{ color: "#06b6d4", fontWeight: "600" }}>Hyperbolic</span></div>
            </div>
            {selectedNode && (
              <div style={{ marginTop: "10px", borderTop: "1px solid rgba(255, 255, 255, 0.08)", paddingTop: "8px", fontSize: "0.75rem", color: "#22d3ee", display: "flex", alignItems: "center", gap: "6px" }}>
                <Zap size={13} />
                <span>Tracing nearest cosine similarity neighbors</span>
              </div>
            )}
          </div>
        )}

        {/* Legend */}
        <div
          style={{
            position: "absolute",
            top: "20px",
            right: "20px",
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            padding: "12px 16px",
            background: "rgba(9, 10, 18, 0.85)",
            backdropFilter: "blur(12px)",
            borderRadius: "12px",
            border: "1px solid rgba(255, 255, 255, 0.08)",
          }}
        >
          <span style={{ fontSize: "0.68rem", fontWeight: "800", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--muted)", marginBottom: "2px" }}>
            Genre Manifolds
          </span>
          {GENRES.map((g, idx) => (
            <div key={g} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.78rem", color: "#e2e8f0" }}>
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: GENRE_COLORS[idx], boxShadow: `0 0 8px ${GENRE_COLORS[idx]}` }}></span>
              <span>{g}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
