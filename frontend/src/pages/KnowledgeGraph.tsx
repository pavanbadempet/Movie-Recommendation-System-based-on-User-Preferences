import React, { useEffect, useRef, useState, useCallback } from "react";
import * as d3 from "d3";
import { Film, Loader2, Search, X, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";
import { useKnowledgeGraph } from "../hooks/useKnowledgeGraph";
import type { GraphNode, GraphEdge } from "../hooks/useKnowledgeGraph";
import type { Movie, MovieTitle } from "../types";

const imageBase = "https://image.tmdb.org/t/p/w185";

function posterUrl(path?: string | null): string {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${imageBase}${path}`;
}

function fullPosterUrl(path?: string | null): string {
  if (!path) return "https://placehold.co/500x750/141418/f8fafc?text=Movie";
  if (path.startsWith("http")) return path;
  return `https://image.tmdb.org/t/p/w500${path}`;
}

// ─── Side Panel ───────────────────────────────────────────────────────────────

function NodeSidePanel({ movie, onClose }: { movie: Movie; onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const firstFocusable = panelRef.current?.querySelector<HTMLElement>(
      'button, [href], input, [tabindex]:not([tabindex="-1"])',
    );
    firstFocusable?.focus();
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <aside ref={panelRef} className="kg-side-panel" aria-label={`Details for ${movie.title}`}>
      <button className="kg-panel-close" type="button" aria-label="Close details panel" onClick={onClose}>
        <X size={16} aria-hidden="true" />
      </button>
      <div className="kg-panel-poster-container">
        <img src={fullPosterUrl(movie.poster_path)} alt={`Poster for ${movie.title}`} className="kg-panel-poster" />
      </div>
      <h3 className="kg-panel-title">{movie.title}</h3>

      <div className="kg-panel-meta-row">
        {movie.release_date && (
          <span className="kg-meta-badge year">{movie.release_date.slice(0, 4)}</span>
        )}
        {movie.vote_average != null && movie.vote_average > 0 && (
          <span className="kg-meta-badge rating">⭐ {Number(movie.vote_average).toFixed(1)}</span>
        )}
      </div>

      {movie.genres && (
        <div className="kg-panel-genres">
          {movie.genres.split(",").map((genre) => {
            const trimmed = genre.trim();
            return trimmed ? (
              <span key={trimmed} className="kg-genre-badge">{trimmed}</span>
            ) : null;
          })}
        </div>
      )}

      {movie.overview && <p className="kg-panel-overview">{movie.overview}</p>}
    </aside>
  );
}

// ─── D3 Force Graph ───────────────────────────────────────────────────────────

interface D3Node extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  type: "seed" | "recommendation";
  movie: Movie;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface D3Link extends d3.SimulationLinkDatum<D3Node> {
  label: string;
}

function ForceGraph({
  nodes,
  edges,
  onNodeClick,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick: (movie: Movie) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<HTMLCanvasElement, unknown> | null>(null);

  const handleZoomIn = useCallback(() => {
    if (canvasRef.current && zoomRef.current) {
      d3.select(canvasRef.current).transition().duration(300).call(zoomRef.current.scaleBy, 1.4);
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (canvasRef.current && zoomRef.current) {
      d3.select(canvasRef.current).transition().duration(300).call(zoomRef.current.scaleBy, 0.7);
    }
  }, []);

  const handleReset = useCallback(() => {
    if (canvasRef.current && zoomRef.current) {
      d3.select(canvasRef.current).transition().duration(400).call(zoomRef.current.transform, d3.zoomIdentity);
    }
  }, []);

  useEffect(() => {
    if (!canvasRef.current || !containerRef.current || nodes.length === 0) return;

    const width = containerRef.current.clientWidth || 700;
    const height = containerRef.current.clientHeight || 500;

    const dpr = window.devicePixelRatio || 1;
    const canvas = canvasRef.current;
    canvas.width = width * dpr;
    canvas.height = height * dpr;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const d3Nodes: D3Node[] = nodes.map((n) => ({ ...n }));
    const nodeById = new Map(d3Nodes.map((n) => [n.id, n]));

    const d3Links: D3Link[] = edges
      .map((e) => ({
        source: nodeById.get(e.source) ?? e.source,
        target: nodeById.get(e.target) ?? e.target,
        label: e.label,
      }))
      .filter((l) => l.source && l.target) as unknown as D3Link[];

    let transform = d3.zoomIdentity;
    let hoveredNode: D3Node | null = null;

    // Load poster images
    const images = new Map<string, HTMLImageElement>();
    d3Nodes.forEach((node) => {
      const url = posterUrl(node.movie.poster_path);
      if (url) {
        const img = new Image();
        img.src = url;
        img.onload = () => {
          draw();
        };
        images.set(node.id, img);
      }
    });

    const draw = () => {
      if (!ctx) return;
      const currentHovered: D3Node | null = hoveredNode;
      ctx.clearRect(0, 0, width, height);
      ctx.save();
      ctx.translate(transform.x, transform.y);
      ctx.scale(transform.k, transform.k);

      // 1. Draw Links
      d3Links.forEach((link) => {
        const src = link.source as unknown as D3Node;
        const tgt = link.target as unknown as D3Node;
        if (src.x == null || src.y == null || tgt.x == null || tgt.y == null) return;

        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);

        let strokeColor = "rgba(124,58,237,0.3)";
        let lineWidth = 1.5;

        if (currentHovered) {
          const isConnected = src.id === currentHovered.id || tgt.id === currentHovered.id;
          strokeColor = isConnected ? "#ec4899" : "rgba(124,58,237,0.08)";
          lineWidth = isConnected ? 2.5 : 1.0;
        }

        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = lineWidth;
        ctx.stroke();

        // Arrow head drawing helper
        const angle = Math.atan2(tgt.y - src.y, tgt.x - src.x);
        const nodeRadius = tgt.type === "seed" ? 22 : 16;
        const arrowX = tgt.x - (nodeRadius + 2) * Math.cos(angle);
        const arrowY = tgt.y - (nodeRadius + 2) * Math.sin(angle);
        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(arrowX - 8 * Math.cos(angle - Math.PI / 6), arrowY - 8 * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(arrowX - 8 * Math.cos(angle + Math.PI / 6), arrowY - 8 * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fillStyle = strokeColor;
        ctx.fill();

        // Link Label
        const mx = (src.x + tgt.x) / 2;
        const my = (src.y + tgt.y) / 2;
        ctx.save();
        ctx.fillStyle = currentHovered 
          ? ((src.id === currentHovered.id || tgt.id === currentHovered.id) ? "rgba(255,255,255,0.7)" : "rgba(255,255,255,0.1)")
          : "rgba(255,255,255,0.35)";
        ctx.font = "8px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(link.label, mx, my);
        ctx.restore();
      });

      // 2. Draw Nodes
      d3Nodes.forEach((node) => {
        if (node.x == null || node.y == null) return;
        const r = node.type === "seed" ? 22 : 16;

        let opacity = 1.0;
        if (currentHovered) {
          const isConnected = d3Links.some((l) => {
            const src = l.source as unknown as D3Node;
            const tgt = l.target as unknown as D3Node;
            return (
              (src.id === currentHovered.id && tgt.id === node.id) ||
              (tgt.id === currentHovered.id && src.id === node.id)
            );
          });
          opacity = (node.id === currentHovered.id || isConnected) ? 1.0 : 0.3;
        }

        ctx.save();
        ctx.globalAlpha = opacity;

        // Draw circular border / background
        ctx.beginPath();
        ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = node.type === "seed" ? "rgba(124,58,237,0.9)" : "rgba(22,24,34,0.95)";
        ctx.strokeStyle = node.type === "seed" ? "#7c3aed" : "rgba(255,255,255,0.18)";
        ctx.lineWidth = node.type === "seed" ? 2.5 : 1.5;
        ctx.fill();
        ctx.stroke();

        // Draw poster image inside node (clipped)
        const img = images.get(node.id);
        if (img && img.complete && img.naturalWidth !== 0) {
          ctx.save();
          ctx.beginPath();
          ctx.arc(node.x, node.y, r - 2, 0, 2 * Math.PI);
          ctx.clip();
          ctx.drawImage(img, node.x - r + 2, node.y - r + 2, (r - 2) * 2, (r - 2) * 2);
          ctx.restore();
        }

        // Draw label text below node
        const labelY = node.y + (node.type === "seed" ? 38 : 30);
        ctx.fillStyle = "#e8ecf5";
        ctx.font = node.type === "seed" ? "bold 11px sans-serif" : "500 9px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        const shortLabel = node.label.length > 16 ? `${node.label.slice(0, 14)}…` : node.label;
        ctx.fillText(shortLabel, node.x, labelY);

        ctx.restore();
      });

      ctx.restore();
    };

    const simulation = d3
      .forceSimulation<D3Node>(d3Nodes)
      .force("link", d3.forceLink<D3Node, D3Link>(d3Links).id((d) => d.id).distance(140))
      .force("charge", d3.forceManyBody().strength(-400))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(44));

    // Zoom behaviour
    const zoom = d3.zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.3, 4])
      .on("zoom", (event) => {
        transform = event.transform;
        draw();
      });
    zoomRef.current = zoom;
    d3.select(canvas).call(zoom);

    simulation.on("tick", () => {
      draw();
    });

    // Interaction Handlers (Hover, Click, Drag)
    d3.select(canvas).on("mousemove", (event) => {
      const [mx, my] = d3.pointer(event, canvas);
      const point = transform.invert([mx, my]);
      const px = point[0];
      const py = point[1];

      let found: D3Node | null = null;
      for (const node of d3Nodes) {
        if (node.x == null || node.y == null) continue;
        const r = node.type === "seed" ? 22 : 16;
        const dist = Math.hypot(node.x - px, node.y - py);
        if (dist <= r) {
          found = node;
          break;
        }
      }

      if (found !== hoveredNode) {
        hoveredNode = found;
        draw();
      }
    });

    d3.select(canvas).on("click", (event) => {
      const [mx, my] = d3.pointer(event, canvas);
      const point = transform.invert([mx, my]);
      const px = point[0];
      const py = point[1];

      for (const node of d3Nodes) {
        if (node.x == null || node.y == null) continue;
        const r = node.type === "seed" ? 22 : 16;
        if (Math.hypot(node.x - px, node.y - py) <= r) {
          onNodeClick(node.movie);
          break;
        }
      }
    });

    // Drag behavior setup
    const drag = d3.drag<HTMLCanvasElement, D3Node>()
      .subject((event) => {
        const [mx, my] = d3.pointer(event, canvas);
        const point = transform.invert([mx, my]);
        const px = point[0];
        const py = point[1];

        for (const node of d3Nodes) {
          if (node.x == null || node.y == null) continue;
          const r = node.type === "seed" ? 22 : 16;
          if (Math.hypot(node.x - px, node.y - py) <= r) {
            return node;
          }
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        return null as any;
      })
      .on("start", (event) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      })
      .on("drag", (event) => {
        const [mx, my] = d3.pointer(event, canvas);
        const point = transform.invert([mx, my]);
        event.subject.fx = point[0];
        event.subject.fy = point[1];
      })
      .on("end", (event) => {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    d3.select(canvas).call(drag as any);

    return () => {
      simulation.stop();
    };
  }, [nodes, edges, onNodeClick]);

  return (
    <div ref={containerRef} className="kg-graph-container" aria-label="Knowledge graph">
      <canvas ref={canvasRef} className="kg-svg" style={{ display: "block", width: "100%", height: "100%" }} />
      {/* Zoom controls */}
      <div className="kg-zoom-controls" aria-label="Graph zoom controls">
        <button type="button" onClick={handleZoomIn} aria-label="Zoom in" title="Zoom in" className="kg-zoom-btn">
          <ZoomIn size={16} aria-hidden="true" />
        </button>
        <button type="button" onClick={handleZoomOut} aria-label="Zoom out" title="Zoom out" className="kg-zoom-btn">
          <ZoomOut size={16} aria-hidden="true" />
        </button>
        <button type="button" onClick={handleReset} aria-label="Reset zoom" title="Reset zoom" className="kg-zoom-btn">
          <Maximize2 size={16} aria-hidden="true" />
        </button>
      </div>
      {/* Legend */}
      <div className="kg-legend" aria-label="Graph legend">
        <span className="kg-legend-seed" aria-label="Seed movie node">Seed</span>
        <span className="kg-legend-rec" aria-label="Recommendation node">Recommendation</span>
      </div>
    </div>
  );
}

// ─── Knowledge Graph Page ─────────────────────────────────────────────────────

export function KnowledgeGraphPage({ titles }: { titles: MovieTitle[] }) {
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [selectedPanel, setSelectedPanel] = useState<Movie | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(false);

  const { graphData, loading, error } = useKnowledgeGraph(selectedId);

  const filtered = React.useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return titles.slice(0, 20);
    return titles.filter((t) => t.title.toLowerCase().includes(q)).slice(0, 20);
  }, [titles, query]);

  function selectTitle(t: MovieTitle) {
    setQuery(t.title);
    setSelectedId(t.id);
    setShowSuggestions(false);
    setSelectedPanel(null);
  }

  function clearSelection() {
    setQuery("");
    setSelectedId(null);
    setSelectedPanel(null);
  }

  const nodeCount = graphData?.nodes.length ?? 0;
  const edgeCount = graphData?.edges.length ?? 0;

  return (
    <section className="kg-shell" aria-labelledby="kg-heading">
      <div className="kg-header">
        <h2 id="kg-heading">Knowledge Graph</h2>
        <p className="dashboard-subtitle">
          Multi-hop semantic reasoning — explore how movies connect through themes, genres, and concepts.
        </p>
      </div>

      {/* Search */}
      <div className="kg-search-row">
        <div className="search-box kg-search-box" role="search">
          <label htmlFor="kg-movie-search" className="visually-hidden">Search for a seed movie</label>
          <Search size={16} aria-hidden="true" />
          <input
            id="kg-movie-search"
            type="text"
            role="combobox"
            placeholder="Search for a seed movie…"
            value={query}
            aria-label="Search for a seed movie"
            aria-autocomplete="list"
            aria-controls="kg-suggestions"
            aria-expanded={showSuggestions && filtered.length > 0}
            aria-haspopup="listbox"
            onChange={(e) => { setQuery(e.target.value); setShowSuggestions(true); }}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
            onKeyDown={(e) => {
              if (e.key === "Escape") { setShowSuggestions(false); clearSelection(); }
              if (e.key === "Enter" && filtered[0]) selectTitle(filtered[0]);
            }}
          />
          {query && (
            <button type="button" aria-label="Clear search" onClick={clearSelection} className="kg-clear-btn">
              <X size={14} aria-hidden="true" />
            </button>
          )}
        </div>

        {showSuggestions && filtered.length > 0 && (
          <ul id="kg-suggestions" className="kg-suggestions" role="listbox" aria-label="Movie suggestions">
            {filtered.map((t) => (
              <li key={t.id} role="option" aria-selected={t.id === selectedId}>
                <button type="button" onMouseDown={() => selectTitle(t)} tabIndex={0}>
                  <Film size={13} aria-hidden="true" />
                  {t.title}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Graph stats bar */}
      {nodeCount > 0 && (
        <div className="kg-stats-bar" aria-label="Graph statistics">
          <span>{nodeCount} nodes</span>
          <span>{edgeCount} connections</span>
          <span className="kg-stats-hint">Drag nodes · Scroll to zoom · Click for details</span>
        </div>
      )}

      {/* Graph area */}
      <div className="kg-content">
        {!selectedId && (
          <div className="kg-empty" role="status">
            <Film size={40} aria-hidden="true" />
            <p>Select a movie above to explore its knowledge graph connections.</p>
          </div>
        )}
        {selectedId && loading && (
          <div className="kg-loading" role="status" aria-live="polite">
            <Loader2 size={28} className="spin" aria-hidden="true" />
            <span>Building knowledge graph…</span>
          </div>
        )}
        {selectedId && error && (
          <div className="kg-empty" role="alert">
            <X size={40} aria-hidden="true" style={{ color: "var(--danger)" }} />
            <p style={{ color: "var(--text)", fontWeight: 600, marginBottom: 4 }}>
              Knowledge Graph Unavailable
            </p>
            <p style={{ color: "var(--muted)", fontSize: "0.85rem", maxWidth: 420, textAlign: "center", lineHeight: 1.5 }}>
              {error.includes("503") || error.includes("disabled")
                ? "The Knowledge Graph requires precomputed graph artifacts that aren't loaded in this environment. This feature is available when running with the full data pipeline."
                : error}
            </p>
            <button
              type="button"
              style={{
                marginTop: 12,
                padding: "8px 20px",
                background: "var(--panel-hover)",
                border: "1px solid var(--line)",
                borderRadius: 8,
                color: "var(--text)",
                fontSize: "0.82rem",
                fontWeight: 600,
                cursor: "pointer",
              }}
              onClick={() => { setSelectedId(null); setQuery(""); }}
            >
              Try another movie
            </button>
          </div>
        )}
        {selectedId && !loading && !error && graphData && graphData.nodes.length === 0 && (
          <p role="status">No knowledge graph connections found for this movie.</p>
        )}
        {selectedId && !loading && !error && graphData && graphData.nodes.length > 0 && (
          <div className="kg-graph-wrapper">
            <ForceGraph nodes={graphData.nodes} edges={graphData.edges} onNodeClick={(movie) => setSelectedPanel(movie)} />
            {selectedPanel && <NodeSidePanel movie={selectedPanel} onClose={() => setSelectedPanel(null)} />}
          </div>
        )}
      </div>
    </section>
  );
}
