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
        <X size={18} aria-hidden="true" />
      </button>
      <img src={fullPosterUrl(movie.poster_path)} alt={`Poster for ${movie.title}`} className="kg-panel-poster" />
      <h3 className="kg-panel-title">{movie.title}</h3>
      {movie.release_date && <p className="kg-panel-meta">{movie.release_date.slice(0, 4)}</p>}
      {movie.genres && <p className="kg-panel-meta">{movie.genres}</p>}
      {movie.vote_average != null && movie.vote_average > 0 && (
        <p className="kg-panel-meta">⭐ {Number(movie.vote_average).toFixed(1)}</p>
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
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  const handleZoomIn = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(300).call(zoomRef.current.scaleBy, 1.4);
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(300).call(zoomRef.current.scaleBy, 0.7);
    }
  }, []);

  const handleReset = useCallback(() => {
    if (svgRef.current && zoomRef.current) {
      d3.select(svgRef.current).transition().duration(400).call(zoomRef.current.transform, d3.zoomIdentity);
    }
  }, []);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || nodes.length === 0) return;

    const width = containerRef.current.clientWidth || 700;
    const height = containerRef.current.clientHeight || 500;

    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("aria-label", "Knowledge graph visualization");

    // Defs: arrowhead + clip path for poster thumbnails
    const defs = svg.append("defs");
    defs.append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 26)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "rgba(229,9,20,0.6)");

    // Clip path for circular poster thumbnails
    defs.append("clipPath")
      .attr("id", "circle-clip-seed")
      .append("circle")
      .attr("r", 20);

    defs.append("clipPath")
      .attr("id", "circle-clip-rec")
      .append("circle")
      .attr("r", 14);

    const d3Nodes: D3Node[] = nodes.map((n) => ({ ...n }));
    const nodeById = new Map(d3Nodes.map((n) => [n.id, n]));

    const d3Links: D3Link[] = edges
      .map((e) => ({
        source: nodeById.get(e.source) ?? e.source,
        target: nodeById.get(e.target) ?? e.target,
        label: e.label,
      }))
      .filter((l) => l.source && l.target);

    // Zoom behaviour
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 4])
      .on("zoom", (event) => {
        graphGroup.attr("transform", event.transform);
      });
    zoomRef.current = zoom;
    svg.call(zoom);

    const graphGroup = svg.append("g");

    const simulation = d3
      .forceSimulation<D3Node>(d3Nodes)
      .force("link", d3.forceLink<D3Node, D3Link>(d3Links).id((d) => d.id).distance(140))
      .force("charge", d3.forceManyBody().strength(-400))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(44));

    const linkGroup = graphGroup.append("g").attr("aria-hidden", "true");
    const link = linkGroup
      .selectAll("line")
      .data(d3Links)
      .join("line")
      .attr("stroke", "rgba(229,9,20,0.3)")
      .attr("stroke-width", 1.5)
      .attr("marker-end", "url(#arrowhead)");

    const linkLabel = linkGroup
      .selectAll("text")
      .data(d3Links)
      .join("text")
      .attr("fill", "rgba(255,255,255,0.35)")
      .attr("font-size", "8px")
      .attr("text-anchor", "middle")
      .text((d: D3Link) => d.label);

    const nodeGroup = graphGroup.append("g");
    const node = nodeGroup
      .selectAll<SVGGElement, D3Node>("g")
      .data(d3Nodes)
      .join("g")
      .attr("role", "button")
      .attr("tabindex", "0")
      .attr("aria-label", (d: D3Node) => `${d.label} — click to view details`)
      .style("cursor", "pointer")
      .on("click", (_event: MouseEvent, d: D3Node) => onNodeClick(d.movie))
      .on("keydown", (event: KeyboardEvent, d: D3Node) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onNodeClick(d.movie);
        }
      });

    const drag = d3
      .drag<SVGGElement, D3Node>()
      .on("start", (event: d3.D3DragEvent<SVGGElement, D3Node, D3Node>, d: D3Node) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on("drag", (event: d3.D3DragEvent<SVGGElement, D3Node, D3Node>, d: D3Node) => {
        d.fx = event.x; d.fy = event.y;
      })
      .on("end", (event: d3.D3DragEvent<SVGGElement, D3Node, D3Node>, d: D3Node) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
      });

    node.call(drag as never);

    // Background circle
    node.append("circle")
      .attr("r", (d: D3Node) => (d.type === "seed" ? 22 : 16))
      .attr("fill", (d: D3Node) => d.type === "seed" ? "rgba(229,9,20,0.9)" : "rgba(22,24,34,0.95)")
      .attr("stroke", (d: D3Node) => d.type === "seed" ? "#e50914" : "rgba(255,255,255,0.18)")
      .attr("stroke-width", (d: D3Node) => (d.type === "seed" ? 2.5 : 1.5));

    // Poster thumbnail (clipped to circle)
    node.each(function(d: D3Node) {
      const url = posterUrl(d.movie.poster_path);
      if (!url) return;
      const r = d.type === "seed" ? 20 : 14;
      const clipId = `clip-${d.id.replace(/[^a-zA-Z0-9]/g, "-")}`;
      // Add per-node clip path
      defs.append("clipPath")
        .attr("id", clipId)
        .append("circle")
        .attr("r", r);

      d3.select(this).append("image")
        .attr("href", url)
        .attr("x", -r)
        .attr("y", -r)
        .attr("width", r * 2)
        .attr("height", r * 2)
        .attr("clip-path", `url(#${clipId})`)
        .attr("preserveAspectRatio", "xMidYMid slice");
    });

    // Label below node
    node.append("text")
      .attr("dy", (d: D3Node) => (d.type === "seed" ? 38 : 30))
      .attr("text-anchor", "middle")
      .attr("fill", "#e8ecf5")
      .attr("font-size", (d: D3Node) => (d.type === "seed" ? "11px" : "9px"))
      .attr("font-weight", (d: D3Node) => (d.type === "seed" ? "800" : "500"))
      .text((d: D3Node) => (d.label.length > 16 ? `${d.label.slice(0, 14)}…` : d.label));

    simulation.on("tick", () => {
      link
        .attr("x1", (d: D3Link) => (d.source as D3Node).x ?? 0)
        .attr("y1", (d: D3Link) => (d.source as D3Node).y ?? 0)
        .attr("x2", (d: D3Link) => (d.target as D3Node).x ?? 0)
        .attr("y2", (d: D3Link) => (d.target as D3Node).y ?? 0);

      linkLabel
        .attr("x", (d: D3Link) => (((d.source as D3Node).x ?? 0) + ((d.target as D3Node).x ?? 0)) / 2)
        .attr("y", (d: D3Link) => (((d.source as D3Node).y ?? 0) + ((d.target as D3Node).y ?? 0)) / 2);

      node.attr("transform", (d: D3Node) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => { simulation.stop(); };
  }, [nodes, edges, onNodeClick]);

  return (
    <div ref={containerRef} className="kg-graph-container" aria-label="Knowledge graph">
      <svg ref={svgRef} className="kg-svg" />
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
        {selectedId && error && <p className="dashboard-error" role="alert">{error}</p>}
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
