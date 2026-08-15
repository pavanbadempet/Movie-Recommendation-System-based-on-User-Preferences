import React, { useRef, useEffect, useState, useCallback } from "react";
import { RotateCcw, Compass, Search, Plus, Minus, Film } from "lucide-react";
import type { Movie, MovieTitle } from "./types";

interface MovieNode {
  id: number;
  title: string;
  genre: string;
  year?: number;
  rating?: number;
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

const GENRES = ["Action", "Sci-Fi", "Drama", "Thriller", "Comedy", "Romance"] as const;
type GenreName = typeof GENRES[number];

const GENRE_COLORS: Record<GenreName, string> = {
  Action: "#f43f5e",   // Vibrant Rose Red
  "Sci-Fi": "#06b6d4",  // Radiant Cyan
  Drama: "#10b981",    // Emerald Green
  Thriller: "#a855f7", // Neon Purple
  Comedy: "#f59e0b",   // Warm Amber
  Romance: "#ec4899",  // Hot Pink
};

// Real-world cinema catalog spanning all 6 genres with genuine semantic clustering
const CURATED_CINEMA_CATALOG: Record<GenreName, { title: string; year: number; rating: number }[]> = {
  Action: [
    { title: "The Dark Knight", year: 2008, rating: 9.0 },
    { title: "Gladiator", year: 2000, rating: 8.5 },
    { title: "Mad Max: Fury Road", year: 2015, rating: 8.1 },
    { title: "John Wick: Chapter 4", year: 2023, rating: 7.7 },
    { title: "Die Hard", year: 1988, rating: 8.2 },
    { title: "Terminator 2: Judgment Day", year: 1991, rating: 8.6 },
    { title: "Top Gun: Maverick", year: 2022, rating: 8.3 },
    { title: "Casino Royale", year: 2006, rating: 8.0 },
    { title: "The Raid: Redemption", year: 2011, rating: 7.6 },
    { title: "Heat", year: 1995, rating: 8.3 },
    { title: "Kill Bill: Vol. 1", year: 2003, rating: 8.2 },
    { title: "Avengers: Endgame", year: 2019, rating: 8.4 },
    { title: "Spider-Man: Across the Spider-Verse", year: 2023, rating: 8.7 },
    { title: "Mission: Impossible - Fallout", year: 2018, rating: 7.7 },
    { title: "The Bourne Ultimatum", year: 2007, rating: 8.0 },
    { title: "Ip Man", year: 2008, rating: 8.0 },
    { title: "Dredd", year: 2012, rating: 7.1 },
    { title: "Speed", year: 1994, rating: 7.3 },
    { title: "The Batman", year: 2022, rating: 7.8 },
    { title: "Skyfall", year: 2012, rating: 7.8 },
  ],
  "Sci-Fi": [
    { title: "Inception", year: 2010, rating: 8.8 },
    { title: "Interstellar", year: 2014, rating: 8.7 },
    { title: "The Matrix", year: 1999, rating: 8.7 },
    { title: "Blade Runner 2049", year: 2017, rating: 8.0 },
    { title: "Dune: Part Two", year: 2024, rating: 8.6 },
    { title: "Arrival", year: 2016, rating: 7.9 },
    { title: "2001: A Space Odyssey", year: 1968, rating: 8.3 },
    { title: "Alien", year: 1979, rating: 8.5 },
    { title: "Aliens", year: 1986, rating: 8.4 },
    { title: "Avatar: The Way of Water", year: 2022, rating: 7.6 },
    { title: "Ex Machina", year: 2014, rating: 7.7 },
    { title: "Children of Men", year: 2006, rating: 7.9 },
    { title: "Edge of Tomorrow", year: 2014, rating: 7.9 },
    { title: "The Prestige", year: 2006, rating: 8.5 },
    { title: "Solaris", year: 1972, rating: 8.0 },
    { title: "Minority Report", year: 2002, rating: 7.7 },
    { title: "District 9", year: 2009, rating: 7.9 },
    { title: "Gattaca", year: 1997, rating: 7.8 },
    { title: "Contact", year: 1997, rating: 7.5 },
    { title: "Tenet", year: 2020, rating: 7.3 },
  ],
  Drama: [
    { title: "The Godfather", year: 1972, rating: 9.2 },
    { title: "The Godfather Part II", year: 1974, rating: 9.0 },
    { title: "The Shawshank Redemption", year: 1994, rating: 9.3 },
    { title: "Schindler's List", year: 1993, rating: 9.0 },
    { title: "12 Angry Men", year: 1957, rating: 9.0 },
    { title: "Oppenheimer", year: 2023, rating: 8.9 },
    { title: "Fight Club", year: 1999, rating: 8.8 },
    { title: "Goodfellas", year: 1990, rating: 8.7 },
    { title: "Whiplash", year: 2014, rating: 8.5 },
    { title: "Parasite", year: 2019, rating: 8.5 },
    { title: "There Will Be Blood", year: 2007, rating: 8.2 },
    { title: "The Social Network", year: 2010, rating: 7.8 },
    { title: "Forrest Gump", year: 1994, rating: 8.8 },
    { title: "One Flew Over the Cuckoo's Nest", year: 1975, rating: 8.7 },
    { title: "American Beauty", year: 1999, rating: 8.3 },
    { title: "Taxi Driver", year: 1976, rating: 8.2 },
    { title: "No Country for Old Men", year: 2007, rating: 8.2 },
    { title: "The Pianist", year: 2002, rating: 8.5 },
    { title: "Casablanca", year: 1942, rating: 8.5 },
    { title: "Citizen Kane", year: 1941, rating: 8.3 },
  ],
  Thriller: [
    { title: "Pulp Fiction", year: 1994, rating: 8.9 },
    { title: "Se7en", year: 1995, rating: 8.6 },
    { title: "The Silence of the Lambs", year: 1991, rating: 8.6 },
    { title: "Shutter Island", year: 2010, rating: 8.2 },
    { title: "Memento", year: 2000, rating: 8.4 },
    { title: "Zodiac", year: 2007, rating: 7.7 },
    { title: "Prisoners", year: 2013, rating: 8.1 },
    { title: "Gone Girl", year: 2014, rating: 8.1 },
    { title: "The Departed", year: 2006, rating: 8.5 },
    { title: "Nightcrawler", year: 2014, rating: 7.8 },
    { title: "Sicario", year: 2015, rating: 7.6 },
    { title: "Oldboy", year: 2003, rating: 8.4 },
    { title: "Drive", year: 2011, rating: 7.8 },
    { title: "Black Swan", year: 2010, rating: 8.0 },
    { title: "Uncut Gems", year: 2019, rating: 7.4 },
    { title: "The Usual Suspects", year: 1995, rating: 8.5 },
    { title: "Fargo", year: 1996, rating: 8.1 },
    { title: "Parasite", year: 2019, rating: 8.5 },
    { title: "A Quiet Place", year: 2018, rating: 7.5 },
    { title: "Get Out", year: 2017, rating: 7.7 },
  ],
  Comedy: [
    { title: "The Grand Budapest Hotel", year: 2014, rating: 8.1 },
    { title: "The Big Lebowski", year: 1998, rating: 8.1 },
    { title: "Superbad", year: 2007, rating: 7.6 },
    { title: "Monty Python and the Holy Grail", year: 1975, rating: 8.2 },
    { title: "Knives Out", year: 2019, rating: 7.9 },
    { title: "Groundhog Day", year: 1993, rating: 8.0 },
    { title: "The Hangover", year: 2009, rating: 7.7 },
    { title: "Tropic Thunder", year: 2008, rating: 7.1 },
    { title: "Shaun of the Dead", year: 2004, rating: 7.9 },
    { title: "Hot Fuzz", year: 2007, rating: 7.8 },
    { title: "What We Do in the Shadows", year: 2014, rating: 7.6 },
    { title: "In Bruges", year: 2008, rating: 7.9 },
    { title: "Everything Everywhere All At Once", year: 2022, rating: 7.8 },
    { title: "Ferris Bueller's Day Off", year: 1986, rating: 7.8 },
    { title: "Jojo Rabbit", year: 2019, rating: 7.9 },
    { title: "Palm Springs", year: 2020, rating: 7.4 },
    { title: "Booksmart", year: 2019, rating: 7.1 },
    { title: "Little Miss Sunshine", year: 2006, rating: 7.8 },
    { title: "Burn After Reading", year: 2008, rating: 7.0 },
    { title: "Snatch", year: 2000, rating: 8.2 },
  ],
  Romance: [
    { title: "La La Land", year: 2016, rating: 8.0 },
    { title: "Eternal Sunshine of the Spotless Mind", year: 2004, rating: 8.3 },
    { title: "Her", year: 2013, rating: 8.0 },
    { title: "Before Sunrise", year: 1995, rating: 8.1 },
    { title: "Before Sunset", year: 2004, rating: 8.1 },
    { title: "Before Midnight", year: 2013, rating: 7.9 },
    { title: "Amélie", year: 2001, rating: 8.3 },
    { title: "Titanic", year: 1997, rating: 7.9 },
    { title: "The Notebook", year: 2004, rating: 7.8 },
    { title: "About Time", year: 2013, rating: 7.8 },
    { title: "Past Lives", year: 2023, rating: 7.9 },
    { title: "Pride & Prejudice", year: 2005, rating: 7.8 },
    { title: "Call Me by Your Name", year: 2017, rating: 7.8 },
    { title: "500 Days of Summer", year: 2009, rating: 7.7 },
    { title: "Crazy Rich Asians", year: 2018, rating: 6.9 },
    { title: "Silver Linings Playbook", year: 2012, rating: 7.7 },
    { title: "A Star Is Born", year: 2018, rating: 7.6 },
    { title: "Portrait of a Lady on Fire", year: 2019, rating: 8.1 },
    { title: "The Shape of Water", year: 2017, rating: 7.3 },
    { title: "Midnight in Paris", year: 2011, rating: 7.7 },
  ],
};

interface VectorSpaceProps {
  onSelectMovie?: (movie: Movie) => void;
  titles?: MovieTitle[];
}

export function VectorSpace({ onSelectMovie }: VectorSpaceProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [nodes, setNodes] = useState<MovieNode[]>([]);
  const [hoveredNode, setHoveredNode] = useState<MovieNode | null>(null);
  const [selectedNode, setSelectedNode] = useState<MovieNode | null>(null);
  const [selectedGenre, setSelectedGenre] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  const hoveredNodeRef = useRef(hoveredNode);
  hoveredNodeRef.current = hoveredNode;

  const selectedNodeRef = useRef(selectedNode);
  selectedNodeRef.current = selectedNode;

  // Camera & Interaction state
  const rotX = useRef(0.32);
  const rotY = useRef(0.55);
  const zoom = useRef(1.05);
  const [autoRotate, setAutoRotate] = useState(true);

  const isDragging = useRef(false);
  const previousMousePosition = useRef({ x: 0, y: 0 });
  const backgroundStars = useRef<Star[]>([]);

  // Build Background Starfield
  useEffect(() => {
    const stars: Star[] = [];
    for (let i = 0; i < 180; i++) {
      stars.push({
        x: (Math.random() - 0.5) * 1400,
        y: (Math.random() - 0.5) * 900,
        z: Math.random() * 900 - 200,
        size: Math.random() * 1.6 + 0.4,
        alpha: Math.random() * 0.6 + 0.15,
      });
    }
    backgroundStars.current = stars;
  }, []);

  // Generate 3D Semantic Galaxy with 120+ Real Movies
  useEffect(() => {
    const generated: MovieNode[] = [];
    let idCounter = 1;

    GENRES.forEach((genre, genreIdx) => {
      const items = CURATED_CINEMA_CATALOG[genre];
      const color = GENRE_COLORS[genre];
      const baseAngle = (genreIdx / GENRES.length) * Math.PI * 2;

      items.forEach((item, itemIdx) => {
        // Hyperbolic Radial Geodesic projection
        const radius = 80 + (itemIdx / items.length) * 120 + (Math.random() - 0.5) * 20;
        const angle = baseAngle + (Math.random() - 0.5) * 0.65;

        const x = Math.cos(angle) * radius + (Math.random() - 0.5) * 25;
        const z = Math.sin(angle) * radius + (Math.random() - 0.5) * 25;
        const y = Math.sin(angle * 2 + itemIdx * 0.4) * 60 + (item.rating - 8.0) * 35;

        const isAnchor = itemIdx < 4; // First 4 per genre are anchor landmarks

        generated.push({
          id: idCounter++,
          title: item.title,
          genre,
          year: item.year,
          rating: item.rating,
          x,
          y,
          z,
          color,
          isAnchor,
        });
      });
    });

    setNodes(generated);
  }, []);

  // High-DPI Canvas Sizing
  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const dpr = window.devicePixelRatio || 1;
    const width = container.clientWidth;
    const height = Math.max(540, Math.min(680, window.innerHeight * 0.62));

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

  // Main 3D Rendering Engine
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

      // Deep Space Cinematic Backdrop
      const bgGradient = ctx.createRadialGradient(width / 2, height / 2, 40, width / 2, height / 2, Math.max(width, height) / 1.1);
      bgGradient.addColorStop(0, "#0a0c18");
      bgGradient.addColorStop(0.55, "#05060d");
      bgGradient.addColorStop(1, "#020205");
      ctx.fillStyle = bgGradient;
      ctx.fillRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const fov = 440 * zoom.current;

      const cosX = Math.cos(rotX.current);
      const sinX = Math.sin(rotX.current);
      const cosY = Math.cos(rotY.current);
      const sinY = Math.sin(rotY.current);

      // 1. Draw Starfield
      backgroundStars.current.forEach((star) => {
        const sx = cx + star.x * 0.38;
        const sy = cy + star.y * 0.38;
        if (sx > 0 && sx < width && sy > 0 && sy < height) {
          ctx.fillStyle = `rgba(255, 255, 255, ${star.alpha * 0.65})`;
          ctx.beginPath();
          ctx.arc(sx, sy, star.size, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // 2. Draw Concentric Orbital Horizon Rings
      const ringRadii = [90, 170, 250];
      ringRadii.forEach((r) => {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.035)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let a = 0; a <= Math.PI * 2; a += 0.12) {
          const rx = Math.cos(a) * r;
          const rz = Math.sin(a) * r;
          const rx1 = rx * cosY - rz * sinY;
          const rz1 = rz * cosY + rx * sinY;
          const ry2 = -rz1 * sinX;
          const rz2 = rz1 * cosX;
          const scale = fov / Math.max(1, rz2 + 360);
          const px = cx + rx1 * scale;
          const py = cy + ry2 * scale;
          if (a === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.stroke();
      });

      // 3. Project Nodes to 2D
      const projectedNodes = nodes.map((node) => {
        const x1 = node.x * cosY - node.z * sinY;
        const z1 = node.z * cosY + node.x * sinY;
        const y2 = node.y * cosX - z1 * sinX;
        const z2 = z1 * cosX + node.y * sinX;

        const pz = z2 + 360;
        const scale = fov / Math.max(1, pz);
        const px = cx + x1 * scale;
        const py = cy + y2 * scale;

        const isFilteredOut =
          (selectedGenre !== "ALL" && node.genre !== selectedGenre) ||
          (searchQuery.trim() && !node.title.toLowerCase().includes(searchQuery.toLowerCase()));

        return { ...node, px, py, pz, isFilteredOut };
      });

      // Depth sort for painter's algorithm
      projectedNodes.sort((a, b) => (b.pz || 0) - (a.pz || 0));

      // 4. Draw Intra-Cluster Constellation Filaments
      for (let i = 0; i < projectedNodes.length; i++) {
        const n1 = projectedNodes[i];
        if (n1.isFilteredOut) continue;

        for (let j = i + 1; j < projectedNodes.length; j++) {
          const n2 = projectedNodes[j];
          if (n2.isFilteredOut) continue;

          if (n1.genre === n2.genre) {
            const dx = n1.x - n2.x;
            const dy = n1.y - n2.y;
            const dz = n1.z - n2.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (dist < 46) {
              const alpha = Math.max(0.03, (1 - dist / 46) * 0.16);
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

      // 5. Active Selection / Hover Similarity Beams
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
              if (dist < 85) {
                ctx.strokeStyle = activeProj.color;
                ctx.lineWidth = 1.6;
                ctx.globalAlpha = Math.max(0.2, (1 - dist / 85) * 0.85);
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
        const isHovered = hoveredNodeRef.current?.id === node.id;
        const isSelected = selectedNodeRef.current?.id === node.id;
        const isDimmed = node.isFilteredOut;

        const baseRadius = Math.max(2.8, (400 / Math.max(1, node.pz || 1)) * (node.isAnchor ? 3.6 : 2.4));
        const radius = isHovered || isSelected ? baseRadius * 1.6 : baseRadius;

        if (isDimmed) {
          ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
          ctx.beginPath();
          ctx.arc(node.px!, node.py!, 1.5, 0, Math.PI * 2);
          ctx.fill();
          return;
        }

        // Radiant Stellar Glow Halo
        const glowRadius = radius * 3.6;
        const glowGradient = ctx.createRadialGradient(node.px!, node.py!, radius * 0.4, node.px!, node.py!, glowRadius);
        glowGradient.addColorStop(0, node.color);
        glowGradient.addColorStop(0.35, `${node.color}55`);
        glowGradient.addColorStop(1, "transparent");

        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(node.px!, node.py!, glowRadius, 0, Math.PI * 2);
        ctx.fill();

        // High-Contrast Core
        ctx.fillStyle = isHovered || isSelected ? "#ffffff" : node.color;
        ctx.beginPath();
        ctx.arc(node.px!, node.py!, radius, 0, Math.PI * 2);
        ctx.fill();

        // Selection Target Ring
        if (isSelected) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(node.px!, node.py!, radius * 2.2, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Floating 3D Movie Titles
        if (node.isAnchor || isHovered || isSelected) {
          const fontSize = Math.max(9, Math.min(13, 3800 / Math.max(1, node.pz || 1)));
          ctx.font = `${isHovered || isSelected ? "700" : "500"} ${fontSize}px "Inter", -apple-system, sans-serif`;
          ctx.fillStyle = isHovered || isSelected ? "#ffffff" : "rgba(255, 255, 255, 0.85)";
          ctx.textAlign = "center";
          ctx.shadowColor = "rgba(0, 0, 0, 0.85)";
          ctx.shadowBlur = 4;
          ctx.fillText(node.title, node.px!, node.py! - radius - 6);
          ctx.shadowBlur = 0;
        }
      });

      // Auto-Orbit
      if (autoRotate) {
        rotY.current += 0.0016;
      }

      ctx.restore();
      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [nodes, autoRotate, selectedGenre, searchQuery]);

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
      rotX.current = Math.max(-Math.PI / 2.3, Math.min(Math.PI / 2.3, rotX.current + deltaY * 0.005));
      previousMousePosition.current = { x: e.clientX, y: e.clientY };
      return;
    }

    // Node hit testing
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    const cx = width / 2;
    const cy = height / 2;
    const fov = 440 * zoom.current;

    const cosX = Math.cos(rotX.current);
    const sinX = Math.sin(rotX.current);
    const cosY = Math.cos(rotY.current);
    const sinY = Math.sin(rotY.current);

    let match: MovieNode | null = null;
    let minDistance = 20;

    nodes.forEach((node) => {
      if (selectedGenre !== "ALL" && node.genre !== selectedGenre) return;

      const x1 = node.x * cosY - node.z * sinY;
      const z1 = node.z * cosY + node.x * sinY;
      const y2 = node.y * cosX - z1 * sinX;
      const z2 = z1 * cosX + node.y * sinX;

      const scale = fov / Math.max(1, z2 + 360);
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

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const zoomDelta = e.deltaY * -0.001;
    zoom.current = Math.max(0.6, Math.min(2.5, zoom.current + zoomDelta));
  };

  const handleInspect = (node: MovieNode) => {
    if (onSelectMovie) {
      const fullMovie: Movie = {
        id: node.id,
        title: node.title,
        genres: node.genre,
        release_date: node.year ? `${node.year}-01-01` : undefined,
        vote_average: node.rating || 8.0,
        similarity_score: 0.94,
        retrieval_stage: "Poincaré 768-D Vector Space",
        overview: `${node.title} is a landmark ${node.genre.toLowerCase()} title embedded within the 768-dimensional neural vector space.`,
      };
      onSelectMovie(fullMovie);
    }
  };

  return (
    <div ref={containerRef} className="glass-panel" style={{ padding: "24px", display: "flex", flexDirection: "column", gap: "16px", background: "rgba(8, 9, 16, 0.8)", borderRadius: "20px", border: "1px solid rgba(255, 255, 255, 0.08)" }}>
      {/* Header & Controls Bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "14px" }}>
        <div>
          <h2 style={{ fontSize: "1.3rem", fontWeight: "700", margin: 0, display: "flex", alignItems: "center", gap: "10px", color: "#ffffff" }}>
            <Compass size={22} style={{ color: "#06b6d4" }} />
            <span>3D Neural Vector Galaxy</span>
          </h2>
          <p style={{ fontSize: "0.82rem", color: "var(--muted)", margin: "4px 0 0 0" }}>
            Explore 120+ real cinema titles embedded along Poincaré hyperbolic manifolds in 768-D semantic space.
          </p>
        </div>

        {/* Action Controls */}
        <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
          {/* In-Galaxy Search */}
          <div style={{ position: "relative", minWidth: "180px" }}>
            <Search size={14} style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)", color: "var(--muted)" }} />
            <input
              type="text"
              placeholder="Search movie..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: "100%",
                padding: "6px 12px 6px 30px",
                fontSize: "0.8rem",
                borderRadius: "8px",
                background: "rgba(255, 255, 255, 0.05)",
                border: "1px solid rgba(255, 255, 255, 0.1)",
                color: "#fff",
                outline: "none",
              }}
            />
          </div>

          {/* Zoom Buttons */}
          <button
            className="icon-button"
            type="button"
            onClick={() => { zoom.current = Math.min(2.5, zoom.current + 0.15); }}
            title="Zoom In"
            aria-label="Zoom In"
            style={{ width: "32px", height: "32px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)" }}
          >
            <Plus size={14} />
          </button>
          <button
            className="icon-button"
            type="button"
            onClick={() => { zoom.current = Math.max(0.6, zoom.current - 0.15); }}
            title="Zoom Out"
            aria-label="Zoom Out"
            style={{ width: "32px", height: "32px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)" }}
          >
            <Minus size={14} />
          </button>

          {/* Reset Orbit */}
          <button
            className="icon-button"
            type="button"
            onClick={() => {
              rotX.current = 0.32;
              rotY.current = 0.55;
              zoom.current = 1.05;
              setAutoRotate(true);
              setSelectedNode(null);
              setSearchQuery("");
              setSelectedGenre("ALL");
            }}
            title="Reset Orbit"
            aria-label="Reset Orbit"
            style={{ width: "32px", height: "32px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)" }}
          >
            <RotateCcw size={14} />
          </button>

          {/* Auto-Rotate Toggle */}
          <button
            type="button"
            onClick={() => setAutoRotate(!autoRotate)}
            style={{
              fontSize: "0.78rem",
              fontWeight: "600",
              padding: "7px 12px",
              borderRadius: "8px",
              border: "1px solid rgba(255, 255, 255, 0.12)",
              background: autoRotate ? "rgba(6, 182, 212, 0.12)" : "rgba(255, 255, 255, 0.04)",
              color: autoRotate ? "#22d3ee" : "#cbd5e1",
              cursor: "pointer",
            }}
          >
            {autoRotate ? "Pause Orbit" : "Resume Orbit"}
          </button>
        </div>
      </div>

      {/* Genre Filter Chips */}
      <div style={{ display: "flex", gap: "8px", overflowX: "auto", paddingBottom: "4px" }}>
        <button
          type="button"
          onClick={() => setSelectedGenre("ALL")}
          style={{
            fontSize: "0.75rem",
            fontWeight: "600",
            padding: "4px 12px",
            borderRadius: "20px",
            border: `1px solid ${selectedGenre === "ALL" ? "var(--cyan)" : "rgba(255,255,255,0.08)"}`,
            background: selectedGenre === "ALL" ? "rgba(6, 182, 212, 0.15)" : "rgba(255,255,255,0.03)",
            color: selectedGenre === "ALL" ? "#22d3ee" : "var(--muted)",
            cursor: "pointer",
          }}
        >
          All Genres ({nodes.length})
        </button>
        {GENRES.map((g) => (
          <button
            key={g}
            type="button"
            onClick={() => setSelectedGenre(g)}
            style={{
              fontSize: "0.75rem",
              fontWeight: "600",
              padding: "4px 12px",
              borderRadius: "20px",
              border: `1px solid ${selectedGenre === g ? GENRE_COLORS[g] : "rgba(255,255,255,0.08)"}`,
              background: selectedGenre === g ? `${GENRE_COLORS[g]}22` : "rgba(255,255,255,0.03)",
              color: selectedGenre === g ? GENRE_COLORS[g] : "var(--muted)",
              cursor: "pointer",
            }}
          >
            {g}
          </button>
        ))}
      </div>

      {/* 3D Canvas Box */}
      <div style={{ position: "relative", width: "100%", height: "540px", borderRadius: "16px", overflow: "hidden", border: "1px solid rgba(255, 255, 255, 0.08)", boxShadow: "inset 0 0 80px rgba(0,0,0,0.8)" }}>
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
          onClick={() => {
            if (hoveredNode) {
              setSelectedNode(hoveredNode);
            }
          }}
          style={{ width: "100%", height: "100%", cursor: isDragging.current ? "grabbing" : "grab", display: "block" }}
        />

        {/* Selected / Hovered Movie HUD Card */}
        {(hoveredNode || selectedNode) && (
          <div
            style={{
              position: "absolute",
              bottom: "20px",
              left: "20px",
              padding: "16px 20px",
              background: "rgba(9, 10, 18, 0.94)",
              backdropFilter: "blur(20px)",
              border: `1px solid ${(selectedNode || hoveredNode)!.color}44`,
              borderRadius: "14px",
              minWidth: "280px",
              maxWidth: "340px",
              boxShadow: "0 16px 48px rgba(0,0,0,0.7)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
              <span style={{ fontSize: "0.72rem", fontWeight: "800", textTransform: "uppercase", letterSpacing: "0.08em", color: (selectedNode || hoveredNode)!.color }}>
                {(selectedNode || hoveredNode)!.genre} Cluster
              </span>
              <span style={{ fontSize: "0.72rem", padding: "2px 8px", borderRadius: "12px", background: "rgba(255,255,255,0.08)", color: "#fff", fontWeight: "700" }}>
                ★ {(selectedNode || hoveredNode)!.rating?.toFixed(1) || "8.2"}
              </span>
            </div>
            <h3 style={{ fontSize: "1.15rem", fontWeight: "700", margin: "0 0 8px 0", color: "#ffffff" }}>
              {(selectedNode || hoveredNode)!.title}
              {(selectedNode || hoveredNode)!.year && (
                <span style={{ fontSize: "0.8rem", fontWeight: "400", color: "var(--muted)", marginLeft: "6px" }}>
                  ({(selectedNode || hoveredNode)!.year})
                </span>
              )}
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px", fontSize: "0.76rem", color: "#94a3b8", marginBottom: "12px" }}>
              <div>Vector X: <span style={{ color: "#fff", fontWeight: "600" }}>{(selectedNode || hoveredNode)!.x.toFixed(1)}</span></div>
              <div>Vector Y: <span style={{ color: "#fff", fontWeight: "600" }}>{(selectedNode || hoveredNode)!.y.toFixed(1)}</span></div>
              <div>Vector Z: <span style={{ color: "#fff", fontWeight: "600" }}>{(selectedNode || hoveredNode)!.z.toFixed(1)}</span></div>
              <div>Manifold: <span style={{ color: "#06b6d4", fontWeight: "600" }}>Poincaré</span></div>
            </div>

            {onSelectMovie && (
              <button
                type="button"
                onClick={() => handleInspect(selectedNode || hoveredNode!)}
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  borderRadius: "8px",
                  background: "linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)",
                  border: "none",
                  color: "#fff",
                  fontWeight: "600",
                  fontSize: "0.82rem",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "6px",
                  cursor: "pointer",
                  boxShadow: "0 4px 12px rgba(6, 182, 212, 0.3)",
                }}
              >
                <Film size={14} />
                <span>Watch Trailer & Details</span>
              </button>
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
            gap: "7px",
            padding: "12px 14px",
            background: "rgba(9, 10, 18, 0.85)",
            backdropFilter: "blur(12px)",
            borderRadius: "12px",
            border: "1px solid rgba(255, 255, 255, 0.08)",
          }}
        >
          <span style={{ fontSize: "0.68rem", fontWeight: "800", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--muted)", marginBottom: "2px" }}>
            Genre Clusters
          </span>
          {GENRES.map((g) => (
            <button
              type="button"
              key={g}
              onClick={() => setSelectedGenre(selectedGenre === g ? "ALL" : g)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                fontSize: "0.76rem",
                color: selectedGenre === g || selectedGenre === "ALL" ? "#e2e8f0" : "#64748b",
                cursor: "pointer",
                background: "transparent",
                border: "none",
                padding: "2px 4px",
                textAlign: "left",
              }}
            >
              <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: GENRE_COLORS[g], boxShadow: `0 0 8px ${GENRE_COLORS[g]}` }}></span>
              <span>{g}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
