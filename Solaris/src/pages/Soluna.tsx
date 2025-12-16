import React, { useState } from "react";
import axios from "axios";
import { BACKEND_URL } from "../config/env";

// Define types based on the backend response
interface AnimeTitle {
  romaji: string;
  english: string | null;
}

interface AnimeCoverImage {
  large: string;
}

interface Anime {
  id: number;
  title: AnimeTitle;
  description: string;
  genres: string[];
  averageScore: number;
  status: string;
  season: string;
  seasonYear: number;
  coverImage: AnimeCoverImage;
}

interface SearchResponse {
  results: Anime[];
}

export const Soluna = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Anime[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${BACKEND_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SearchResponse = await response.json();
      setResults(data.results);
    } catch (err) {
      console.error("Search failed:", err);
      setError(
        "Failed to fetch data from backend. Please ensure Lunaris is running.",
      );
    }
  };

  return (
    <div
      style={{
        padding: "2rem",
        fontFamily: "Arial, sans-serif",
        maxWidth: "1200px",
        margin: "0 auto",
      }}
    >
      <header style={{ marginBottom: "2rem", textAlign: "center" }}>
        <h1 style={{ color: "#333" }}>Soluna Anime Search</h1>
        <p style={{ color: "#666" }}>Powered by AniList & MCP</p>
      </header>

      <form
        onSubmit={handleSearch}
        style={{
          display: "flex",
          gap: "10px",
          justifyContent: "center",
          marginBottom: "2rem",
        }}
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter anime name (e.g., Frieren)"
          style={{
            padding: "10px",
            fontSize: "16px",
            width: "300px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            color: "#222",
            background: "#fff",
          }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "10px 20px",
            fontSize: "16px",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error && (
        <div
          style={{ color: "red", textAlign: "center", marginBottom: "1rem" }}
        >
          {error}
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))",
          gap: "20px",
        }}
      >
        {results.map((anime) => (
          <div
            key={anime.id}
            style={{
              border: "1px solid #eee",
              borderRadius: "8px",
              overflow: "hidden",
              boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
              transition: "transform 0.2s",
            }}
          >
            <img
              src={anime.coverImage.large}
              alt={anime.title.romaji}
              style={{ width: "100%", height: "350px", objectFit: "cover" }}
            />
            <div style={{ padding: "1rem" }}>
              <h3
                style={{
                  margin: "0 0 0.5rem 0",
                  fontSize: "1.1rem",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {anime.title.english || anime.title.romaji}
              </h3>
              <p style={{ fontSize: "0.9rem", color: "#555", margin: "0" }}>
                {anime.season} {anime.seasonYear}
              </p>
              <div
                style={{
                  marginTop: "0.5rem",
                  display: "flex",
                  flexWrap: "wrap",
                  gap: "4px",
                }}
              >
                {anime.genres.slice(0, 3).map((genre) => (
                  <span
                    key={genre}
                    style={{
                      backgroundColor: "#f0f0f0",
                      padding: "2px 6px",
                      borderRadius: "4px",
                      fontSize: "0.8rem",
                      color: "#666",
                    }}
                  >
                    {genre}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      {!loading && results.length === 0 && query && !error && (
        <p style={{ textAlign: "center", color: "#888" }}>No results found.</p>
      )}
    </div>
  );
};
