"use client";

import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

type MetaResponse = {
  documents: { total: number; by_language: Record<string, number> };
};

type Warning = {
  threshold: number;
  top_score: number;
  message: string;
};

type Timing = Record<string, number>;

type SearchResult = {
  rank: number;
  score: number;
  title: string;
  url: string;
  date: string | null;
  language: string;
  source: string;
  snippet: string;
  debug: Record<string, number>;
};

type SearchResponse = {
  query: string;
  query_variants: { en: string; bn: string };
  k: number;
  language_filter: string;
  timing_ms: Timing;
  warning: Warning | null;
  results: SearchResult[];
};

export default function HomePage() {
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [query, setQuery] = useState("");
  const [langFilter, setLangFilter] = useState("all");
  const [k, setK] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [showDebug, setShowDebug] = useState(false);

  useEffect(() => {
    const loadMeta = async () => {
      try {
        const res = await fetch(`${API_BASE}/meta`);
        if (res.ok) {
          const data = (await res.json()) as MetaResponse;
          setMeta(data);
        }
      } catch (err) {
        console.warn("Failed to load meta", err);
      }
    };
    loadMeta();
  }, []);

  const handleSearch = async () => {
    if (!query.trim()) {
      setError("Enter a query");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const params = new URLSearchParams({
        q: query.trim(),
        lang: langFilter,
        k: String(k),
      });
      const res = await fetch(`${API_BASE}/search?${params.toString()}`);
      if (!res.ok) {
        throw new Error(`Search failed (${res.status})`);
      }
      const data = (await res.json()) as SearchResponse;
      setResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  };

  const timingEntries = useMemo(() => {
    if (!response) return [];
    return Object.entries(response.timing_ms);
  }, [response]);

  return (
    <main className="page">
      <section className="card">
        <h1>CLIR Search</h1>
        <p>
          Cross-lingual news search across Bangla and English corpora. Type a query in either language and compare how
          lexical, semantic, and fuzzy models respond.
        </p>
        {meta && (
          <div style={{ marginTop: "1rem", display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            <div>
              <strong>Total docs:</strong> {meta.documents.total.toLocaleString()}
            </div>
            {Object.entries(meta.documents.by_language).map(([lang, count]) => (
              <div key={lang}>
                <strong>{lang.toUpperCase()}:</strong> {count.toLocaleString()}
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="card">
        <form
          onSubmit={(ev) => {
            ev.preventDefault();
            handleSearch();
          }}
        >
          <label htmlFor="query-input" style={{ display: "block", fontWeight: 600, marginBottom: 8 }}>
            Query
          </label>
          <textarea
            id="query-input"
            placeholder="Bangladesh election results 2026"
            value={query}
            onChange={(ev) => setQuery(ev.target.value)}
            style={{
              width: "100%",
              minHeight: 90,
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.15)",
              padding: "0.75rem",
              background: "#0f172a",
              color: "#f8fafc",
              resize: "vertical",
            }}
          />

          <div style={{ display: "flex", gap: "1rem", marginTop: "1rem", flexWrap: "wrap" }}>
            <label style={{ flex: "1 1 200px" }}>
              <span style={{ display: "block", fontWeight: 600, marginBottom: 4 }}>Language filter</span>
              <select
                value={langFilter}
                onChange={(ev) => setLangFilter(ev.target.value)}
                style={{
                  width: "100%",
                  borderRadius: 8,
                  border: "1px solid rgba(255,255,255,0.15)",
                  padding: "0.55rem",
                  background: "#0f172a",
                  color: "#f8fafc",
                }}
              >
                <option value="all">All</option>
                <option value="en">English</option>
                <option value="bn">Bangla</option>
              </select>
            </label>

            <label style={{ width: 140 }}>
              <span style={{ display: "block", fontWeight: 600, marginBottom: 4 }}>Results (k)</span>
              <input
                type="number"
                min={1}
                max={50}
                value={k}
                onChange={(ev) => setK(Number(ev.target.value))}
                style={{
                  width: "100%",
                  borderRadius: 8,
                  border: "1px solid rgba(255,255,255,0.15)",
                  padding: "0.55rem",
                  background: "#0f172a",
                  color: "#f8fafc",
                }}
              />
            </label>
          </div>

          <div style={{ marginTop: "1.25rem", display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <button type="submit" className="btn" disabled={loading}>
              {loading ? "Searching…" : "Search"}
            </button>
            <button
              type="button"
              className="btn"
              style={{ background: "#334155", color: "#e2e8f0" }}
              onClick={() => setShowDebug((prev) => !prev)}
            >
              {showDebug ? "Hide debug" : "Show debug"}
            </button>
          </div>
        </form>
        {error && (
          <p style={{ color: "#fca5a5", marginTop: "1rem" }}>
            {error}
          </p>
        )}
      </section>

      {response && (
        <>
          {response.warning && (
            <section
              className="card"
              style={{ borderColor: "#facc15", background: "rgba(250,204,21,0.1)", color: "#facc15" }}
            >
              <strong>Low confidence:</strong> {response.warning.message} (top score {response.warning.top_score.toFixed(3)})
            </section>
          )}

          <section className="card">
            <h2>Timing</h2>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem", marginTop: "0.5rem" }}>
              {timingEntries.map(([key, value]) => (
                <div key={key} style={{ minWidth: 120 }}>
                  <strong>{key}</strong>
                  <div>{value} ms</div>
                </div>
              ))}
            </div>
            <p style={{ marginTop: "1rem", fontSize: "0.9rem", color: "#94a3b8" }}>
              EN variant: {response.query_variants.en || "—"}
              <br />
              BN variant: {response.query_variants.bn || "—"}
            </p>
          </section>

          <section className="card">
            <h2>Results</h2>
            {response.results.length === 0 && <p>No matches found.</p>}
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              {response.results.map((result) => (
                <article
                  key={`${result.rank}-${result.url}`}
                  style={{
                    padding: "1rem",
                    borderRadius: 10,
                    border: "1px solid rgba(255,255,255,0.1)",
                    background: "#0f172a",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "0.75rem" }}>
                    <div>
                      <div style={{ fontSize: "0.85rem", color: "#94a3b8" }}>
                        #{result.rank} · {result.language.toUpperCase()} · {result.source || "Unknown"}
                      </div>
                      <a
                        href={result.url || "#"}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ fontSize: "1.1rem", fontWeight: 600 }}
                      >
                        {result.title || "Untitled"}
                      </a>
                    </div>
                    <div style={{ fontWeight: 700, fontSize: "1.15rem", color: "#22d3ee" }}>
                      {result.score.toFixed(3)}
                    </div>
                  </div>
                  <p style={{ marginTop: "0.5rem", color: "#cbd5f5", lineHeight: 1.5 }}>{result.snippet || "—"}</p>
                  {result.date && <div style={{ fontSize: "0.85rem", color: "#94a3b8" }}>{new Date(result.date).toLocaleString()}</div>}
                  {showDebug && (
                    <pre style={{ marginTop: "0.75rem", fontSize: "0.8rem", color: "#94a3b8" }}>
{`bm25=${result.debug.bm25.toFixed(3)} tfidf=${result.debug.tfidf.toFixed(3)} fuzzy=${result.debug.fuzzy.toFixed(3)} semantic=${result.debug.semantic.toFixed(3)}`}
                    </pre>
                  )}
                </article>
              ))}
            </div>
          </section>
        </>
      )}
    </main>
  );
}
