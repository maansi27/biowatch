import { useEffect, useState } from "react";
import Home from "./pages/Home";
import ZoneMap from "./pages/ZoneMap";
import Models from "./pages/Models";
import Features from "./pages/Features";
import Priority from "./pages/Priority";

export const PAGES = [
  { id: "home", label: "Home" },
  { id: "map", label: "Zone Map" },
  { id: "priority", label: "Priority" },
  { id: "models", label: "Models" },
  { id: "features", label: "Climate Drivers" },
] as const;

export type PageId = (typeof PAGES)[number]["id"];

const current = (): PageId => {
  const h = window.location.hash.replace("#/", "").replace("#", "");
  return (PAGES.find((p) => p.id === h)?.id ?? "home") as PageId;
};

export const go = (id: PageId) => {
  window.location.hash = `#/${id}`;
};

export default function App() {
  const [page, setPage] = useState<PageId>(current);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onHash = () => {
      setPage(current());
      setMenuOpen(false);
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const label = PAGES.find((p) => p.id === page)?.label;
    document.title = page === "home" ? "BioWatch | Lantana camara in India" : `${label} | BioWatch`;
  }, [page]);

  return (
    <>
      <header className="topbar">
        <a className="brand" href="#/home">
          <img src="./lantana.svg" alt="" />
          <span>
            Bio<em>Watch</em>
          </span>
        </a>
        <button
          className="menu-btn"
          aria-label="Open menu"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((o) => !o)}
        >
          <span />
          <span />
          <span />
        </button>
        <nav className={menuOpen ? "nav open" : "nav"}>
          {PAGES.map((p) => (
            <a key={p.id} href={`#/${p.id}`} className={p.id === page ? "active" : ""}>
              {p.label}
            </a>
          ))}
        </nav>
      </header>

      <main className={page === "map" ? "page page-wide" : "page"}>
        {page === "home" && <Home />}
        {page === "map" && <ZoneMap />}
        {page === "priority" && <Priority />}
        {page === "models" && <Models />}
        {page === "features" && <Features />}
      </main>

      <footer className="footer">
        <div className="footer-petals" />
        <p>
          <strong>BioWatch</strong> · GBIF occurrence records · WorldClim 2.1 climate · scikit-learn &amp;
          XGBoost
        </p>
        <p className="footer-note">
          Suitability scores show where climate resembles places Lantana already grows. They are not
          confirmed invasion or ecological damage.
        </p>
      </footer>
    </>
  );
}
