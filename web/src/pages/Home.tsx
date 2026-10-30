import { fmt, useData } from "../data";
import { CLASS_COLORS, type SuitClass } from "../theme";
import { IconAlert, IconMap, IconSearch } from "../components/Icons";

const STEPS = [
  { n: "01", title: "Collect sightings", text: "Thousands of Lantana occurrence records across India from GBIF and iNaturalist." },
  { n: "02", title: "Read the climate", text: "19 WorldClim bioclimatic layers plus elevation for every sighting and background point." },
  { n: "03", title: "Train & compare", text: "Logistic Regression, Random Forest and XGBoost, tested with spatial block cross-validation." },
  { n: "04", title: "Map the zones", text: "The best model scores a 0.1° grid across India, dissolved into suitability and hotspot zones." },
];

function rememberZone(id: string) {
  try {
    sessionStorage.setItem("biowatch-zone", id);
  } catch {
    // Storage can be unavailable (private mode); the map then opens on all of India
  }
}

export default function Home() {
  const summary = useData("summary");
  const hotspots = useData("hotspots");
  const top = hotspots?.["0.6"]?.table.slice(0, 5) ?? [];
  const hotCount = hotspots?.["0.6"]?.table.length;

  return (
    <>
      <section className="hero">
        <div className="hero-text">
          <span className="kicker">Invasive species intelligence</span>
          <h1>
            Where could <span className="grad">Lantana</span> spread next?
          </h1>
          <p className="lead">
            <i>Lantana camara</i> is one of the world's most invasive shrubs and has spread through large
            parts of India's forests and grasslands. BioWatch combines thousands of sightings with
            climate data and machine learning to map the zones where conditions suit the plant.
          </p>
          <div className="hero-actions">
            <a className="btn btn-primary" href="#/map">
              Explore the zone map →
            </a>
            <a className="btn btn-ghost" href="#/models">
              How the model works
            </a>
          </div>
          <div className="chips">
            <span className="chip y">Native to tropical America</span>
            <span className="chip o">Spread by fruit-eating birds</span>
            <span className="chip p">Toxic to livestock</span>
            <span className="chip m">IUCN top-100 invader</span>
          </div>
        </div>
        <div className="hero-visual">
          <div className="hero-photo" role="img" aria-label="Lantana camara flower cluster" />
          <div className="float-card fc-1">
            <span className="fc-label">Best model AUC</span>
            <span className="fc-value">{summary ? summary.bestAuc.toFixed(3) : "—"}</span>
            <span className="fc-sub">{summary?.bestModel}</span>
          </div>
          <div className="float-card fc-2">
            <span className="fc-dot" />
            <span>
              <b>{hotCount ?? "—"}</b> hotspot zones
            </span>
          </div>
        </div>
      </section>

      <section className="stats">
        {[
          ["Total records", summary && fmt(summary.totalRecords), "pink"],
          ["Lantana sightings", summary && fmt(summary.presenceRecords), "orange"],
          ["Unique locations", summary && fmt(summary.uniqueLocations), "yellow"],
          ["Climate features", summary && String(summary.features), "leaf"],
          ["Grid cells scored", summary && fmt(summary.gridCells), "magenta"],
        ].map(([k, v, c]) => (
          <div className={`stat stat-${c}`} key={k}>
            <span className="stat-value">{v ?? "—"}</span>
            <span className="stat-label">{k}</span>
          </div>
        ))}
      </section>

      <section className="section">
        <div className="section-head">
          <span className="eyebrow">The species</span>
          <h2>Know the invader</h2>
        </div>
        <div className="cards-3">
          <article className="card">
            <span className="card-icon ci-yellow"><IconSearch /></span>
            <h3>How to recognise it</h3>
            <p>
              A thorny, aromatic shrub with rough leaves and round clusters of tiny flowers that change
              colour as they age: yellow at the centre, turning orange, pink and red.
            </p>
          </article>
          <article className="card">
            <span className="card-icon ci-pink"><IconAlert /></span>
            <h3>Why it matters</h3>
            <p>
              It forms dense thickets that crowd out native plants, reduces fodder for wildlife, is toxic
              to cattle and makes forest fires burn hotter.
            </p>
          </article>
          <article className="card">
            <span className="card-icon ci-leaf"><IconMap /></span>
            <h3>How BioWatch helps</h3>
            <p>
              Suitability zones show where the climate matches places Lantana already grows, so field
              teams can plan surveys and early removal where it matters most.
            </p>
          </article>
        </div>
      </section>

      <section className="section split">
        <div className="panel">
          <div className="section-head">
            <span className="eyebrow">Across India</span>
            <h2>Share of land by suitability</h2>
          </div>
          <div className="stack-bar">
            {summary?.classShare.map((c) => (
              <div
                key={c.zone}
                title={`${c.zone}: ${c.share.toFixed(1)}%`}
                style={{ flex: c.share, background: CLASS_COLORS[c.zone as SuitClass] }}
              />
            ))}
          </div>
          <ul className="share-list">
            {summary?.classShare.map((c) => (
              <li key={c.zone}>
                <span className="swatch" style={{ background: CLASS_COLORS[c.zone as SuitClass] }} />
                <span className="share-name">{c.zone}</span>
                <span className="share-bar">
                  <span
                    style={{
                      width: `${Math.max(2, c.share)}%`,
                      background: CLASS_COLORS[c.zone as SuitClass],
                    }}
                  />
                </span>
                <span className="share-val">{c.share.toFixed(1)}%</span>
              </li>
            ))}
          </ul>
          <p className="fine">Share of 0.1° grid cells inside India in each zone.</p>
        </div>

        <div className="panel">
          <div className="section-head">
            <span className="eyebrow">Largest hotspots</span>
            <h2>Zones to watch</h2>
          </div>
          <ol className="top-zones">
            {top.map((z) => (
              <li key={z.id}>
                <a href={`#/map`} onClick={() => rememberZone(z.id)}>
                  <span className="zone-badge">{z.id}</span>
                  <span className="tz-main">
                    <b>Near {z.near}</b>
                    <span>
                      {fmt(z.area)} km² · {fmt(z.sightings)} sightings
                    </span>
                  </span>
                  <span className="tz-score" style={{ ["--p" as string]: z.mean }}>
                    {z.mean.toFixed(2)}
                  </span>
                </a>
              </li>
            ))}
          </ol>
          <a className="btn btn-ghost btn-sm" href="#/map">
            See all zones on the map →
          </a>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <span className="eyebrow">Method</span>
          <h2>From sightings to zones</h2>
        </div>
        <div className="steps">
          {STEPS.map((s) => (
            <div className="step" key={s.n}>
              <span className="step-n">{s.n}</span>
              <h3>{s.title}</h3>
              <p>{s.text}</p>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
