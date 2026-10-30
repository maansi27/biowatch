import { useEffect, useMemo, useState } from "react";
import { CircleMarker, GeoJSON, Marker, useMap } from "react-leaflet";
import L from "leaflet";
import type { Feature, Geometry } from "geojson";
import { IndiaMap, INDIA_BOUNDS, MapLegend } from "../components/IndiaMap";
import { IconClose, IconTarget } from "../components/Icons";
import { fmt, useData, type Hotspot, type HotspotSet, type ZoneProps } from "../data";
import { C, CLASS_COLORS, CLASS_ORDER, type SuitClass } from "../theme";

const THRESHOLDS = ["0.5", "0.6", "0.7", "0.8"];

function initialZone(): string | null {
  try {
    return sessionStorage.getItem("biowatch-zone");
  } catch {
    return null;
  }
}

function clearInitialZone() {
  try {
    sessionStorage.removeItem("biowatch-zone");
  } catch {
    // Storage unavailable; nothing to clear
  }
}

/** Fits the map to the selected hotspot, or back to India when cleared. */
function FocusController({ feature }: { feature?: Feature<Geometry> }) {
  const map = useMap();
  useEffect(() => {
    if (feature) {
      map.flyToBounds(L.geoJSON(feature).getBounds().pad(0.35), { duration: 0.8, maxZoom: 9 });
    } else {
      map.flyToBounds(INDIA_BOUNDS, { duration: 0.8 });
    }
  }, [feature, map]);
  return null;
}

function ClassZones({
  zones,
  shown,
  opacity,
}: {
  zones: GeoJSON.FeatureCollection<Geometry, ZoneProps>;
  shown: Set<string>;
  opacity: number;
}) {
  const data = useMemo(
    () => ({ ...zones, features: zones.features.filter((f) => shown.has(f.properties.zone)) }),
    [zones, shown],
  );
  return (
    <GeoJSON
      key={[...shown].join("|") + opacity}
      data={data}
      style={(f) => {
        const c = CLASS_COLORS[f!.properties.zone as SuitClass];
        return { fillColor: c, color: c, weight: 0.4, fillOpacity: opacity };
      }}
      onEachFeature={(f, layer) => {
        const p = f.properties as ZoneProps;
        layer.bindTooltip(
          `<div class="tt"><span class="tt-sw" style="background:${CLASS_COLORS[p.zone as SuitClass]}"></span>
           <b>${p.zone}</b> suitability<br/>${p.share.toFixed(1)}% of India · mean score ${
             p.mean_score?.toFixed(3) ?? "—"
           }</div>`,
          { sticky: true, className: "bw-tooltip", direction: "top", offset: [0, -8] },
        );
        layer.on({
          mouseover: (e) => e.target.setStyle({ weight: 1.6, color: C.leafDark, fillOpacity: Math.min(1, opacity + 0.15) }),
          mouseout: (e) => {
            const c = CLASS_COLORS[p.zone as SuitClass];
            e.target.setStyle({ weight: 0.4, color: c, fillOpacity: opacity });
          },
        });
      }}
    />
  );
}

function HotspotZones({
  set,
  selected,
  onSelect,
}: {
  set: HotspotSet;
  selected: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <>
      <GeoJSON
        key={String(selected)}
        data={{ type: "FeatureCollection", features: set.features } as GeoJSON.FeatureCollection}
        style={(f) => {
          const on = f!.properties.zone_id === selected;
          return {
            color: C.magenta,
            weight: on ? 3.5 : 1.8,
            dashArray: on ? undefined : "5 4",
            fillColor: C.magenta,
            fillOpacity: on ? 0.18 : 0.02,
          };
        }}
        onEachFeature={(f, layer) => {
          const p = f.properties as { zone_id: string; near: string };
          layer.bindTooltip(`<div class="tt"><b>Hotspot ${p.zone_id}</b> · near ${p.near}<br/><span class="tt-hint">Click for details</span></div>`, {
            sticky: true,
            className: "bw-tooltip",
            direction: "top",
            offset: [0, -8],
          });
          layer.on({
            click: () => onSelect(p.zone_id),
            mouseover: (e) => e.target.setStyle({ fillOpacity: 0.22, weight: 3 }),
            mouseout: (e) => {
              const on = p.zone_id === selected;
              e.target.setStyle({ fillOpacity: on ? 0.18 : 0.02, weight: on ? 3.5 : 1.8 });
            },
          });
        }}
      />
      {set.table.slice(0, 18).map((z) => (
        <Marker
          key={z.id}
          position={[z.lat, z.lon]}
          eventHandlers={{ click: () => onSelect(z.id) }}
          icon={L.divIcon({
            className: "",
            html: `<span class="zone-pin${z.id === selected ? " on" : ""}">${z.id}</span>`,
            iconSize: [0, 0],
          })}
        />
      ))}
    </>
  );
}

function ZoneDetail({ z, onClose }: { z: Hotspot; onClose: () => void }) {
  const level = z.mean >= 0.8 ? "Very High" : z.mean >= 0.6 ? "High" : "Moderate";
  return (
    <div className="zone-detail">
      <div className="zd-head">
        <span className="zone-badge lg">{z.id}</span>
        <div>
          <h3>Near {z.near}</h3>
          <span className="zd-level" style={{ background: CLASS_COLORS[level as SuitClass] }}>
            {level} suitability
          </span>
        </div>
        <button className="icon-btn" aria-label="Close zone" onClick={onClose}>
          <IconClose />
        </button>
      </div>
      <div className="zd-grid">
        <div>
          <span>Area</span>
          <b>{fmt(z.area)} km²</b>
        </div>
        <div>
          <span>Sightings</span>
          <b>{fmt(z.sightings)}</b>
        </div>
        <div>
          <span>Mean score</span>
          <b>{z.mean.toFixed(3)}</b>
        </div>
        <div>
          <span>Peak score</span>
          <b>{z.peak.toFixed(3)}</b>
        </div>
      </div>
      <div className="meter">
        <span style={{ width: `${z.mean * 100}%` }} />
      </div>
      <p className="fine">
        {z.veryHigh} cells score 0.8 or above. Centre {z.lat.toFixed(2)}°N, {z.lon.toFixed(2)}°E.
      </p>
    </div>
  );
}

export default function ZoneMap() {
  const zones = useData("zones");
  const hotspots = useData("hotspots");
  const sightings = useData("sightings");

  const [shown, setShown] = useState<Set<string>>(new Set(CLASS_ORDER));
  const [threshold, setThreshold] = useState("0.6");
  const [opacity, setOpacity] = useState(0.7);
  const [showHot, setShowHot] = useState(true);
  const [showSightings, setShowSightings] = useState(false);
  const [selected, setSelected] = useState<string | null>(initialZone);
  const [query, setQuery] = useState("");

  useEffect(clearInitialZone, []);

  const set = hotspots?.[threshold];
  const selectedZone = set?.table.find((z) => z.id === selected);
  const selectedFeature = set?.features.find((f) => f.properties.zone_id === selected);

  // Zone ids are re-numbered per threshold, so clear a selection that no longer exists
  useEffect(() => {
    if (set && selected && !set.table.some((z) => z.id === selected)) setSelected(null);
  }, [set, selected]);

  const toggleClass = (c: string) =>
    setShown((s) => {
      const n = new Set(s);
      n.has(c) ? n.delete(c) : n.add(c);
      return n;
    });

  const list = (set?.table ?? []).filter(
    (z) => !query || z.near.toLowerCase().includes(query.toLowerCase()) || z.id.toLowerCase().includes(query.toLowerCase()),
  );
  const totalArea = set?.table.reduce((a, z) => a + z.area, 0) ?? 0;

  return (
    <div className="mapview">
      <aside className="map-side map-controls">
        <div className="side-head">
          <span className="eyebrow">Zone map</span>
          <h1>Lantana suitability zones of India</h1>
          <p className="fine">Hover a zone for its class. Click a hotspot to see its details.</p>
        </div>

        <div className="ctrl">
          <label className="ctrl-label">Suitability zones</label>
          <div className="class-toggles">
            {CLASS_ORDER.map((c) => {
              const share = zones?.features.find((f) => f.properties.zone === c)?.properties.share;
              return (
                <button
                  key={c}
                  className={shown.has(c) ? "class-toggle on" : "class-toggle"}
                  onClick={() => toggleClass(c)}
                  aria-pressed={shown.has(c)}
                >
                  <span className="swatch" style={{ background: CLASS_COLORS[c] }} />
                  <span className="ct-name">{c}</span>
                  <span className="ct-share">{share !== undefined ? `${share.toFixed(1)}%` : ""}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="ctrl">
          <label className="ctrl-label">
            Zone opacity <span className="ctrl-val">{Math.round(opacity * 100)}%</span>
          </label>
          <input
            type="range"
            min={0.2}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
          />
        </div>

        <div className="ctrl">
          <label className="ctrl-label">Hotspot threshold</label>
          <div className="segmented">
            {THRESHOLDS.map((t) => (
              <button key={t} className={t === threshold ? "on" : ""} onClick={() => setThreshold(t)}>
                ≥ {t}
              </button>
            ))}
          </div>
          <p className="fine">Connected areas scoring at or above this form a hotspot zone.</p>
        </div>

        <div className="ctrl">
          <label className="switch">
            <input type="checkbox" checked={showHot} onChange={(e) => setShowHot(e.target.checked)} />
            <span className="switch-track" />
            Hotspot zones
          </label>
          <label className="switch">
            <input
              type="checkbox"
              checked={showSightings}
              onChange={(e) => setShowSightings(e.target.checked)}
            />
            <span className="switch-track" />
            Recorded sightings {sightings && <span className="ctrl-val">{fmt(sightings.length)}</span>}
          </label>
        </div>
      </aside>

      <div className="map-wrap">
        {!zones && <div className="map-loading">Loading zones…</div>}
        <IndiaMap>
          {zones && <ClassZones zones={zones} shown={shown} opacity={opacity} />}
          {set && showHot && <HotspotZones key={threshold} set={set} selected={selected} onSelect={setSelected} />}
          {showSightings &&
            sightings?.map(([lat, lon], i) => (
              <CircleMarker
                key={i}
                center={[lat, lon]}
                radius={2.4}
                pathOptions={{ color: C.leafDark, weight: 1, fillColor: "#fff", fillOpacity: 0.95 }}
                interactive={false}
              />
            ))}
          <FocusController feature={selectedFeature} />
        </IndiaMap>
        <MapLegend
          title="Lantana suitability"
          items={[
            ...CLASS_ORDER.map((c) => ({ label: c, color: CLASS_COLORS[c] })),
            { label: "Hotspot zone", color: C.magenta, dashed: true },
          ]}
        />
        {selected && (
          <button className="map-reset" onClick={() => setSelected(null)}>
            <IconTarget /> Show all of India
          </button>
        )}
      </div>

      <aside className="map-side map-list">
        {selectedZone ? (
          <ZoneDetail z={selectedZone} onClose={() => setSelected(null)} />
        ) : (
          <div className="zone-summary">
            <span className="zs-num">{set?.table.length ?? "—"}</span>
            <span>
              hotspot zones scoring ≥ {threshold}
              <br />
              <b>{fmt(totalArea)} km²</b> in total
            </span>
          </div>
        )}

        <input
          className="search"
          placeholder="Search by zone or town…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <ul className="zone-list">
          {list.map((z) => (
            <li key={z.id}>
              <button className={z.id === selected ? "on" : ""} onClick={() => setSelected(z.id)}>
                <span className="zone-badge">{z.id}</span>
                <span className="zl-main">
                  <b>Near {z.near}</b>
                  <span>
                    {fmt(z.area)} km² · {fmt(z.sightings)} sightings
                  </span>
                </span>
                <span className="zl-score">
                  <span className="mini-meter">
                    <span style={{ width: `${z.mean * 100}%` }} />
                  </span>
                  {z.mean.toFixed(2)}
                </span>
              </button>
            </li>
          ))}
          {set && list.length === 0 && <li className="fine">No zones match “{query}”.</li>}
        </ul>
        <p className="fine">
          India outline from Natural Earth 1:110m (coarse; small islands not shown). Zone names refer to the
          nearest reference town.
        </p>
      </aside>
    </div>
  );
}
