import { useMemo, useState } from "react";
import { GeoJSON } from "react-leaflet";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { IndiaMap, MapLegend } from "../components/IndiaMap";
import { fmt, useData, type PriorityTierProps } from "../data";
import { C, TIER_COLORS } from "../theme";

const TOP_N = Array.from({ length: 10 }, (_, i) => (i + 1) * 500);
const TIERS = ["Urgent", "High", "Elevated", "Watch"];

export default function Priority() {
  const priority = useData("priority");
  const [nIdx, setNIdx] = useState(3);
  const n = TOP_N[nIdx];
  const tiers = priority?.byTopN[String(n)] ?? [];

  const cutoff = useMemo(() => Math.min(...tiers.map((t) => t.properties.min)), [tiers]);

  return (
    <>
      <section className="page-head">
        <span className="eyebrow">Field planning</span>
        <h1>Monitoring priority</h1>
        <p className="lead">
          Priority combines habitat suitability with how close a place is to known Lantana sightings. Use it
          to decide where field surveys should go first.
        </p>
      </section>

      <section className="split split-map">
        <div className="panel map-panel">
          <IndiaMap className="map map-md">
            {tiers.length > 0 && (
              <GeoJSON
                key={n}
                data={{ type: "FeatureCollection", features: tiers } as GeoJSON.FeatureCollection}
                style={(f) => {
                  const c = TIER_COLORS[(f!.properties as PriorityTierProps).tier];
                  return { fillColor: c, color: c, weight: 0.6, fillOpacity: 0.85 };
                }}
                onEachFeature={(f, layer) => {
                  const p = f.properties as PriorityTierProps;
                  layer.bindTooltip(
                    `<div class="tt"><span class="tt-sw" style="background:${TIER_COLORS[p.tier]}"></span><b>${p.tier} priority</b><br/>Score ${p.min.toFixed(3)} – ${p.max.toFixed(3)} · ${fmt(p.cells)} cells</div>`,
                    { sticky: true, className: "bw-tooltip", direction: "top", offset: [0, -8] },
                  );
                  layer.on({
                    mouseover: (e) => e.target.setStyle({ weight: 2, color: C.leafDark }),
                    mouseout: (e) => e.target.setStyle({ weight: 0.6, color: TIER_COLORS[p.tier] }),
                  });
                }}
              />
            )}
          </IndiaMap>
          <MapLegend title="Monitoring priority" items={TIERS.map((t) => ({ label: t, color: TIER_COLORS[t] }))} />
        </div>

        <div className="panel">
          <div className="ctrl">
            <label className="ctrl-label">
              Cells to show <span className="ctrl-val">top {fmt(n)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={TOP_N.length - 1}
              step={1}
              value={nIdx}
              onChange={(e) => setNIdx(Number(e.target.value))}
            />
            <div className="range-scale">
              <span>500</span>
              <span>5,000</span>
            </div>
          </div>

          <ul className="tier-list">
            {TIERS.map((name) => {
              const t = tiers.find((x) => x.properties.tier === name)?.properties;
              return (
                <li key={name}>
                  <span className="swatch lg" style={{ background: TIER_COLORS[name] }} />
                  <span className="tl-main">
                    <b>{name}</b>
                    <span>{t ? `${t.min.toFixed(3)} – ${t.max.toFixed(3)}` : "—"}</span>
                  </span>
                  <span className="tl-cells">{t ? fmt(t.cells) : "—"} cells</span>
                </li>
              );
            })}
          </ul>
          <p className="fine">
            Tiers split the top {fmt(n)} cells into quartiles. Every shown cell scores at least{" "}
            {Number.isFinite(cutoff) ? cutoff.toFixed(3) : "—"}.
          </p>

          <div className="formula">
            <span className="eyebrow">Formula</span>
            <code>Priority = 0.7 × Suitability + 0.3 × Proximity</code>
            <p className="fine">
              Proximity = exp(−d / 150), where d is the distance in km to the nearest sighting. It falls to about
              0.19 at 250 km and 0.04 at 500 km. The weights are planning heuristics, not ecologically validated
              values.
            </p>
          </div>
        </div>
      </section>

      <section className="panel section">
        <div className="section-head">
          <span className="eyebrow">Distribution</span>
          <h2>Priority scores across India</h2>
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={priority?.histogram ?? []} margin={{ top: 10, right: 10, left: 0, bottom: 10 }}>
            <CartesianGrid vertical={false} stroke={C.line} />
            <XAxis dataKey="x" tickFormatter={(v) => v.toFixed(1)} interval={7} tick={{ fill: C.muted, fontSize: 12 }} />
            <YAxis tick={{ fill: C.muted, fontSize: 12 }} width={50} />
            <Tooltip
              cursor={{ fill: "rgba(226,85,143,.08)" }}
              formatter={(v: number) => [fmt(v), "Cells"]}
              labelFormatter={(v: number) => `Score ≈ ${v.toFixed(3)}`}
              contentStyle={{ borderRadius: 10, border: `1px solid ${C.line}` }}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {(priority?.histogram ?? []).map((d) => (
                <Cell key={d.x} fill={d.x >= cutoff ? C.magenta : C.pink} fillOpacity={d.x >= cutoff ? 1 : 0.45} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="fine">Bars in dark magenta fall inside the selected top {fmt(n)} cells.</p>
      </section>
    </>
  );
}
