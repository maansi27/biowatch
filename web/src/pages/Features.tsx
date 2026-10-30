import { useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useData } from "../data";
import { BIO_CODE, C, label } from "../theme";

const IMPORTANCE_LABEL: Record<string, string> = {
  gini_impurity: "Mean decrease in Gini impurity",
  absolute_coefficient: "Absolute coefficient",
};

// Yellow → orange → pink → magenta by rank
const RAMP = [C.magenta, C.pink, C.orange, "#f5a030", C.yellow];
const rampColor = (i: number, n: number) => RAMP[Math.min(RAMP.length - 1, Math.floor((i / n) * RAMP.length))];

export default function Features() {
  const metrics = useData("metrics");
  const dists = useData("distributions");
  const withFi = metrics?.filter((m) => m.importance.items.length) ?? [];
  const [modelName, setModelName] = useState<string>();
  const [chosen, setChosen] = useState<string[]>([]);

  const model = withFi.find((m) => m.name === modelName) ?? withFi.find((m) => m.name === "Random Forest") ?? withFi[0];
  const items = model ? [...model.importance.items].sort((a, b) => b.value - a.value) : [];
  const top = items.slice(0, 12).map((d) => ({ ...d, name: label(d.feature) }));

  // Default to the three strongest drivers of the current model
  useEffect(() => {
    if (items.length && chosen.length === 0) setChosen(items.slice(0, 3).map((d) => d.feature));
  }, [items, chosen.length]);

  const toggle = (f: string) =>
    setChosen((c) => (c.includes(f) ? c.filter((x) => x !== f) : c.length >= 6 ? c : [...c, f]));

  if (!metrics) return <div className="loading">Loading…</div>;

  return (
    <>
      <section className="page-head">
        <span className="eyebrow">Climate drivers</span>
        <h1>What makes a place suit Lantana?</h1>
        <p className="lead">
          Feature importance shows which climate variables best separate Lantana sightings from background
          locations. It shows association, not cause.
        </p>
      </section>

      <section className="panel section">
        <div className="panel-toolbar">
          <div className="section-head">
            <span className="eyebrow">Importance</span>
            <h2>Top climate variables</h2>
          </div>
          <div className="segmented">
            {withFi.map((m) => (
              <button key={m.name} className={m.name === model?.name ? "on" : ""} onClick={() => setModelName(m.name)}>
                {m.name}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={top.length * 34 + 40}>
          <BarChart data={top} layout="vertical" margin={{ top: 4, right: 30, left: 10, bottom: 4 }}>
            <CartesianGrid horizontal={false} stroke={C.line} />
            <XAxis type="number" tick={{ fill: C.muted, fontSize: 12 }} />
            <YAxis
              type="category"
              dataKey="name"
              width={210}
              tick={{ fill: C.leafDark, fontSize: 13 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              formatter={(v: number) => [v.toFixed(4), IMPORTANCE_LABEL[model?.importance.type ?? ""] ?? "Importance"]}
              labelFormatter={(n: string) => {
                const f = top.find((d) => d.name === n)?.feature ?? "";
                return `${n} (${BIO_CODE[f] ?? f})`;
              }}
              cursor={{ fill: "rgba(226,85,143,.06)" }}
              contentStyle={{ borderRadius: 10, border: `1px solid ${C.line}` }}
            />
            <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={20}>
              {top.map((d, i) => (
                <Cell key={d.feature} fill={rampColor(i, top.length)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="fine">Importance type: {IMPORTANCE_LABEL[model?.importance.type ?? ""] ?? model?.importance.type}</p>
      </section>

      <section className="panel section">
        <div className="section-head">
          <span className="eyebrow">Distributions</span>
          <h2>Where sightings sit on each variable</h2>
        </div>
        <div className="feature-chips">
          {items.map((d) => (
            <button
              key={d.feature}
              className={chosen.includes(d.feature) ? "fchip on" : "fchip"}
              onClick={() => toggle(d.feature)}
            >
              {label(d.feature)}
            </button>
          ))}
        </div>
        <div className="legend-inline">
          <span>
            <span className="swatch" style={{ background: C.pink }} /> Lantana sightings
          </span>
          <span>
            <span className="swatch" style={{ background: "#9fb59a" }} /> Background
          </span>
        </div>
        <div className="dist-grid">
          {chosen.map((f) => (
            <div className="dist-card" key={f}>
              <h4>
                {label(f)} <small>{BIO_CODE[f]}</small>
              </h4>
              <ResponsiveContainer width="100%" height={190}>
                <AreaChart data={dists?.[f] ?? []} margin={{ top: 6, right: 6, left: -18, bottom: 0 }}>
                  <CartesianGrid vertical={false} stroke={C.line} />
                  <XAxis dataKey="x" tick={{ fill: C.muted, fontSize: 11 }} minTickGap={30} tickFormatter={(v: number) => (Math.abs(v) >= 10 ? String(Math.round(v)) : v.toFixed(1))} />
                  <YAxis tick={{ fill: C.muted, fontSize: 11 }} />
                  <Tooltip contentStyle={{ borderRadius: 10, border: `1px solid ${C.line}` }} />
                  <Area type="monotone" dataKey="background" name="Background" stroke="#7f9a79" fill="#9fb59a" fillOpacity={0.35} />
                  <Area type="monotone" dataKey="presence" name="Sightings" stroke={C.pink} fill={C.pink} fillOpacity={0.35} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
        <p className="fine">Pick up to six variables.</p>
      </section>
    </>
  );
}
