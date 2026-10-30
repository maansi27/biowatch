import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmt, useData, type ModelMetrics } from "../data";
import { C, MODEL_COLORS } from "../theme";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function ConfusionMatrix({ m, color }: { m: ModelMetrics; color: string }) {
  const [[tn, fp], [fn, tp]] = m.confusion;
  const max = Math.max(tn, fp, fn, tp);
  const cell = (v: number, good: boolean, label: string) => (
    <div
      className="cm-cell"
      style={{
        background: good ? color : C.sand,
        opacity: good ? 0.35 + 0.65 * (v / max) : 1,
        color: good ? "#fff" : C.leafDark,
      }}
    >
      <b>{fmt(v)}</b>
      <span>{label}</span>
    </div>
  );
  return (
    <div className="cm">
      <div className="cm-axis-y">Actual</div>
      <div className="cm-body">
        <div className="cm-cols">
          <span>Background</span>
          <span>Presence</span>
        </div>
        <div className="cm-row">
          <span className="cm-rowlabel">Background</span>
          {cell(tn, true, "true negative")}
          {cell(fp, false, "false positive")}
        </div>
        <div className="cm-row">
          <span className="cm-rowlabel">Presence</span>
          {cell(fn, false, "false negative")}
          {cell(tp, true, "true positive")}
        </div>
        <div className="cm-axis-x">Predicted</div>
      </div>
    </div>
  );
}

export default function Models() {
  const metrics = useData("metrics");
  if (!metrics) return <div className="loading">Loading models…</div>;

  const best = metrics.reduce((a, b) => (b.auc > a.auc ? b : a));
  const folds = Math.max(...metrics.map((m) => m.cvFolds.length));
  const cvData = Array.from({ length: folds }, (_, i) => ({
    fold: `Fold ${i + 1}`,
    ...Object.fromEntries(metrics.map((m) => [m.name, m.cvFolds[i]])),
  }));

  return (
    <>
      <section className="page-head">
        <span className="eyebrow">Model comparison</span>
        <h1>Three models, one winner</h1>
        <p className="lead">
          Each model learns to tell Lantana sightings apart from random background locations. Background points
          are pseudo-absences, so these scores measure a habitat signal, not confirmed absence accuracy.
        </p>
      </section>

      <section className="model-cards">
        {metrics.map((m, i) => (
          <article key={m.name} className={m.name === best.name ? "model-card best" : "model-card"}>
            {m.name === best.name && <span className="best-tag">Selected model</span>}
            <h3>
              <span className="dot" style={{ background: MODEL_COLORS[i] }} />
              {m.name}
            </h3>
            <div className="mc-auc">
              <span>{m.auc.toFixed(3)}</span>
              <small>hold-out AUC</small>
            </div>
            <dl className="mc-grid">
              <div>
                <dt>Spatial CV AUC</dt>
                <dd>{m.cvMean !== null ? `${m.cvMean.toFixed(3)} ± ${(m.cvStd ?? 0).toFixed(3)}` : "—"}</dd>
              </div>
              <div>
                <dt>Precision</dt>
                <dd>{pct(m.precision)}</dd>
              </div>
              <div>
                <dt>Recall</dt>
                <dd>{pct(m.recall)}</dd>
              </div>
              <div>
                <dt>F1-score</dt>
                <dd>{pct(m.f1)}</dd>
              </div>
            </dl>
          </article>
        ))}
      </section>

      <section className="split">
        <div className="panel">
          <div className="section-head">
            <span className="eyebrow">Discrimination</span>
            <h2>ROC curves</h2>
          </div>
          <ResponsiveContainer width="100%" height={380}>
            <LineChart margin={{ top: 10, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid stroke={C.line} />
              <XAxis
                type="number"
                dataKey="fpr"
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tick={{ fill: C.muted, fontSize: 12 }}
                label={{ value: "False positive rate", position: "insideBottom", offset: -10, fill: C.muted, fontSize: 12 }}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tick={{ fill: C.muted, fontSize: 12 }}
                width={40}
              />
              <ReferenceLine
                segment={[
                  { x: 0, y: 0 },
                  { x: 1, y: 1 },
                ]}
                stroke="#c9c1ae"
                strokeDasharray="5 4"
              />
              <Tooltip
                formatter={(v: number, n: string) => [v.toFixed(3), n]}
                labelFormatter={(v: number) => `FPR ${Number(v).toFixed(3)}`}
                contentStyle={{ borderRadius: 10, border: `1px solid ${C.line}` }}
              />
              <Legend verticalAlign="bottom" iconType="circle" wrapperStyle={{ paddingTop: 28, fontSize: 13 }} />
              {metrics.map((m, i) => (
                <Line
                  key={m.name}
                  data={m.roc}
                  dataKey="tpr"
                  name={`${m.name} (${m.auc.toFixed(3)})`}
                  stroke={MODEL_COLORS[i]}
                  strokeWidth={2.5}
                  dot={false}
                  type="monotone"
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="panel">
          <div className="section-head">
            <span className="eyebrow">Generalisation</span>
            <h2>Spatial block cross-validation</h2>
          </div>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={cvData} margin={{ top: 10, right: 10, left: 0, bottom: 18 }}>
              <CartesianGrid vertical={false} stroke={C.line} />
              <XAxis dataKey="fold" tick={{ fill: C.muted, fontSize: 12 }} />
              <YAxis domain={[0.5, 1]} tick={{ fill: C.muted, fontSize: 12 }} width={40} />
              <Tooltip
                formatter={(v: number, n: string) => [v?.toFixed(3) ?? "—", n]}
                cursor={{ fill: "rgba(226,85,143,.06)" }}
                contentStyle={{ borderRadius: 10, border: `1px solid ${C.line}` }}
              />
              <Legend verticalAlign="top" height={30} iconType="circle" />
              {metrics.map((m, i) => (
                <Bar key={m.name} dataKey={m.name} fill={MODEL_COLORS[i]} radius={[4, 4, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
          <p className="fine">
            Each fold holds out a whole geographic block, so the model is tested on regions it never saw. This is
            the fairer score for presence-background data.
          </p>
        </div>
      </section>

      <section className="panel section">
        <div className="section-head">
          <span className="eyebrow">Errors</span>
          <h2>Confusion matrices</h2>
        </div>
        <div className="cm-grid">
          {metrics.map((m, i) => (
            <div key={m.name}>
              <h3 className="cm-title">
                <span className="dot" style={{ background: MODEL_COLORS[i] }} />
                {m.name}
              </h3>
              <ConfusionMatrix m={m} color={MODEL_COLORS[i]} />
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
