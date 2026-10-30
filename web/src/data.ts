import { useEffect, useState } from "react";
import type { Feature, FeatureCollection, Geometry } from "geojson";

// Files are produced by scripts/11_export_web_data.py
export interface Summary {
  totalRecords: number;
  presenceRecords: number;
  uniqueLocations: number;
  features: number;
  gridCells: number;
  bestModel: string;
  bestAuc: number;
  models: string[];
  classShare: { zone: string; share: number; cells: number }[];
}

export interface India {
  outline: Geometry;
  mask: Geometry;
}

export interface ZoneProps {
  zone: string;
  cells: number;
  mean_score: number | null;
  share: number;
}

export interface Hotspot {
  id: string;
  near: string;
  area: number;
  mean: number;
  peak: number;
  veryHigh: number;
  sightings: number;
  lat: number;
  lon: number;
}

export interface HotspotSet {
  features: Feature<Geometry, { zone_id: string; near: string }>[];
  table: Hotspot[];
}

export interface ModelMetrics {
  name: string;
  auc: number;
  precision: number;
  recall: number;
  f1: number;
  cvMean: number | null;
  cvStd: number | null;
  cvFolds: (number | null)[];
  confusion: number[][];
  roc: { fpr: number; tpr: number }[];
  importance: { type: string; items: { feature: string; value: number }[] };
}

export type Distributions = Record<string, { x: number; presence: number; background: number }[]>;

export interface PriorityTierProps {
  tier: string;
  rank: number;
  cells: number;
  min: number;
  max: number;
}

export interface Priority {
  histogram: { x: number; count: number }[];
  byTopN: Record<string, Feature<Geometry, PriorityTierProps>[]>;
}

export interface DataFiles {
  summary: Summary;
  india: India;
  zones: FeatureCollection<Geometry, ZoneProps>;
  hotspots: Record<string, HotspotSet>;
  sightings: [number, number][];
  metrics: ModelMetrics[];
  distributions: Distributions;
  priority: Priority;
}

const cache = new Map<string, Promise<unknown>>();

function load<T>(name: string): Promise<T> {
  if (!cache.has(name)) {
    cache.set(
      name,
      fetch(`./data/${name}.json`).then((r) => {
        if (!r.ok) throw new Error(`${name}.json: ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(name) as Promise<T>;
}

/** Load one exported data file; returns undefined while loading. */
export function useData<K extends keyof DataFiles>(name: K): DataFiles[K] | undefined {
  const [value, setValue] = useState<DataFiles[K]>();
  useEffect(() => {
    let alive = true;
    load<DataFiles[K]>(name)
      .then((v) => alive && setValue(v))
      .catch((e) => console.error(e));
    return () => {
      alive = false;
    };
  }, [name]);
  return value;
}

export const fmt = (n: number) => n.toLocaleString("en-IN");
