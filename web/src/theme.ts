// Lantana palette: leaf greens plus the flower's yellow → orange → pink → magenta
export const C = {
  leafDark: "#1f3d2b",
  leaf: "#2f6b3f",
  leafSoft: "#b9d3ae",
  cream: "#fffaf2",
  sand: "#f6efe2",
  line: "#eadfca",
  yellow: "#f9c22e",
  orange: "#f28c28",
  pink: "#e2558f",
  magenta: "#a4134a",
  muted: "#6b7b6e",
};

export const CLASS_ORDER = ["Very Low", "Low", "Moderate", "High", "Very High"] as const;
export type SuitClass = (typeof CLASS_ORDER)[number];

// Suitability classes follow the colour change of an ageing Lantana floret
export const CLASS_COLORS: Record<SuitClass, string> = {
  "Very Low": "#b9d3ae",
  Low: "#f9d65c",
  Moderate: "#f5a030",
  High: "#e2558f",
  "Very High": "#a4134a",
};

export const TIER_COLORS: Record<string, string> = {
  Watch: C.yellow,
  Elevated: C.orange,
  High: C.pink,
  Urgent: C.magenta,
};

export const MODEL_COLORS = [C.pink, C.leaf, C.orange];

export const FEATURE_LABELS: Record<string, string> = {
  temperature: "Annual Mean Temperature",
  rainfall: "Annual Precipitation",
  elevation: "Elevation (m)",
  bio_2: "Mean Diurnal Range",
  bio_3: "Isothermality",
  bio_4: "Temperature Seasonality",
  bio_5: "Max Temp Warmest Month",
  bio_6: "Min Temp Coldest Month",
  bio_7: "Temperature Annual Range",
  bio_8: "Mean Temp Wettest Quarter",
  bio_9: "Mean Temp Driest Quarter",
  bio_10: "Mean Temp Warmest Quarter",
  bio_11: "Mean Temp Coldest Quarter",
  bio_13: "Precip Wettest Month",
  bio_14: "Precip Driest Month",
  bio_15: "Precipitation Seasonality",
  bio_16: "Precip Wettest Quarter",
  bio_17: "Precip Driest Quarter",
  bio_18: "Precip Warmest Quarter",
  bio_19: "Precip Coldest Quarter",
};

export const BIO_CODE: Record<string, string> = {
  temperature: "Bio 1",
  rainfall: "Bio 12",
  elevation: "Elev",
  ...Object.fromEntries(
    Array.from({ length: 19 }, (_, i) => [`bio_${i + 1}`, `Bio ${i + 1}`]),
  ),
};

export const label = (f: string) => FEATURE_LABELS[f] ?? f;
