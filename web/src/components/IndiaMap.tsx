import { ReactNode } from "react";
import { GeoJSON, MapContainer, TileLayer, ZoomControl } from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import { useData } from "../data";
import { C } from "../theme";

export const INDIA_BOUNDS: LatLngBoundsExpression = [
  [6.5, 67.5],
  [36.5, 97.5],
];

/** Light basemap locked to India, with everything outside India dimmed. */
export function IndiaMap({ children, className }: { children?: ReactNode; className?: string }) {
  const india = useData("india");
  return (
    <MapContainer
      className={className ?? "map"}
      bounds={INDIA_BOUNDS}
      maxBounds={[
        [-5, 50],
        [45, 115],
      ]}
      minZoom={4}
      zoomSnap={0.25}
      zoomDelta={0.5}
      maxZoom={12}
      zoomControl={false}
      preferCanvas
      scrollWheelZoom
    >
      <ZoomControl position="topright" />
      <TileLayer
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        attribution="Tiles &copy; Esri"
        maxNativeZoom={16}
      />
      {india && (
        <GeoJSON
          data={india.mask}
          interactive={false}
          style={{ fillColor: C.cream, fillOpacity: 0.85, weight: 0 }}
        />
      )}
      {children}
      {india && (
        <GeoJSON
          data={india.outline}
          interactive={false}
          style={{ color: C.leafDark, weight: 2, fillOpacity: 0 }}
        />
      )}
      <TileLayer
        url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
        maxNativeZoom={16}
        zIndex={650}
        pane="shadowPane"
      />
    </MapContainer>
  );
}

export function MapLegend({
  title,
  items,
}: {
  title: string;
  items: { label: string; color: string; dashed?: boolean; note?: string }[];
}) {
  return (
    <div className="map-legend">
      <div className="map-legend-title">{title}</div>
      {items.map((it) => (
        <div className="map-legend-row" key={it.label}>
          <span
            className="swatch"
            style={
              it.dashed
                ? { background: "transparent", border: `2px dashed ${it.color}` }
                : { background: it.color }
            }
          />
          <span>{it.label}</span>
          {it.note && <span className="map-legend-note">{it.note}</span>}
        </div>
      ))}
    </div>
  );
}
