/**
 * Heli.OS Domain Configuration Types
 *
 * Each deployment domain (fire, pipeline, SAR, construction, etc.)
 * provides a DomainConfig that drives the entire UI — vocabulary,
 * colors, map layers, command examples, and entity labeling.
 *
 * One codebase, domain-aware configuration.
 */

export interface DomainPalette {
  /** Primary accent color — used for headings, active states, borders */
  accent: string;
  /** Dimmed version of accent — secondary text, inactive states */
  accentDim: string;
  /** Dark version of accent — borders at low opacity, backgrounds */
  accentDark: string;
  /** Alert / warning color */
  warning: string;
  /** Critical / error color */
  critical: string;
  /** Success / nominal color */
  nominal: string;
  /** Active mission / in-progress color */
  active: string;
  /** Background tint (very subtle) for the page background */
  backgroundTint: string;
  /** Panel background */
  panelBg: string;
  /** Border color at low opacity */
  border: string;
  /** Scanline overlay color (the Pip-Boy DNA) */
  scanline: string;
}

export interface EntityLabel {
  /** Display name shown in UI */
  displayName: string;
  /** Short icon/emoji for compact views */
  icon: string;
  /** Color for this entity classification */
  color: string;
}

export interface AssetType {
  /** Internal type key */
  type: string;
  /** Display name */
  label: string;
  /** Icon character */
  icon: string;
}

export interface MapLayerPreset {
  id: string;
  name: string;
  enabled: boolean;
  color: string;
  icon: string;
}

export interface AlertType {
  id: string;
  label: string;
  icon: string;
  color: string;
}

export interface MissionTemplate {
  id: string;
  label: string;
  intent: string;
  description: string;
}

export interface DomainTerminology {
  /** What to call "Mission" — e.g. "Incident" for fire, "Inspection" for pipeline */
  mission: string;
  /** What to call "Asset" — e.g. "Unit" for fire, "Equipment" for construction */
  asset: string;
  /** What to call "Alert" — e.g. "Detection" for fire, "Anomaly" for pipeline */
  alert: string;
  /** What to call "Entity" */
  entity: string;
  /** What to call the operator view */
  operatorView: string;
  /** What to call the supervisor view */
  supervisorView: string;
}

export interface DomainConfig {
  /** Unique domain identifier */
  id: string;
  /** Human-readable domain name */
  name: string;
  /** Short description */
  description: string;
  /** Color palette */
  palette: DomainPalette;
  /** Entity classification → display label mapping */
  entityLabels: Record<string, EntityLabel>;
  /** Expected asset types for the sidebar empty state */
  assetTypes: AssetType[];
  /** Default map layers */
  mapLayers: MapLayerPreset[];
  /** Command bar placeholder examples */
  commandExamples: string[];
  /** Domain-specific alert types */
  alertTypes: AlertType[];
  /** Quick-action mission templates */
  missionTemplates: MissionTemplate[];
  /** UI terminology overrides */
  terminology: DomainTerminology;
}
