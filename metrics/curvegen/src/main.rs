// Dump the insulin preset catalogue and each preset's per-5-min action curve as JSON.
use serde_json::json;
use t1dm_core::{insulin_preset_catalog, preset_curve, InsulinFamily};

/// A rapid curve's shape depends on the dose; this is SPEC/invariants.md §5's reference dose.
const REFERENCE_UNITS: f64 = 5.0;

fn main() {
    let mut out = Vec::new();
    for spec in insulin_preset_catalog() {
        let rapid = matches!(spec.family, InsulinFamily::RapidGamma);
        let curve: Vec<f64> = preset_curve(REFERENCE_UNITS, spec.clone())
            .iter()
            .map(|v| v / REFERENCE_UNITS)
            .collect();
        out.push(json!({
            "label": spec.label,
            "family": if rapid { "rapid_gamma" } else { "basal_bateman" },
            "gamma_k": spec.gamma_k,
            "gamma_theta": spec.gamma_theta,
            "dia_base_hours": spec.dia_base_hours,
            "ka_per_hour": spec.ka_per_hour,
            "ke_per_hour": spec.ke_per_hour,
            "action_min": spec.action_min,
            "citation": spec.citation,
            "curve_per_5min_unit_total": curve,
        }));
    }
    println!("{}", serde_json::to_string_pretty(&json!(out)).unwrap());
}
