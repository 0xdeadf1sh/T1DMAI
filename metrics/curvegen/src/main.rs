// Dump the app's insulin preset catalogue and each rapid preset's per-5-min action
// curve as JSON. The Loop exponential model stays single-copy in t1dm-core; this only
// reads it out so a probe elsewhere can consume the curve as data.
use serde_json::json;
use t1dm_core::{exp_action_curve, insulin_preset_catalog, InsulinFamily};

fn main() {
    let mut out = Vec::new();
    for spec in insulin_preset_catalog() {
        let rapid = matches!(spec.family, InsulinFamily::RapidExp);
        let curve = if rapid {
            exp_action_curve(1.0, spec.peak_min, spec.dia_min)
        } else {
            Vec::new()
        };
        out.push(json!({
            "label": spec.label,
            "family": if rapid { "rapid_exp" } else { "basal_bateman" },
            "peak_min": spec.peak_min,
            "dia_min": spec.dia_min,
            "ka_per_hour": spec.ka_per_hour,
            "ke_per_hour": spec.ke_per_hour,
            "off_distribution": spec.off_distribution,
            "citation": spec.citation,
            "curve_per_5min_unit_total": curve,
        }));
    }
    println!("{}", serde_json::to_string_pretty(&json!(out)).unwrap());
}
