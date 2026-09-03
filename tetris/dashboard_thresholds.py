"""Single source for dashboard gates and operational red-flag thresholds."""

THRESHOLDS = {
    "stage4_exact_match": 0.70,
    "stage4_parse_rate": 0.99,
    "stage4_legality_rate": 0.99,
    "stage5_min_mean_lines": 10.0,
    "heartbeat_stale_seconds": 300,
    "orphan_instance_seconds": 900,
    "disk_red_percent_free": 10,
    "disk_amber_percent_free": 20,
    "gpu_low_percent": 10,
}
