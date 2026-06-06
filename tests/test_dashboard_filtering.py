from sak.reporting.dashboard import DEFAULT_DASHBOARD_VARIANTS, dashboard_variant_groups


def test_default_dashboard_variants_are_global_selected_models() -> None:
    assert DEFAULT_DASHBOARD_VARIANTS == (
        "pca_global",
        "dense_autoencoder_global",
        "tcn_autoencoder_global",
    )


def test_mode_aware_variants_are_advanced_by_default() -> None:
    groups = dashboard_variant_groups(
        {
            "dataset": {},
            "pca_global": {},
            "dense_autoencoder_global": {},
            "tcn_autoencoder_global": {},
            "pca_mode_aware": {},
            "tcn_autoencoder_mode_aware": {},
        }
    )

    assert groups["default"] == list(DEFAULT_DASHBOARD_VARIANTS)
    assert "pca_mode_aware" in groups["advanced"]
    assert "tcn_autoencoder_mode_aware" in groups["advanced"]
