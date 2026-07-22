# Synthetic fixture provenance

All fixtures approved for public tests are generated from first principles or are small, hand-authored software-validation examples. Identifiers are fictional (for example, `synthetic_sample_01`). They contain no patient data, research data, or transformed private staging values and are not biologically interpretable.

Expected effects exist only to validate software behavior. Fixture suites cover null and excluded features, insufficient replication, complete confounding, estimability failure, LODO directional agreement and opposite-direction failure, TF agreement and discordance, and cross-resource consensus. New public fixtures must use deterministic generators and pytest temporary directories where practical.
