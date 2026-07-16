# CoMPASS-OF

CoMPASS-OF extends the CoMPASS framework to open-field behavior with a focus on thigmotactic organization. Rather than modeling reward-directed navigation, this pipeline quantifies how mice structure behavior relative to the arena boundary.

The open-field implementation uses boundary-referenced features such as KDE-based spatial occupancy, distance from the wall, and wall-referenced movement angle to capture behavioral states. This allows open-field behavior to be represented in terms of thigmotactic state structure rather than general exploratory activity alone.

This implementation was developed for analyzing open-field behavior together with PPC single-unit activity. It provides an interpretable state-based description of boundary-guided behavior that can be used to compare animals, quantify spatial-motor organization, and relate thigmotactic behavioral dynamics to neural firing patterns.
