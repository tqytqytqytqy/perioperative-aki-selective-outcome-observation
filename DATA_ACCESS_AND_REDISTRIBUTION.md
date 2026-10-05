# Data access and redistribution boundary

The provider pages linked below were checked on 2026-10-05. This release retains an aggregate-only redistribution boundary irrespective of whether a provider permits wider redistribution.

## INSPIRE 1.4.2

Access requires a credentialed PhysioNet account, required CITI training, and a signed project DUA. The Korea Credentialed Health Data License and Agreement prohibit disclosure to anyone else and make access non-transferable. Official release: https://physionet.org/content/inspire/1.4.2/. Terms: https://physionet.org/content/inspire/view-license/1.4.2/.

## MOVER

Access requires a signed UCI-OR/MOVER data-use agreement, which expressly prohibits redistribution. Official metadata: https://archive.ics.uci.edu/dataset/877/mover-medical-informatics-operating-room-vitals-and-events-repository. Agreement: https://mover.ics.uci.edu/download.html.

## VitalDB

The current provider overview and PhysioNet v1.0.0 record identify CC BY 4.0. The provider's current agreement permits sharing and adaptation with attribution and prohibits attempts to identify individuals. This corrects the older package's access-description wording; it does not change this package's aggregate-only boundary or reinterpret the terms under which any earlier local copy was acquired. Dataset object: https://physionet.org/content/vitaldb/1.0.0/. Terms: https://vitaldb.net/docs/?documentId=OpenDataset%2FOverview.md.

## Applied boundary

This package does not redistribute raw data, row-level derived data, patient identifiers, local file hashes tied to private copies, or serialized model objects. It contains only analysis code, aggregate results, figures, a JSON model specification, and non-sensitive metadata. The public input manifest records versions, official locations, terms, and expected filenames without local paths.

Historical audit summaries that compare earlier and corrected local cohorts do not include the earlier patient-level comparators. They document the correction, but those historical comparisons cannot be rerun from the aggregate summaries alone. Follow the reproduction instructions to rebuild the current cohorts and aggregate outputs from separately obtained source datasets.

Provider terms should be rechecked before each new public release. This file documents the package boundary; it is not legal advice and does not grant redistribution rights for the source datasets.

## Repository licenses

Source code under `scripts/` is licensed under MIT. Original documentation, aggregate tables, figures, model metadata, and workbook content are licensed under CC BY 4.0 except where otherwise noted. These licenses apply only to rights held by the authors; they do not override third-party dataset terms or grant access to provider-controlled data. See `LICENSE`, `LICENSE-CODE`, and `LICENSE-CONTENT`.
