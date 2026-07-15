# Data access and redistribution boundary

Terms were verified against the originating providers on 2026-07-14. They must be checked again at the time of publication.

## INSPIRE 1.4.2

Access requires a credentialed PhysioNet account, required CITI training, and a signed project DUA. The Korea Credentialed Health Data License and Agreement prohibit disclosure to anyone else and make access non-transferable. Official release: https://physionet.org/content/inspire/1.4.2/. Terms: https://physionet.org/content/inspire/view-license/1.4.2/.

## MOVER

Access requires a signed UCI-OR/MOVER data-use agreement, which expressly prohibits redistribution. Official metadata: https://archive.ics.uci.edu/dataset/877/mover-medical-informatics-operating-room-vitals-and-events-repository. Agreement: https://mover.ics.uci.edu/download.html.

## VitalDB

The provider identifies CC BY-NC-SA 4.0 and also imposes a DUA that restricts disclosure without provider consent and makes access rights non-transferable. This package applies the more restrictive operational boundary. Dataset object: https://physionet.org/content/vitaldb/1.0.0/. Terms: https://vitaldb.net/docs/?documentId=OpenDataset%2FOverview.md.

## Applied boundary

This package does not redistribute raw data, row-level derived data, patient identifiers, local file hashes tied to private copies, or serialized model objects. It contains only analysis code, aggregate results, figures, a JSON model specification, and non-sensitive metadata. The public input manifest records versions, official locations, terms, and expected filenames without local paths.

The included `qa/raw_rebuild_equivalence_v32.csv` is frozen internal audit evidence. Its prior locked patient-level comparators are not redistributed, so that historical comparison cannot be regenerated from the public package. Public replay remains available for rebuilding the current cohorts and aggregate analysis outputs from separately obtained source datasets.

Provider terms should be rechecked before each new public release. This file documents the package boundary; it is not legal advice and does not grant redistribution rights for the source datasets.

## Repository licenses

Source code under `scripts/` is licensed under MIT. Original documentation, aggregate tables, figures, model metadata, and workbook content are licensed under CC BY 4.0 except where otherwise noted. These licenses apply only to rights held by the authors; they do not override third-party dataset terms or grant access to provider-controlled data. See `LICENSE`, `LICENSE-CODE`, and `LICENSE-CONTENT`.
