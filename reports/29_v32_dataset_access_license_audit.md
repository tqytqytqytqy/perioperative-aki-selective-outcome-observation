# v3.2 dataset access and redistribution audit

Verified against the originating providers on 2026-07-14. This audit defines a conservative repository boundary; it is not legal advice and does not replace author or institutional approval.

## INSPIRE 1.4.2

- Official release: https://physionet.org/content/inspire/1.4.2/
- Current terms: https://physionet.org/content/inspire/view-license/1.4.2/
- Access requires a credentialed PhysioNet account, required CITI training, and a signed project DUA.
- The Korea Credentialed Health Data License and Agreement restrict use to research, prohibit disclosure to anyone else, and make access non-transferable.
- Release decision: no INSPIRE raw or row-level derived data may enter the public package.

## MOVER

- Official metadata and DOI: https://archive.ics.uci.edu/dataset/877/mover-medical-informatics-operating-room-vitals-and-events-repository
- Current access agreement: https://mover.ics.uci.edu/download.html
- Access requires a signed MOVER data-use agreement. The agreement expressly prohibits redistribution.
- Release decision: no MOVER raw or row-level derived data may enter the public package.

## VitalDB

- Dataset object: https://physionet.org/content/vitaldb/1.0.0/
- Current provider terms and DUA: https://vitaldb.net/docs/?documentId=OpenDataset%2FOverview.md
- The provider identifies CC BY-NC-SA 4.0 and also imposes DUA restrictions on disclosure and transfer. The more restrictive operational boundary is used here.
- Release decision: no VitalDB raw or row-level derived data may enter the public package.

## Package boundary

The local release candidate contains analysis code, aggregate tables, figures, a JSON model specification, and non-sensitive metadata. It excludes raw archives and source tables, patient- or operation-level derived data, local paths and private hashes, serialized Python model objects, and submission administration files. Actual publication remains blocked until the authors and institution approve release and verify the provider terms again at publication time.
