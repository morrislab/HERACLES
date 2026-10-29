# RapidNJ provenance

- Upstream: <https://github.com/johnlees/rapidnj>
- Pinned commit: `65e7aa7f6817beb904573959bafff56a86916495`
- Imported: 2026-08-18
- License: GNU General Public License, version 2; see `LICENSE`.

Local patches (2026-09-09), so the source builds on every host:

- `src/sse_compat.h` (new): scalar stand-ins for the 11 SSE2 intrinsics used by
  the alignment-to-distance code (`bitDistanceGap`, `bitDistanceProtein`, the
  sequence loaders).  HERACLES only runs RapidNJ on precomputed distance
  matrices (`-i pd`), so that code is never executed here.
- `src/stdinclude.h`: include `<emmintrin.h>` on x86 only, `sse_compat.h` elsewhere.
- `src/distanceCalculation/bitDistanceProtein.cpp`: same include switch.

The binary is built by `heracles.rapidnj.build_rapidnj` (at `pip install`
time via `setup.py`, or on first use into the user's cache directory); the
upstream `Makefile` is kept for reference only.
