# FastNoiseLite (vendored)

`FastNoiseLite.h`, VERSION 1.0.1, from <https://github.com/Auburn/FastNoise>.
MIT, © 2020 Jordan Peck and contributors — the licence text is embedded at the
top of the header itself, so there is no separate LICENSE file here.

Vendored verbatim, byte-for-byte as shipped in the `pyfastnoiselite` 0.0.5 sdist
(`ext/FastNoise/Cpp/FastNoiseLite.h`), which is the noise kernel terrain-diffusion
generates its synthetic conditioning map with. Taking it from that sdist rather
than from Auburn's repo directly guarantees we match the exact revision upstream
runs against, so a generated world's climate fields agree with upstream's for the
same seed.

**Not the same library as FastNoise2**, which bro vendors elsewhere for procedural
terrain. Same author, different algorithms and different values for the same seed —
they are not interchangeable, and using FastNoise2 here would silently produce a
different planet.

Do not edit. `scripts/terrain_synthetic_parity.sh` gates our sampling against
`pyfastnoiselite` and would catch a drift, but only if this stays pristine.
