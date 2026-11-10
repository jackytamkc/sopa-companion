# proseg-live, the patched proseg

`proseg-live` is [proseg](https://github.com/dcjones/proseg) 3.2.0 with `proseg-live.patch` applied: per-iteration
snapshots, pause, step, finish and live settings, and proseg's own count of each cell, for the app's Proseg Live tool.
The app runs it as a separate program.

- **Proseg's own count** (`--live-counts`, with `--live-counts-dir` and `--live-counts-start`): with the frames the app
  asks for (a `counts` line in the control file) and the states proseg starts from, refines, pauses, holds, finishes or
  ends on, it writes each cell's count, gene by gene: assigned, averaged over proseg's random background draw, and
  sampled (the saved table's at the end), plus where exports of a box (`where` lines). A second writer thread writes
  them; `counts.jsonl` is their commit point. Every number the app shows about a cell's RNA reads it. Counting only
  reads the model: seeded runs with and without it save byte-identical results. The files are described in the header of
  the patched tree's `src/live.rs`.
- **Licence.** proseg is Copyright (C) 2024 Daniel C. Jones and licensed under the GNU General Public License v3:
  `LICENSE.md` is proseg's own licence file, as its crate ships it. The patch and the program built from it are derived
  from proseg and covered by the GPL v3 too.
- **Source.** proseg 3.2.0 as crates.io publishes it (<https://crates.io/crates/proseg/3.2.0>), plus
  `proseg-live.patch`, which holds every change to it (its `Cargo.lock`'s too: the build uses exactly those
  dependency versions).
- **For users** it is a package of its own, `proseg-live` (`package/`, version 3.2.0.2: proseg's version with a fourth
  number for the patch), installed with the app. Its source distribution is the patched crate with the patch beside it;
  every release carries it next to the wheels. Its `--version` says `proseg 3.2.0`; the app knows one that keeps
  proseg's own count by its `--help` listing `--live-counts`.
- **For development** `build.sh` makes `bin/proseg-live` here (Rust 1.88 or newer; `--tree`, `--package DIR` and
  `--wheel DIR` make the tree, the package's source, or a wheel).
