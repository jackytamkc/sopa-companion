# proseg-live

proseg 3.2.0 with live mode: per-iteration snapshots, pause, step, finish, live settings and proseg's own count of
every cell and gene, for the SoPA companion's Proseg Live tool. It installs one program, `proseg-live`, and no Python
module.

- **Licence.** proseg is Copyright (C) 2024 Daniel C. Jones and licensed under the GNU General Public License v3
  (`LICENSE.md`). proseg-live is derived from it and covered by the GPL v3 too.
- **Source.** This package's source distribution, `proseg_live-<version>.tar.gz`, is the complete source: proseg 3.2.0
  as crates.io publishes it (https://crates.io/crates/proseg/3.2.0) with `proseg-live.patch` applied (also included,
  so the changes can be read on their own), its `Cargo.lock` (the exact dependency versions), and this file. Every
  release that carries a wheel carries this source distribution beside it.
- **Building it.** pip builds it from the source distribution where no wheel fits: that needs Rust 1.88 or newer and a
  C compiler, as `cargo install proseg` does.
- It does not replace sopa's `proseg`: the program is called `proseg-live`.
