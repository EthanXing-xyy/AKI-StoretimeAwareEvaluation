# Official upstream source

- Repository: `https://github.com/Jovinus/aki-timeseries-continuous-monitoring`
- Pinned commit: `6794acb7f615b5864171f8aa7c8ae610049cb947` (`6794acb`)
- Local snapshot: `external/aki-timeseries-continuous-monitoring`

The directory is a byte-for-byte copy of files tracked by the clean upstream working tree at the pinned commit. Its nested `.git` history is intentionally excluded so this project can later be published as one repository.

Step 1 loads the upstream LSTM-Attention, Masked CNN, and ITE-Transformer architecture implementations. Step 2 reuses upstream reference-time matching and downstream clinical-faithfulness/alert-burden components. Step 4 applies the project's availability adapter around the same frozen inference path.

Project-specific cohort construction, path handling, validation gates, and availability rules live in the numbered step directories. Do not modify the upstream source; use wrappers or adapters.

The upstream `LICENSE` file is retained unchanged inside the snapshot. Third-party upstream code remains governed by that license and is not relicensed by the project-level MIT License.
