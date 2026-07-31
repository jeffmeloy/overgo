# License inventory

The machine-readable inventory is `SBOM.cdx.json`.

| Component | License status |
| --- | --- |
| llamacpp2go Go source | `NOASSERTION` — the repository owner has not declared a project license |
| Project-owned CUDA C++ sources and generated PTX | `NOASSERTION` |
| Pinned llama.cpp compatibility baseline | MIT |
| Generated IQ codebook tables extracted from pinned `ggml-common.h` | MIT |
| Go standard library | BSD-3-Clause |
| `golang.org/x/text` Unicode normalization tables/runtime | BSD-3-Clause |
| `github.com/nikolalohinski/gonja/v2` Jinja runtime | MIT |
| `github.com/dustin/go-humanize` | MIT |
| `github.com/json-iterator/go` | MIT |
| `github.com/modern-go/concurrent` and `reflect2` | Apache-2.0 |
| `github.com/pkg/errors` | BSD-2-Clause |
| `github.com/sirupsen/logrus` | MIT |
| `golang.org/x/exp` and `golang.org/x/sys` | BSD-3-Clause |
| NVIDIA CUDA Driver API and CUDA Toolkit | NVIDIA Software License Agreement |
| GGUF model files | Not distributed by this repository; governed by each model's license |

`NOASSERTION` is intentional and must not be replaced with a guessed license.
Before distributing binaries, the repository owner must choose a project
license or otherwise document distribution rights for the Go and kernel code.

Regenerate the CycloneDX artifact after module or kernel changes:

```powershell
go run ./cmd/sbom > SBOM.cdx.json
go run ./cmd/sbom -check
```
