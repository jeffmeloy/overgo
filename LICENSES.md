# License inventory

The machine-readable inventory is `SBOM.cdx.json`.

| Component | License status |
| --- | --- |
| llamacpp2go Go source | `NOASSERTION` — the repository owner has not declared a project license |
| Project-owned CUDA C++ sources and generated PTX | `NOASSERTION` |
| Pinned llama.cpp compatibility baseline | MIT |
| Generated IQ codebook tables extracted from pinned `ggml-common.h` | MIT |
| Adaptive/Muon optimizer mathematics adapted from `adaptive_new` commit `9d4b38364c2cacba3ee99b998bd84d4645bfbe46` | MIT |
| Go standard library | BSD-3-Clause |
| `golang.org/x/text` Unicode normalization tables/runtime | BSD-3-Clause |
| `github.com/nikolalohinski/gonja/v2` Jinja runtime | MIT |
| `github.com/dlclark/regexp2/v2` bounded ECMAScript regex runtime | MIT |
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

## adaptive_new optimizer notice

The optimizer mathematics in `internal/optimizer` is adapted from
`adaptive_new`, Copyright (c) 2026 Jeff Meloy, under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Regenerate the CycloneDX artifact after module or kernel changes:

```powershell
go run ./cmd/sbom > SBOM.cdx.json
go run ./cmd/sbom -check
```
