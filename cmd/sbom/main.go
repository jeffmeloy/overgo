package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/clioptions"
)

const (
	projectVersion = "0.1.0"
	upstreamCommit = "42fc243060709331ff9b158a9ed2cbe37219ae83"
	sbomPath       = "SBOM.cdx.json"
)

var kernelFiles = []string{
	"internal/cuda/kernel/ops_f32.ptx",
	"internal/cuda/kernel/vector_add.ptx",
	"internal/quant/iq_tables_generated.go",
	"kernels/cuda/iq_tables_generated.cuh",
	"kernels/cuda/ops_f32.cu",
	"kernels/cuda/vector_add.cu",
	"kernels/manifest.json",
}

type module struct {
	Path    string
	Version string
	Main    bool
}

func main() {
	clioptions.Main(run)
}

func run() error {
	check := flag.Bool("check", false, "verify SBOM.cdx.json is current")
	update := flag.Bool("update", false, "write generated SBOM.cdx.json")
	flag.Parse()
	if flag.NArg() != 0 || *check && *update {
		return errors.New("usage: sbom [-check|-update]")
	}
	data, err := generate(".")
	if err != nil {
		return err
	}
	return clioptions.OutputGenerated(
		data, sbomPath, *check, *update,
		"SBOM.cdx.json is stale; regenerate with: go run ./cmd/sbom -update",
		os.Stdout,
	)
}

func generate(root string) ([]byte, error) {
	modules, err := listModules(root)
	if err != nil {
		return nil, err
	}
	components := []map[string]any{
		component("library", "llama.cpp compatibility baseline", upstreamCommit, "MIT", "pkg:github/ggml-org/llama.cpp@"+upstreamCommit),
		component("framework", "Go standard library", "1.26", "BSD-3-Clause", "pkg:golang/stdlib@1.26"),
		component("framework", "NVIDIA CUDA Driver API", "13.2-compatible", "NVIDIA Software License Agreement", ""),
		component("framework", "NVIDIA CUDA Toolkit", "12.9.86", "NVIDIA Software License Agreement", ""),
	}
	dependsOn := []string{
		"pkg:github/ggml-org/llama.cpp@" + upstreamCommit,
		"pkg:golang/stdlib@1.26",
		"component:nvidia-cuda-driver-api",
		"component:nvidia-cuda-toolkit",
	}
	for _, item := range modules {
		if item.Main {
			continue
		}
		version := item.Version
		if version == "" {
			version = "unknown"
		}
		purl := "pkg:golang/" + item.Path + "@" + version
		license := moduleLicense(item.Path)
		components = append(components, component("library", item.Path, version, license, purl))
		dependsOn = append(dependsOn, purl)
	}
	for _, path := range kernelFiles {
		data, readErr := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if readErr != nil {
			return nil, readErr
		}
		sum := sha256.Sum256(data)
		ref := "file:" + path
		license := "NOASSERTION"
		if path == "internal/quant/iq_tables_generated.go" ||
			path == "kernels/cuda/iq_tables_generated.cuh" {
			license = "MIT"
		}
		components = append(components, map[string]any{
			"type":    "file",
			"name":    path,
			"bom-ref": ref,
			"hashes": []map[string]string{{
				"alg":     "SHA-256",
				"content": hex.EncodeToString(sum[:]),
			}},
			"licenses": []map[string]any{{"license": map[string]string{"name": license}}},
		})
		dependsOn = append(dependsOn, ref)
	}
	sort.Slice(components, func(i, j int) bool {
		return fmt.Sprint(components[i]["bom-ref"]) < fmt.Sprint(components[j]["bom-ref"])
	})
	sort.Strings(dependsOn)
	document := map[string]any{
		"bomFormat":    "CycloneDX",
		"specVersion":  "1.6",
		"version":      1,
		"components":   components,
		"dependencies": []map[string]any{{"ref": "pkg:golang/overgo", "dependsOn": dependsOn}},
		"metadata": map[string]any{
			"component": component("application", "overgo", projectVersion, "NOASSERTION", "pkg:golang/overgo"),
			"properties": []map[string]string{
				{"name": "overgo:cgo", "value": "false"},
				{"name": "overgo:cuda-target", "value": "compute_89"},
				{"name": "overgo:upstream-commit", "value": upstreamCommit},
			},
			"tools": map[string]any{"components": []map[string]any{
				component("application", "overgo sbom generator", "1", "NOASSERTION", ""),
			}},
		},
	}
	output, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(output, '\n'), nil
}

func moduleLicense(path string) string {
	switch path {
	case "github.com/dlclark/regexp2/v2":
		return "MIT"
	default:
		return "NOASSERTION"
	}
}

func component(kind, name, version, license, purl string) map[string]any {
	ref := purl
	if ref == "" {
		ref = "component:" + strings.NewReplacer(" ", "-", ".", "-").Replace(strings.ToLower(name))
	}
	result := map[string]any{
		"type":     kind,
		"name":     name,
		"version":  version,
		"bom-ref":  ref,
		"licenses": []map[string]any{{"license": map[string]string{"name": license}}},
	}
	if purl != "" {
		result["purl"] = purl
	}
	return result
}

func listModules(root string) ([]module, error) {
	command := exec.Command("go", "list", "-m", "-json", "all")
	command.Dir = root
	output, err := command.Output()
	if err != nil {
		return nil, fmt.Errorf("sbom: go list modules: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(output))
	var result []module
	for {
		var item module
		if err := decoder.Decode(&item); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, err
		}
		result = append(result, item)
	}
	if len(result) == 0 {
		return nil, errors.New("sbom: module list is empty")
	}
	return result, nil
}
