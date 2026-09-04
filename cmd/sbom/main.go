package main

import (
	"bytes"
	"cmp"
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
	"slices"
	"sort"
	"strings"

	"overgo/internal/clioptions"
)

const (
	projectVersion = "0.1.0"
	sbomPath       = "SBOM.cdx.json"
	kernelManifest = "kernels/manifest.json"
)

type kernelInventory struct {
	Modules []struct {
		Source             string `json:"source"`
		Asset              string `json:"asset"`
		SourceDependencies []struct {
			Path string `json:"path"`
		} `json:"sourceDependencies"`
	} `json:"modules"`
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
	kernelFiles, err := usedKernelFiles(root)
	if err != nil {
		return nil, err
	}
	components := []map[string]any{
		component("framework", "Go standard library", "1.26", "BSD-3-Clause", "pkg:golang/stdlib@1.26"),
		component("framework", "NVIDIA CUDA Driver API", "13.2-compatible", "NVIDIA Software License Agreement", ""),
		component("framework", "NVIDIA CUDA Toolkit", "12.9.86", "NVIDIA Software License Agreement", ""),
	}
	dependsOn := []string{
		"pkg:golang/stdlib@1.26",
		"component:nvidia-cuda-driver-api",
		"component:nvidia-cuda-toolkit",
	}
	for _, item := range modules {
		if item.Main {
			continue
		}
		version := item.Version
		version = cmp.Or(version, "unknown")
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
		license := ""
		if path == "internal/quant/iq_tables_generated.go" ||
			path == "kernels/cuda/iq_tables_generated.cuh" {
			license = "MIT"
		}
		entry := map[string]any{
			"type":    "file",
			"name":    path,
			"bom-ref": ref,
			"hashes": []map[string]string{{
				"alg":     "SHA-256",
				"content": hex.EncodeToString(sum[:]),
			}},
		}
		if license != "" {
			entry["licenses"] = []map[string]any{{"license": map[string]string{"name": license}}}
		}
		components = append(components, entry)
		dependsOn = append(dependsOn, ref)
	}
	sort.Slice(components, func(i, j int) bool {
		return fmt.Sprint(components[i]["bom-ref"]) < fmt.Sprint(components[j]["bom-ref"])
	})
	slices.Sort(dependsOn)
	document := map[string]any{
		"bomFormat":    "CycloneDX",
		"specVersion":  "1.6",
		"version":      1,
		"components":   components,
		"dependencies": []map[string]any{{"ref": "pkg:golang/overgo", "dependsOn": dependsOn}},
		"metadata": map[string]any{
			"component": component("application", "overgo", projectVersion, "", "pkg:golang/overgo"),
			"properties": []map[string]string{
				{"name": "overgo:cgo", "value": "false"},
				{"name": "overgo:cuda-target", "value": "compute_89"},
			},
			"tools": map[string]any{"components": []map[string]any{
				component("application", "overgo sbom generator", "1", "", ""),
			}},
		},
	}
	output, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(output, '\n'), nil
}

func usedKernelFiles(root string) ([]string, error) {
	raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(kernelManifest)))
	if err != nil {
		return nil, err
	}
	var inventory kernelInventory
	if err := json.Unmarshal(raw, &inventory); err != nil {
		return nil, fmt.Errorf("sbom: decode kernel manifest: %w", err)
	}
	paths := map[string]struct{}{
		kernelManifest:                          {},
		"internal/quant/iq_tables_generated.go": {},
	}
	for _, module := range inventory.Modules {
		for _, path := range []string{module.Source, module.Asset} {
			if path == "" {
				return nil, errors.New("sbom: kernel manifest has empty source or asset")
			}
			paths[path] = struct{}{}
		}
		for _, dependency := range module.SourceDependencies {
			if dependency.Path == "" {
				return nil, errors.New("sbom: kernel manifest has empty source dependency")
			}
			paths[dependency.Path] = struct{}{}
		}
	}
	var result []string
	for path := range paths {
		result = append(result, path)
	}
	slices.Sort(result)
	return result, nil
}

func moduleLicense(path string) string {
	switch path {
	case "github.com/dlclark/regexp2/v2":
		return "MIT"
	case "github.com/icza/bitio":
		return "Apache-2.0"
	case "github.com/mewkiz/flac", "github.com/mewkiz/pkg", "github.com/mewpkg/term":
		return "Unlicense"
	default:
		return ""
	}
}

func component(kind, name, version, license, purl string) map[string]any {
	ref := purl
	if ref == "" {
		ref = "component:" + strings.NewReplacer(" ", "-", ".", "-").Replace(strings.ToLower(name))
	}
	result := map[string]any{"type": kind, "name": name, "version": version, "bom-ref": ref}
	if license != "" {
		result["licenses"] = []map[string]any{{"license": map[string]string{"name": license}}}
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
