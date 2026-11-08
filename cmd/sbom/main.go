// Command sbom writes and checks SBOM.cdx.json: the third-party code the build compiles in and what its binaries load.
// No model is ever distributed, and code ported or generated into overgo is overgo's own, so neither is a component:
// the SBOM lists the Go standard library, the third-party Go modules the packages compile in, each with its licence,
// and NVIDIA's driver and the CUDA libraries the binaries load, as external requirements the machine provides.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gosource"
)

const (
	projectVersion = "0.1.0"
	sbomPath       = "SBOM.cdx.json"
	// goVersion is the language version go.mod declares, the standard library the build compiles in.
	goVersion = "1.26"
)

// moduleLicenses are the licences of the third-party modules the build compiles in, as each module's own licence file
// states. A compiled-in module not named here is refused, so a dependency enters the build with its licence or not at
// all.
var moduleLicenses = map[string]string{
	"github.com/dlclark/regexp2/v2": "MIT",
	"github.com/icza/bitio":         "Apache-2.0",
	"github.com/mewkiz/flac":        "Unlicense",
	"github.com/mewkiz/pkg":         "Unlicense",
	"github.com/mewpkg/term":        "Unlicense",
}

// moduleNotices are the notices in licenses/ of the third-party modules the build compiles in, which go with its
// binaries (cmd/release): flac and its helpers share the Unlicense.
var moduleNotices = map[string]string{
	"github.com/dlclark/regexp2/v2": "regexp2-MIT.txt",
	"github.com/icza/bitio":         "bitio-Apache-2.0.txt",
	"github.com/mewkiz/flac":        "Unlicense.txt",
	"github.com/mewkiz/pkg":         "Unlicense.txt",
	"github.com/mewpkg/term":        "Unlicense.txt",
}

// requirement is a library the machine provides and the binaries load: none ships with overgo.
type requirement struct {
	name, version, license, scope string
	files                         []string
}

// requirements are what the binaries load from NVIDIA: the driver and its management library always, cuBLAS for the
// device's matrix products, and cuDNN where it is installed, for the convolutions that repay it.
var requirements = []requirement{
	{name: "NVIDIA CUDA driver", version: "13.2-compatible", license: "NVIDIA Software License Agreement", scope: "required", files: []string{"nvcuda.dll", "nvml.dll"}},
	{name: "NVIDIA cuBLAS", version: "12", license: "NVIDIA Software License Agreement", scope: "required", files: []string{"cublas64_12.dll", "cublasLt64_12.dll"}},
	{name: "NVIDIA cuDNN", version: "9", license: "NVIDIA Software License Agreement", scope: "optional", files: []string{"cudnn64_9.dll"}},
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
	data, err := generate(".", moduleLicenses)
	if err != nil {
		return err
	}
	return clioptions.OutputGenerated(
		data, sbomPath, *check, *update,
		"SBOM.cdx.json is stale; regenerate with: go run ./cmd/sbom -update",
		os.Stdout,
	)
}

// generate is the SBOM of the packages of root, each compiled-in module's licence read from licenses.
func generate(root string, licenses map[string]string) ([]byte, error) {
	modules, err := compiledModules(root)
	if err != nil {
		return nil, err
	}
	standard := "pkg:golang/stdlib@" + goVersion
	components := []map[string]any{component("framework", "Go standard library", goVersion, "BSD-3-Clause", standard)}
	dependsOn := []string{standard}
	for _, item := range modules {
		license, known := licenses[item.path]
		if !known {
			return nil, fmt.Errorf("sbom: the build compiles in %s, whose licence is not declared; add it to moduleLicenses from the module's licence file", item.path)
		}
		notice, noted := moduleNotices[item.path]
		if !noted {
			return nil, fmt.Errorf("sbom: the build compiles in %s, which has no notice in licenses/; add its licence file there and to moduleNotices", item.path)
		}
		purl := "pkg:golang/" + item.path + "@" + item.version
		entry := component("library", item.path, item.version, license, purl)
		entry["properties"] = []map[string]string{{"name": "overgo:notice", "value": "licenses/" + notice}}
		components = append(components, entry)
		dependsOn = append(dependsOn, purl)
	}
	for _, library := range requirements {
		entry := component("framework", library.name, library.version, library.license, "")
		entry["scope"] = library.scope
		entry["properties"] = []map[string]string{
			{"name": "overgo:provided-by", "value": "the machine; overgo ships none of it"},
			{"name": "overgo:loads", "value": strings.Join(library.files, ", ")},
		}
		components = append(components, entry)
		dependsOn = append(dependsOn, fmt.Sprint(entry["bom-ref"]))
	}
	slices.SortFunc(components, func(left, right map[string]any) int {
		return strings.Compare(fmt.Sprint(left["bom-ref"]), fmt.Sprint(right["bom-ref"]))
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

// module is a third-party module the build compiles in, at the version it builds.
type module struct {
	path, version string
}

// compiledModules are the third-party modules the packages of root compile in: those the package graph reaches from
// them, tests aside (gosource.ListGraph), which is what a binary holds. go list -m all lists the whole module graph,
// modules no package compiles in among them. A package of a third-party module carries its version; the standard
// library carries no module, and overgo's own packages no version.
func compiledModules(root string) ([]module, error) {
	graph, err := gosource.ListGraph(context.Background(), root, true, "./...")
	if err != nil {
		return nil, fmt.Errorf("sbom: the compiled packages: %w", err)
	}
	var modules []module
	for _, pkg := range graph {
		if pkg.Module != nil && pkg.Module.Version != "" && !slices.Contains(modules, module{pkg.Module.Path, pkg.Module.Version}) {
			modules = append(modules, module{pkg.Module.Path, pkg.Module.Version})
		}
	}
	slices.SortFunc(modules, func(left, right module) int { return strings.Compare(left.path, right.path) })
	return modules, nil
}
