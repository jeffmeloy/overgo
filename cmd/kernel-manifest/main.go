package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	cudaKernel "overgo/internal/cuda/kernel"
	"overgo/internal/strictjson"
)

const manifestPath = "kernels/manifest.json"

// compilerWords is how many words the compiler line ends in: release, its version, and the compiler's.
const compilerWords = 3

// compilerLine is what a PTX file's comment says of the compiler that wrote it, release 13.0, V13.0.88.
func compilerLine(data []byte) (string, bool) {
	for line := range strings.Lines(string(data)) {
		named, found := strings.CutPrefix(line, "// Cuda compilation tools, ")
		if !found {
			continue
		}
		named = strings.TrimSpace(named)
		words := strings.Fields(named)
		if len(words) == compilerWords && words[0] == "release" && strings.HasSuffix(words[1], ",") && strings.HasPrefix(words[2], "V") {
			return named, true
		}
	}
	return "", false
}

// projectIncludes are the headers a source includes by quotes: a system header's angle brackets name nothing the manifest pins.
func projectIncludes(data []byte) []string {
	var included []string
	for line := range strings.Lines(string(data)) {
		directive, found := strings.CutPrefix(strings.TrimLeft(line, " \t"), "#")
		if !found {
			continue
		}
		target, found := strings.CutPrefix(strings.TrimLeft(directive, " \t"), "include")
		if !found {
			continue
		}
		quoted, found := strings.CutPrefix(strings.TrimLeft(target, " \t"), `"`)
		if !found {
			continue
		}
		if name, _, closed := strings.Cut(quoted, `"`); closed && name != "" {
			included = append(included, name)
		}
	}
	return included
}

// ptxEntry is one kernel entry of a PTX file: its name and the lines that declare its parameters.
type ptxEntry struct {
	name       string
	parameters []string
}

// ptxEntryList reads a PTX file's entries, each from its .visible .entry line to the line that closes it.
func ptxEntryList(data []byte) []ptxEntry {
	var entries []ptxEntry
	open := false
	for line := range strings.Lines(string(data)) {
		if head, found := strings.CutPrefix(line, ".visible .entry "); found {
			name, rest, _ := strings.Cut(head, "(")
			entries = append(entries, ptxEntry{name: name})
			open = !strings.HasPrefix(rest, ")")
			continue
		}
		if !open {
			continue
		}
		if strings.HasPrefix(line, ")") {
			open = false
			continue
		}
		entries[len(entries)-1].parameters = append(entries[len(entries)-1].parameters, line)
	}
	return entries
}

// parameterWords is the fewest words a parameter line holds: .param, its type and its name.
const parameterWords = 3

// parameterLayout is a parameter's declaration without its name: .u64, or .align 4 .b8[16].
func parameterLayout(line string) string {
	words := strings.Fields(strings.TrimRight(strings.TrimSpace(line), ","))
	if len(words) < parameterWords || words[0] != ".param" {
		return ""
	}
	last := len(words) - 1
	_, extent, sized := strings.Cut(words[last], "[")
	layout := strings.Join(words[1:last], " ")
	if sized {
		layout += "[" + extent
	}
	return layout
}

type manifest struct {
	Schema          int      `json:"schema"`
	ABIVersion      int      `json:"abiVersion"`
	CUDAToolkit     string   `json:"cudaToolkit"`
	NVCC            string   `json:"nvcc"`
	Target          string   `json:"target"`
	CompilerFlags   []string `json:"compilerFlags"`
	DefaultThreads  int      `json:"defaultThreads"`
	VAEConvTile     int      `json:"vaeConvTile"`
	SharedMemoryABI string   `json:"sharedMemoryABI"`
	Modules         []module `json:"modules"`
}

type module struct {
	Executor           bool               `json:"executor,omitempty"`
	Source             string             `json:"source"`
	SourceSHA256       string             `json:"sourceSha256"`
	SourceDependencies []sourceDependency `json:"sourceDependencies,omitempty"`
	Asset              string             `json:"asset"`
	AssetSHA256        string             `json:"assetSha256"`
	// CompilerFlags: optional per-module nvcc flag override recorded when a
	// module deviates from the manifest-global compilerFlags. torch_cuda_randn
	// omits -use_fast_math for bit-exact curand normals.
	CompilerFlags   []string         `json:"compilerFlags,omitempty"`
	Functions       []string         `json:"functions"`
	ArgumentLayouts []argumentLayout `json:"argumentLayouts"`
}

type sourceDependency struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

type argumentLayout struct {
	Parameters []string `json:"parameters"`
	Functions  []string `json:"functions"`
}

func main() {
	update := flag.Bool("update", false, "update source and asset hashes")
	flag.Parse()
	if flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: kernel-manifest [-update]")
		os.Exit(1)
	}
	if err := run(".", *update); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(root string, update bool) error {
	manifestFile := filepath.Join(root, filepath.FromSlash(manifestPath))
	data, err := os.ReadFile(manifestFile)
	if err != nil {
		return err
	}
	var document manifest
	if err := strictjson.DecodeBytes(data, &document); err != nil {
		return fmt.Errorf("kernel manifest: %w", err)
	}
	if document.Schema != 2 || document.ABIVersion < 1 {
		return errors.New("kernel manifest: unsupported schema or ABI version")
	}
	if document.Target != "compute_89" || document.DefaultThreads <= 0 ||
		document.VAEConvTile <= 0 || document.SharedMemoryABI == "" {
		return errors.New("kernel manifest: target or launch ABI is incomplete")
	}
	// An update runs before the Go bindings are regenerated from it (build-kernels), so only a check holds them to it.
	if !update && (document.ABIVersion != cudaKernel.BundleABIVersion ||
		document.Target != cudaKernel.BundleTarget ||
		document.DefaultThreads != cudaKernel.BundleDefaultThreads ||
		document.VAEConvTile != cudaKernel.BundleVAEConvTile) {
		return errors.New("kernel manifest: Go host kernel ABI identity is stale")
	}
	runtimePins := map[string]string{
		"internal/cuda/kernel/vector_add.ptx":    cudaKernel.VectorAddSHA256,
		"internal/cuda/kernel/ops_f32.ptx":       cudaKernel.OpsF32SHA256,
		"internal/cuda/kernel/ops_native.ptx":    cudaKernel.OpsNativeSHA256,
		"internal/cuda/kernel/strata_qsa.ptx":    cudaKernel.StrataQsaSHA256,
		"internal/cuda/kernel/strata_native.ptx": cudaKernel.StrataNativeSHA256,
	}
	seenSource := map[string]struct{}{}
	seenAsset := map[string]struct{}{}
	for index := range document.Modules {
		item := &document.Modules[index]
		if _, duplicate := seenSource[item.Source]; duplicate {
			return fmt.Errorf("kernel manifest: duplicate source %q", item.Source)
		}
		if _, duplicate := seenAsset[item.Asset]; duplicate {
			return fmt.Errorf("kernel manifest: duplicate asset %q", item.Asset)
		}
		seenSource[item.Source] = struct{}{}
		seenAsset[item.Asset] = struct{}{}
		sourceHash, err := fileHash(root, item.Source)
		if err != nil {
			return err
		}
		assetData, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(item.Asset)))
		if err != nil {
			return err
		}
		// An update hashes what it finds, so it is refused too: the manifest would record code it did not describe.
		if err := requireToolchain(document, item.Asset, assetData); err != nil {
			return err
		}
		sum := sha256.Sum256(assetData)
		assetHash := hex.EncodeToString(sum[:])
		if update {
			item.SourceSHA256 = sourceHash
			item.AssetSHA256 = assetHash
		} else if item.SourceSHA256 != sourceHash || item.AssetSHA256 != assetHash {
			return fmt.Errorf("kernel manifest: hashes for %q are stale", item.Asset)
		}
		included, err := includedSources(root, item.Source)
		if err != nil {
			return err
		}
		if update {
			item.SourceDependencies = pinnedAs(included, item.SourceDependencies)
		} else if err := requirePinned(item, included); err != nil {
			return err
		}
		for dependencyIndex := range item.SourceDependencies {
			dependency := &item.SourceDependencies[dependencyIndex]
			dependencyHash, dependencyErr := fileHash(root, dependency.Path)
			if dependencyErr != nil {
				return dependencyErr
			}
			if update {
				dependency.SHA256 = dependencyHash
			} else if dependency.SHA256 != dependencyHash {
				return fmt.Errorf(
					"kernel manifest: source dependency %q is stale",
					dependency.Path,
				)
			}
		}
		if pinned, ok := runtimePins[item.Asset]; ok && pinned != assetHash && !update {
			return fmt.Errorf("kernel manifest: Go host hash for %q is stale", item.Asset)
		}
		gotFunctions := ptxEntries(assetData)
		wantFunctions := append([]string(nil), item.Functions...)
		slices.Sort(wantFunctions)
		if !slices.Equal(gotFunctions, wantFunctions) {
			return fmt.Errorf(
				"kernel manifest: PTX entries for %q differ: got %v want %v",
				item.Asset,
				gotFunctions,
				wantFunctions,
			)
		}
		gotLayouts := ptxEntryABIs(assetData)
		if update {
			item.ArgumentLayouts = groupArgumentLayouts(item.Functions, gotLayouts)
		}
		if err := verifyArgumentLayouts(item, gotLayouts); err != nil {
			return fmt.Errorf("kernel manifest: argument ABI for %q: %w", item.Asset, err)
		}
	}
	if !update {
		return nil
	}
	output, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return err
	}
	return clioptions.WriteOutputFile(manifestFile, append(output, '\n'))
}

// requireToolchain holds a compiled asset to the toolchain the manifest records.
func requireToolchain(document manifest, asset string, data []byte) error {
	named, found := compilerLine(data)
	if !found {
		return fmt.Errorf("kernel manifest: %q names no compiler", asset)
	}
	if recorded := "release " + document.CUDAToolkit + ", V" + document.NVCC; named != recorded {
		return fmt.Errorf("kernel manifest: %q was compiled by %s, and the manifest records %s", asset, named, recorded)
	}
	return nil
}

// includedSources walks a kernel source's project includes, each relative to the file that names it, so a header compiled
// into a module is pinned without anyone listing it.
func includedSources(root, source string) ([]string, error) {
	seen := map[string]bool{}
	var walk func(file string) error
	walk = func(file string) error {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(file)))
		if err != nil {
			return err
		}
		for _, name := range projectIncludes(data) {
			included := path.Join(path.Dir(file), name)
			if seen[included] {
				continue
			}
			seen[included] = true
			if err := walk(included); err != nil {
				return err
			}
		}
		return nil
	}
	if err := walk(source); err != nil {
		return nil, err
	}
	return slices.Sorted(maps.Keys(seen)), nil
}

// pinnedAs is the included files as dependencies, each keeping its pin until the update hashes it.
func pinnedAs(included []string, pinned []sourceDependency) []sourceDependency {
	dependencies := make([]sourceDependency, 0, len(included))
	for _, file := range included {
		at := slices.IndexFunc(pinned, func(dependency sourceDependency) bool { return dependency.Path == file })
		if at >= 0 {
			dependencies = append(dependencies, pinned[at])
		} else {
			dependencies = append(dependencies, sourceDependency{Path: file})
		}
	}
	return dependencies
}

// requirePinned refuses a module whose pins and includes differ: an unpinned header could change the compiled code with the
// manifest still reading current.
func requirePinned(item *module, included []string) error {
	pinned := make([]string, 0, len(item.SourceDependencies))
	for _, dependency := range item.SourceDependencies {
		pinned = append(pinned, dependency.Path)
	}
	slices.Sort(pinned)
	for _, file := range included {
		if !slices.Contains(pinned, file) {
			return fmt.Errorf("kernel manifest: %q includes %q, which the manifest does not pin; run kernel-manifest -update", item.Source, file)
		}
	}
	for _, file := range pinned {
		if !slices.Contains(included, file) {
			return fmt.Errorf("kernel manifest: %q pins %q, which it does not include; run kernel-manifest -update", item.Source, file)
		}
	}
	return nil
}

func verifyArgumentLayouts(item *module, got map[string][]string) error {
	want := make(map[string][]string, len(item.Functions))
	for _, layout := range item.ArgumentLayouts {
		for _, name := range layout.Functions {
			if _, duplicate := want[name]; duplicate {
				return fmt.Errorf("duplicate function %q", name)
			}
			want[name] = layout.Parameters
		}
	}
	if len(want) != len(item.Functions) {
		return fmt.Errorf("layouts cover %d of %d functions", len(want), len(item.Functions))
	}
	for _, name := range item.Functions {
		parameters, ok := want[name]
		if !ok {
			return fmt.Errorf("missing function %q", name)
		}
		actual, found := got[name]
		if !found {
			return fmt.Errorf("could not parse function %q", name)
		}
		if !slices.Equal(actual, parameters) {
			return fmt.Errorf("function %q parameters are %v, want %v", name, actual, parameters)
		}
	}
	return nil
}

func groupArgumentLayouts(functions []string, entries map[string][]string) []argumentLayout {
	result := make([]argumentLayout, 0)
	indexBySignature := map[string]int{}
	for _, name := range functions {
		parameters := entries[name]
		signature := strings.Join(parameters, "\x00")
		index, ok := indexBySignature[signature]
		if !ok {
			index = len(result)
			indexBySignature[signature] = index
			result = append(result, argumentLayout{Parameters: parameters})
		}
		result[index].Functions = append(result[index].Functions, name)
	}
	return result
}

func fileHash(root, name string) (string, error) {
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(name)))
	if err != nil {
		return "", err
	}
	data = bytes.ReplaceAll(data, []byte("\r\n"), []byte("\n"))
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func ptxEntries(data []byte) []string {
	var result []string
	for _, entry := range ptxEntryList(data) {
		result = append(result, entry.name)
	}
	slices.Sort(result)
	return result
}

func ptxEntryABIs(data []byte) map[string][]string {
	result := map[string][]string{}
	for _, entry := range ptxEntryList(data) {
		layout := []string{}
		for _, parameter := range entry.parameters {
			if declared := parameterLayout(parameter); declared != "" {
				layout = append(layout, declared)
			}
		}
		result[entry.name] = layout
	}
	return result
}
