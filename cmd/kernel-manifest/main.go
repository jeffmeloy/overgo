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
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"

	cudaKernel "overgo/internal/cuda/kernel"
)

const manifestPath = "kernels/manifest.json"

var entryPattern = regexp.MustCompile(`(?m)^\.visible \.entry ([A-Za-z_][A-Za-z0-9_]*)\(`)
var entryABIPattern = regexp.MustCompile(`(?ms)^\.visible \.entry ([A-Za-z_][A-Za-z0-9_]*)\((.*?)^\)`)
var parameterPattern = regexp.MustCompile(`\.param\s+((?:\.align\s+\d+\s+)?\.[A-Za-z0-9_]+)\s+[A-Za-z_][A-Za-z0-9_]*(\[[^\]]+\])?`)

type manifest struct {
	Schema          int      `json:"schema"`
	ABIVersion      int      `json:"abiVersion"`
	CUDAToolkit     string   `json:"cudaToolkit"`
	NVCC            string   `json:"nvcc"`
	Target          string   `json:"target"`
	CompilerFlags   []string `json:"compilerFlags"`
	DefaultThreads  int      `json:"defaultThreads"`
	SharedMemoryABI string   `json:"sharedMemoryABI"`
	Modules         []module `json:"modules"`
}

type module struct {
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
	path := filepath.Join(root, filepath.FromSlash(manifestPath))
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var document manifest
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&document); err != nil {
		return fmt.Errorf("kernel manifest: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("kernel manifest: multiple JSON values")
		}
		return fmt.Errorf("kernel manifest: trailing data: %w", err)
	}
	if document.Schema != 2 || document.ABIVersion < 1 {
		return errors.New("kernel manifest: unsupported schema or ABI version")
	}
	if document.Target != "compute_89" || document.DefaultThreads <= 0 || document.SharedMemoryABI == "" {
		return errors.New("kernel manifest: target or launch ABI is incomplete")
	}
	if document.ABIVersion != cudaKernel.BundleABIVersion || document.Target != cudaKernel.BundleTarget {
		return errors.New("kernel manifest: Go host kernel ABI identity is stale")
	}
	runtimePins := map[string]string{
		"internal/cuda/kernel/vector_add.ptx": cudaKernel.VectorAddSHA256,
		"internal/cuda/kernel/ops_f32.ptx":    cudaKernel.OpsF32SHA256,
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
		sum := sha256.Sum256(assetData)
		assetHash := hex.EncodeToString(sum[:])
		if update {
			item.SourceSHA256 = sourceHash
			item.AssetSHA256 = assetHash
		} else if item.SourceSHA256 != sourceHash || item.AssetSHA256 != assetHash {
			return fmt.Errorf("kernel manifest: hashes for %q are stale", item.Asset)
		}
		for dependencyIndex := range item.SourceDependencies {
			dependency := &item.SourceDependencies[dependencyIndex]
			if dependency.Path == "" {
				return fmt.Errorf("kernel manifest: %q has an empty source dependency", item.Source)
			}
			if _, duplicate := seenSource[dependency.Path]; duplicate {
				return fmt.Errorf("kernel manifest: duplicate source %q", dependency.Path)
			}
			seenSource[dependency.Path] = struct{}{}
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
		if pinned, ok := runtimePins[item.Asset]; ok && pinned != assetHash {
			return fmt.Errorf("kernel manifest: Go host hash for %q is stale", item.Asset)
		}
		gotFunctions := ptxEntries(assetData)
		wantFunctions := append([]string(nil), item.Functions...)
		sort.Strings(wantFunctions)
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
	return os.WriteFile(path, append(output, '\n'), 0o644)
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
	matches := entryPattern.FindAllSubmatch(data, -1)
	result := make([]string, len(matches))
	for index, match := range matches {
		result[index] = string(match[1])
	}
	sort.Strings(result)
	return result
}

func ptxEntryABIs(data []byte) map[string][]string {
	result := map[string][]string{}
	for _, entry := range entryABIPattern.FindAllSubmatch(data, -1) {
		parameters := parameterPattern.FindAllSubmatch(entry[2], -1)
		layout := make([]string, len(parameters))
		for index, parameter := range parameters {
			layout[index] = strings.Join(strings.Fields(string(parameter[1])), " ") + string(parameter[2])
		}
		result[string(entry[1])] = layout
	}
	return result
}
