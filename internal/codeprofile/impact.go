package codeprofile

import (
	"bytes"
	"crypto/sha256"
	"go/ast"
	"go/printer"
	"go/token"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/repoanalysis"
)

// FunctionSymbol is one production function or method in an analyzed source
// snapshot. Package is the build-selected import path; Receiver is empty for a
// package function.
type FunctionSymbol struct {
	Package  string `json:"package"`
	File     string `json:"file"`
	Receiver string `json:"receiver,omitzero"`
	Name     string `json:"name"`
}

// ImpactBoundary is one condition the syntactic graph cannot prove independent.
// Kind is mechanically derived; Path and Symbol bind it to exact source.
type ImpactBoundary struct {
	Kind   string `json:"kind"`
	Path   string `json:"path"`
	Symbol string `json:"symbol,omitzero"`
}

// FunctionImpact binds exact changed-body seeds to every production caller
// that can observe them through the syntactic reference index.
type FunctionImpact struct {
	BaseIdentity      string           `json:"base_identity"`
	CandidateIdentity string           `json:"candidate_identity"`
	Seeds             []FunctionSymbol `json:"seeds,omitempty"`
	Reachable         []FunctionSymbol `json:"reachable,omitempty"`
	Packages          []string         `json:"packages,omitempty"`
	Unknown           []ImpactBoundary `json:"unknown,omitempty"`
}

// DeriveFunctionImpact diffs exact function bodies and closes transitively over
// reverse references from both snapshots. The union preserves callers of
// removed functions as well as callers introduced in the candidate.
func DeriveFunctionImpact(
	base, candidate repoanalysis.SourceSnapshot,
	baseSelection, candidateSelection repoanalysis.BuildSelection,
	changedPaths []string,
) (FunctionImpact, error) {
	baseProfile, err := Build(base)
	if err != nil {
		return FunctionImpact{}, err
	}
	candidateProfile, err := Build(candidate)
	if err != nil {
		return FunctionImpact{}, err
	}
	baseIndex, err := productionConsumerIndex(base, baseSelection, nil)
	if err != nil {
		return FunctionImpact{}, err
	}
	candidateIndex, err := productionConsumerIndex(candidate, candidateSelection, nil)
	if err != nil {
		return FunctionImpact{}, err
	}
	baseFunctions := profileFunctionFingerprints(baseProfile)
	candidateFunctions := profileFunctionFingerprints(candidateProfile)
	seedKeys := map[string]bool{}
	for key, fingerprint := range baseFunctions {
		if candidateFingerprint, found := candidateFunctions[key]; !found || candidateFingerprint != fingerprint {
			seedKeys[key] = true
		}
	}
	for key, fingerprint := range candidateFunctions {
		if baseFingerprint, found := baseFunctions[key]; !found || baseFingerprint != fingerprint {
			seedKeys[key] = true
		}
	}
	baseSymbols, baseIndices := indexedFunctions(baseIndex)
	candidateSymbols, candidateIndices := indexedFunctions(candidateIndex)
	reachableKeys := map[string]bool{}
	closeReverseImpact(baseIndex, baseIndices, seedKeys, reachableKeys)
	closeReverseImpact(candidateIndex, candidateIndices, seedKeys, reachableKeys)
	result := FunctionImpact{BaseIdentity: base.Identity(), CandidateIdentity: candidate.Identity()}
	result.Seeds = symbolsForKeys(seedKeys, candidateSymbols, baseSymbols)
	result.Reachable = symbolsForKeys(reachableKeys, candidateSymbols, baseSymbols)
	result.Packages = changedPackageDirectories(base, candidate, changedPaths)
	result.Unknown, err = impactBoundaries(
		base, candidate, baseSelection, candidateSelection, changedPaths,
		baseIndex, candidateIndex, baseIndices, candidateIndices, reachableKeys,
	)
	if err != nil {
		return FunctionImpact{}, err
	}
	return result, nil
}

func changedPackageDirectories(base, candidate repoanalysis.SourceSnapshot, changedPaths []string) []string {
	baseFiles, candidateFiles := snapshotFiles(base), snapshotFiles(candidate)
	packages := map[string]bool{}
	for _, changed := range changedPaths {
		changed = filepath.ToSlash(filepath.Clean(filepath.FromSlash(changed)))
		if !strings.HasSuffix(changed, ".go") {
			continue
		}
		if _, found := baseFiles[changed]; !found {
			if _, found = candidateFiles[changed]; !found {
				continue
			}
		}
		packages[path.Dir(changed)] = true
	}
	result := make([]string, 0, len(packages))
	for packagePath := range packages {
		result = append(result, packagePath)
	}
	slices.Sort(result)
	return result
}

func impactBoundaries(
	base, candidate repoanalysis.SourceSnapshot,
	baseSelection, candidateSelection repoanalysis.BuildSelection,
	changedPaths []string,
	baseIndex, candidateIndex consumerIndex,
	baseIndices, candidateIndices map[string]int,
	reachable map[string]bool,
) ([]ImpactBoundary, error) {
	boundaries := map[string]ImpactBoundary{}
	baseFiles, candidateFiles := snapshotFiles(base), snapshotFiles(candidate)
	if len(changedPaths) == 0 {
		addImpactBoundary(boundaries, ImpactBoundary{Kind: "scope"})
	}
	for _, changed := range changedPaths {
		changed = filepath.ToSlash(filepath.Clean(filepath.FromSlash(changed)))
		if !strings.HasSuffix(changed, ".go") {
			addImpactBoundary(boundaries, ImpactBoundary{Kind: "non-go", Path: changed})
			continue
		}
		baseFile, inBase := baseFiles[changed]
		candidateFile, inCandidate := candidateFiles[changed]
		if !inBase && !inCandidate {
			addImpactBoundary(boundaries, ImpactBoundary{Kind: "outside-snapshot", Path: changed})
			continue
		}
		for _, source := range []repoanalysis.GoFile{baseFile, candidateFile} {
			if source.Path == "" {
				continue
			}
			generated, err := source.Generated()
			if err != nil {
				return nil, err
			}
			if generated {
				addImpactBoundary(boundaries, ImpactBoundary{Kind: "generated", Path: changed})
			}
		}
		if buildSelectionUnknown(changed, baseSelection) || buildSelectionUnknown(changed, candidateSelection) {
			addImpactBoundary(boundaries, ImpactBoundary{Kind: "build-selection", Path: changed})
		}
		if inBase && inCandidate {
			baseExpression, err := baseFile.BuildExpression()
			if err != nil {
				return nil, err
			}
			candidateExpression, err := candidateFile.BuildExpression()
			if err != nil {
				return nil, err
			}
			if baseExpression != candidateExpression {
				addImpactBoundary(boundaries, ImpactBoundary{Kind: "build-expression", Path: changed})
			}
			baseDeclarations, err := nonFunctionFingerprint(baseFile)
			if err != nil {
				return nil, err
			}
			candidateDeclarations, err := nonFunctionFingerprint(candidateFile)
			if err != nil {
				return nil, err
			}
			if baseDeclarations != candidateDeclarations {
				addImpactBoundary(boundaries, ImpactBoundary{Kind: "declaration", Path: changed})
			}
		} else {
			source := baseFile
			if inCandidate {
				source = candidateFile
			}
			unmodeled, err := hasNonFunctionDeclarations(source)
			if err != nil {
				return nil, err
			}
			if unmodeled {
				addImpactBoundary(boundaries, ImpactBoundary{Kind: "declaration", Path: changed})
			}
		}
	}
	appendReachableBoundaries(boundaries, baseIndex, baseIndices, reachable)
	appendReachableBoundaries(boundaries, candidateIndex, candidateIndices, reachable)
	result := make([]ImpactBoundary, 0, len(boundaries))
	for _, boundary := range boundaries {
		result = append(result, boundary)
	}
	slices.SortFunc(result, func(left, right ImpactBoundary) int {
		if order := strings.Compare(left.Kind, right.Kind); order != 0 {
			return order
		}
		if order := strings.Compare(left.Path, right.Path); order != 0 {
			return order
		}
		return strings.Compare(left.Symbol, right.Symbol)
	})
	return result, nil
}

func hasNonFunctionDeclarations(source repoanalysis.GoFile) (bool, error) {
	file, err := source.Syntax()
	if err != nil {
		return false, err
	}
	for _, declaration := range file.Decls {
		if _, function := declaration.(*ast.FuncDecl); !function {
			return true, nil
		}
	}
	return false, nil
}

func snapshotFiles(snapshot repoanalysis.SourceSnapshot) map[string]repoanalysis.GoFile {
	files := make(map[string]repoanalysis.GoFile, len(snapshot.Files))
	for _, file := range snapshot.Files {
		files[file.Path] = file
	}
	return files
}

func buildSelectionUnknown(sourcePath string, selection repoanalysis.BuildSelection) bool {
	selected, known := selection.Files[sourcePath]
	return !known || !selected
}

func nonFunctionFingerprint(source repoanalysis.GoFile) (string, error) {
	file, err := source.Syntax()
	if err != nil {
		return "", err
	}
	var rendered bytes.Buffer
	if err := printer.Fprint(&rendered, token.NewFileSet(), file.Name); err != nil {
		return "", err
	}
	for _, declaration := range file.Decls {
		if _, function := declaration.(*ast.FuncDecl); function {
			continue
		}
		if err := printer.Fprint(&rendered, token.NewFileSet(), declaration); err != nil {
			return "", err
		}
	}
	digest := sha256.Sum256(rendered.Bytes())
	return string(digest[:]), nil
}

func appendReachableBoundaries(boundaries map[string]ImpactBoundary, index consumerIndex, indices map[string]int, reachable map[string]bool) {
	for key, declarationIndex := range indices {
		if !reachable[key] {
			continue
		}
		declaration := index.declarations[declarationIndex]
		switch declaration.Boundary {
		case "reflection", "cgo", "build-variant", "generated", "external":
			symbol := declaration.Name
			if declaration.Receiver != "" {
				symbol = declaration.Receiver + "." + declaration.Name
			}
			addImpactBoundary(boundaries, ImpactBoundary{
				Kind: declaration.Boundary, Path: declaration.File,
				Symbol: symbol,
			})
		}
	}
}

func addImpactBoundary(boundaries map[string]ImpactBoundary, boundary ImpactBoundary) {
	key := boundary.Kind + "\x00" + boundary.Path + "\x00" + boundary.Symbol
	boundaries[key] = boundary
}

func profileFunctionFingerprints(profile Profile) map[string]string {
	result := make(map[string]string, len(profile.Functions))
	for _, function := range profile.Functions {
		if function.fingerprint == "" || function.AdvisoryClass == "test" {
			continue
		}
		result[functionSymbolKey(function.packagePath, function.receiver, function.Name)] = function.fingerprint
	}
	return result
}

func indexedFunctions(index consumerIndex) (map[string]FunctionSymbol, map[string]int) {
	symbols := map[string]FunctionSymbol{}
	indices := map[string]int{}
	for declarationIndex, declaration := range index.declarations {
		if declaration.Kind != "function" && declaration.Kind != "method" {
			continue
		}
		key := functionSymbolKey(path.Dir(declaration.File), declaration.Receiver, declaration.Name)
		symbols[key] = FunctionSymbol{
			Package: declaration.Package, File: declaration.File,
			Receiver: declaration.Receiver, Name: declaration.Name,
		}
		indices[key] = declarationIndex
	}
	return symbols, indices
}

func closeReverseImpact(index consumerIndex, indices map[string]int, seeds, reachable map[string]bool) {
	keys := make(map[int]string, len(indices))
	queue := make([]int, 0, len(seeds))
	visited := map[int]bool{}
	for key, declarationIndex := range indices {
		keys[declarationIndex] = key
		if seeds[key] {
			queue = append(queue, declarationIndex)
			visited[declarationIndex] = true
			reachable[key] = true
		}
	}
	for cursor := 0; cursor < len(queue); cursor++ {
		current := queue[cursor]
		for caller := range index.reverse[current] {
			key := keys[caller]
			if key == "" || visited[caller] {
				continue
			}
			visited[caller] = true
			reachable[key] = true
			queue = append(queue, caller)
		}
	}
}

func symbolsForKeys(keys map[string]bool, preferred, fallback map[string]FunctionSymbol) []FunctionSymbol {
	result := make([]FunctionSymbol, 0, len(keys))
	for key := range keys {
		symbol, found := preferred[key]
		if !found {
			symbol, found = fallback[key]
		}
		if found {
			result = append(result, symbol)
		}
	}
	slices.SortFunc(result, compareFunctionSymbols)
	return result
}

func compareFunctionSymbols(left, right FunctionSymbol) int {
	if order := strings.Compare(left.Package, right.Package); order != 0 {
		return order
	}
	if order := strings.Compare(left.Receiver, right.Receiver); order != 0 {
		return order
	}
	if order := strings.Compare(left.Name, right.Name); order != 0 {
		return order
	}
	return strings.Compare(left.File, right.File)
}

func functionSymbolKey(packagePath, receiver, name string) string {
	return packagePath + "\x00" + receiver + "\x00" + name
}
