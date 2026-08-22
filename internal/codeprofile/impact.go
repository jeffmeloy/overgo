package codeprofile

import (
	"path"
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
	Receiver string `json:"receiver,omitempty"`
	Name     string `json:"name"`
}

// FunctionImpact binds exact changed-body seeds to every production caller
// that can observe them through the syntactic reference index.
type FunctionImpact struct {
	BaseIdentity      string           `json:"base_identity"`
	CandidateIdentity string           `json:"candidate_identity"`
	Seeds             []FunctionSymbol `json:"seeds,omitempty"`
	Reachable         []FunctionSymbol `json:"reachable,omitempty"`
}

// DeriveFunctionImpact diffs exact function bodies and closes transitively over
// reverse references from both snapshots. The union preserves callers of
// removed functions as well as callers introduced in the candidate.
func DeriveFunctionImpact(
	base, candidate repoanalysis.SourceSnapshot,
	baseSelection, candidateSelection repoanalysis.BuildSelection,
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
	return result, nil
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
