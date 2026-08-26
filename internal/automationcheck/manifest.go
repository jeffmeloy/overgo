package automationcheck

import (
	"maps"
	"slices"
	"strings"

	"overgo/internal/codemanifest"
)

// ManifestSurface adapts a snapshot-bound manifest closure to the existing
// verification ownership contract. Build context remains part of manifest
// symbol identity but does not weaken package or symbol ownership matching.
func ManifestSurface(impact codemanifest.Impact) Surface {
	packages := make(map[string]bool, len(impact.Packages))
	for _, packagePath := range impact.Packages {
		packages[packagePath] = true
	}
	symbolSet := map[Symbol]bool{}
	for _, value := range impact.Reachable {
		packages[value.Package] = true
		symbolSet[Symbol{Package: value.Package, Receiver: value.Receiver, Name: value.Name}] = true
	}
	symbols := slices.Collect(maps.Keys(symbolSet))
	slices.SortFunc(symbols, func(left, right Symbol) int {
		return strings.Compare(left.Package+"\x00"+left.Receiver+"\x00"+left.Name, right.Package+"\x00"+right.Receiver+"\x00"+right.Name)
	})
	unknown := make([]string, 0, len(impact.Uncertainty))
	for _, item := range impact.Uncertainty {
		symbol := ""
		if item.Symbol != nil {
			symbol = item.Symbol.Package + "." + item.Symbol.Receiver + "." + item.Symbol.Name
		}
		unknown = append(unknown, strings.Join([]string{string(item.Kind), item.Context, item.Path, symbol, item.Reason}, ":"))
	}
	if impact.Base == "" || impact.Candidate == "" {
		unknown = append(unknown, "manifest impact identities are absent")
	}
	slices.Sort(unknown)
	return Surface{
		Identity: impact.Base + ":" + impact.Candidate,
		Packages: slices.Sorted(maps.Keys(packages)), Symbols: symbols, Unknown: unknown,
	}
}
