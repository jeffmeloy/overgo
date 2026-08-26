package automationcheck

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
)

// ReviewedOwnership is a separate human-reviewed exception layer for
// ownership that structural analysis cannot infer.
type ReviewedOwnership struct {
	Check           string   `json:"check"`
	Reason          string   `json:"reason"`
	Packages        []string `json:"packages,omitempty"`
	PackagePrefixes []string `json:"package_prefixes,omitempty"`
	Symbols         []Symbol `json:"symbols,omitempty"`
}

// OwnershipCoverage reports manifest surfaces that no selectable check owns.
type OwnershipCoverage struct {
	Checks            int      `json:"checks"`
	OwnedChecks       int      `json:"owned_checks"`
	UncoveredPackages []string `json:"uncovered_packages,omitempty"`
	UncoveredSymbols  []Symbol `json:"uncovered_symbols,omitempty"`
	Unknown           []string `json:"unknown,omitempty"`
}

// CompleteOwnership applies reviewed overlays, validates a non-contradictory
// ownership catalog, and converts every uncovered surface into uncertainty.
func CompleteOwnership(checks []Check, surface Surface, overlays []ReviewedOwnership) ([]Check, Surface, OwnershipCoverage, error) {
	effective := slices.Clone(checks)
	index := make(map[string]int, len(effective))
	facts := map[Fact]string{}
	coverage := OwnershipCoverage{Checks: len(effective)}
	for position, check := range effective {
		if err := validate(check); err != nil {
			return nil, Surface{}, OwnershipCoverage{}, err
		}
		name := check.Descriptor.Name
		if _, exists := index[name]; exists {
			return nil, Surface{}, OwnershipCoverage{}, fmt.Errorf("automation ownership: duplicate check %q", name)
		}
		index[name] = position
		ownership := check.Descriptor.Ownership
		if ownership.Fact == "" {
			continue
		}
		coverage.OwnedChecks++
		if prior := facts[ownership.Fact]; prior != "" && prior != name {
			return nil, Surface{}, OwnershipCoverage{}, fmt.Errorf("automation ownership: fact %q belongs to both %s and %s", ownership.Fact, prior, name)
		}
		facts[ownership.Fact] = name
		if err := validateOwnershipSurface(ownership); err != nil {
			return nil, Surface{}, OwnershipCoverage{}, fmt.Errorf("automation ownership %s: %w", name, err)
		}
	}
	for _, overlay := range overlays {
		position, found := index[overlay.Check]
		if !found || strings.TrimSpace(overlay.Reason) == "" || strings.TrimSpace(overlay.Reason) != overlay.Reason {
			return nil, Surface{}, OwnershipCoverage{}, errors.New("automation ownership: invalid reviewed overlay")
		}
		ownership := &effective[position].Descriptor.Ownership
		if ownership.Fact == "" || len(overlay.Packages)+len(overlay.PackagePrefixes)+len(overlay.Symbols) == 0 {
			return nil, Surface{}, OwnershipCoverage{}, errors.New("automation ownership: reviewed overlay has no owned check or surface")
		}
		ownership.Packages = append(slices.Clone(ownership.Packages), overlay.Packages...)
		ownership.PackagePrefixes = append(slices.Clone(ownership.PackagePrefixes), overlay.PackagePrefixes...)
		ownership.Symbols = append(slices.Clone(ownership.Symbols), overlay.Symbols...)
		normalizeOwnership(ownership)
		if err := validateOwnershipSurface(*ownership); err != nil {
			return nil, Surface{}, OwnershipCoverage{}, fmt.Errorf("automation ownership overlay %s: %w", overlay.Check, err)
		}
	}
	ownedPackages := map[string]bool{}
	for _, packagePath := range surface.Packages {
		if !packageOwned(effective, packagePath) {
			ownedPackages[packagePath] = true
		}
	}
	coverage.UncoveredPackages = slices.Sorted(maps.Keys(ownedPackages))
	for _, symbol := range surface.Symbols {
		if !symbolOwned(effective, symbol) {
			coverage.UncoveredSymbols = append(coverage.UncoveredSymbols, symbol)
		}
	}
	slices.SortFunc(coverage.UncoveredSymbols, compareOwnershipSymbol)
	for _, packagePath := range coverage.UncoveredPackages {
		coverage.Unknown = append(coverage.Unknown, "uncovered-package:"+packagePath)
	}
	for _, symbol := range coverage.UncoveredSymbols {
		coverage.Unknown = append(coverage.Unknown, "uncovered-symbol:"+symbol.Package+":"+symbol.Receiver+":"+symbol.Name)
	}
	coverage.Unknown = append(coverage.Unknown, surface.Unknown...)
	slices.Sort(coverage.Unknown)
	coverage.Unknown = slices.Compact(coverage.Unknown)
	surface.Unknown = slices.Clone(coverage.Unknown)
	return effective, surface, coverage, nil
}

func validateOwnershipSurface(ownership Ownership) error {
	for _, value := range append(slices.Clone(ownership.Packages), ownership.PackagePrefixes...) {
		if strings.TrimSpace(value) == "" || strings.TrimSpace(value) != value || strings.ContainsAny(value, "\x00\r\n\\") {
			return errors.New("invalid package surface")
		}
	}
	for _, symbol := range ownership.Symbols {
		if strings.TrimSpace(symbol.Package) == "" || strings.TrimSpace(symbol.Name) == "" || strings.ContainsAny(symbol.Package+symbol.Receiver+symbol.Name, "\x00\r\n") {
			return errors.New("invalid symbol surface")
		}
	}
	return nil
}

func normalizeOwnership(ownership *Ownership) {
	slices.Sort(ownership.Packages)
	ownership.Packages = slices.Compact(ownership.Packages)
	slices.Sort(ownership.PackagePrefixes)
	ownership.PackagePrefixes = slices.Compact(ownership.PackagePrefixes)
	slices.SortFunc(ownership.Symbols, compareOwnershipSymbol)
	ownership.Symbols = slices.Compact(ownership.Symbols)
}

func packageOwned(checks []Check, packagePath string) bool {
	for _, check := range checks {
		ownership := check.Descriptor.Ownership
		if slices.Contains(ownership.Packages, packagePath) || slices.ContainsFunc(ownership.PackagePrefixes, func(prefix string) bool {
			return packagePath == prefix || strings.HasPrefix(packagePath, prefix+"/")
		}) {
			return true
		}
	}
	return false
}

func symbolOwned(checks []Check, symbol Symbol) bool {
	for _, check := range checks {
		ownership := check.Descriptor.Ownership
		if slices.Contains(ownership.Symbols, symbol) || slices.Contains(ownership.Packages, symbol.Package) || slices.ContainsFunc(ownership.PackagePrefixes, func(prefix string) bool {
			return symbol.Package == prefix || strings.HasPrefix(symbol.Package, prefix+"/")
		}) {
			return true
		}
	}
	return false
}

func compareOwnershipSymbol(left, right Symbol) int {
	return strings.Compare(left.Package+"\x00"+left.Receiver+"\x00"+left.Name, right.Package+"\x00"+right.Receiver+"\x00"+right.Name)
}
