package automationcheck

import (
	"slices"
	"strings"
)

// Symbol is one package function or method owned by a check.
type Symbol struct {
	Package  string `json:"package"`
	Receiver string `json:"receiver,omitempty"`
	Name     string `json:"name"`
}

// Surface is the snapshot-bound change closure presented to check ownership.
// Any Unknown entry prevents this producer from authorizing exclusions.
type Surface struct {
	Identity string   `json:"identity"`
	Packages []string `json:"packages,omitempty"`
	Symbols  []Symbol `json:"symbols,omitempty"`
	Unknown  []string `json:"unknown,omitempty"`
}

// OwnershipImpact intersects a structural change closure with each declared
// check surface. Unknown analysis returns no verdict and therefore runs checks.
func OwnershipImpact(checks []Check, surface Surface) Impact {
	if strings.TrimSpace(surface.Identity) == "" || len(surface.Unknown) != 0 {
		return Impact{}
	}
	packages := make(map[string]bool, len(surface.Packages))
	for _, packagePath := range surface.Packages {
		packages[packagePath] = true
	}
	symbols := make(map[Symbol]bool, len(surface.Symbols))
	for _, symbol := range surface.Symbols {
		symbols[symbol] = true
	}
	var impact Impact
	for _, check := range checks {
		ownership := check.Descriptor.Ownership
		if ownership.Fact == "" {
			continue
		}
		if ownershipIntersects(ownership, packages, symbols) {
			impact.Facts = append(impact.Facts, ownership.Fact)
			continue
		}
		impact.Exclusions = append(impact.Exclusions, Exclusion{
			Check:  check.Descriptor.Name,
			Reason: "snapshot " + surface.Identity + " symbol closure is disjoint from declared ownership",
		})
	}
	slices.Sort(impact.Facts)
	impact.Facts = slices.Compact(impact.Facts)
	slices.SortFunc(impact.Exclusions, func(left, right Exclusion) int {
		return strings.Compare(left.Check, right.Check)
	})
	return impact
}

func ownershipIntersects(ownership Ownership, packages map[string]bool, symbols map[Symbol]bool) bool {
	for _, packagePath := range ownership.Packages {
		if packages[packagePath] {
			return true
		}
	}
	for _, prefix := range ownership.PackagePrefixes {
		for packagePath := range packages {
			if packagePath == prefix || strings.HasPrefix(packagePath, prefix+"/") {
				return true
			}
		}
	}
	for _, symbol := range ownership.Symbols {
		if symbols[symbol] {
			return true
		}
	}
	return false
}
