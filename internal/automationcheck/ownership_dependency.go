package automationcheck

import (
	"slices"
	"strings"
)

// DeviceLanePackages are the repository packages whose tests the device
// lane runs in its full plan; internal/cuda is a prefix, the rest exact.
var DeviceLanePackages = []string{
	"internal/cuda", "internal/model", "internal/projector", "internal/optimizer", "internal/devicemath", "internal/densecausal",
}

// DependencyResolver reports whether any package the ownership names, or any
// package under one of its prefixes, is or transitively depends on the
// changed package.
type DependencyResolver func(ownership Ownership, changed string) bool

// OwnershipByDependency decides package-owned checks by the linker's rule when
// symbol reachability is uncertain: a check's tests can observe a change
// only if a package it owns is, or depends on, a changed package. Checks
// owning symbols stay unresolved and run.
func OwnershipByDependency(checks []Check, changed []string, dependsOn DependencyResolver) Impact {
	var impact Impact
	if dependsOn == nil {
		return impact
	}
	for _, check := range checks {
		ownership := check.Descriptor.Ownership
		if ownership.Fact == "" || len(ownership.Symbols) != 0 {
			continue
		}
		affected := slices.ContainsFunc(changed, func(packagePath string) bool { return dependsOn(ownership, packagePath) })
		if affected {
			impact.Facts = append(impact.Facts, ownership.Fact)
			continue
		}
		impact.Exclusions = append(impact.Exclusions, Exclusion{
			Check:  check.Descriptor.Name,
			Reason: "dependency closure of the owned packages excludes every changed package: " + strings.Join(changed, ","),
		})
	}
	slices.Sort(impact.Facts)
	impact.Facts = slices.Compact(impact.Facts)
	slices.SortFunc(impact.Exclusions, func(left, right Exclusion) int { return strings.Compare(left.Check, right.Check) })
	return impact
}

// OwnedPackage reports whether packagePath is one the ownership names
// exactly or under one of its prefixes.
func (ownership Ownership) OwnedPackage(packagePath string) bool {
	if slices.Contains(ownership.Packages, packagePath) {
		return true
	}
	return slices.ContainsFunc(ownership.PackagePrefixes, func(prefix string) bool {
		return packagePath == prefix || strings.HasPrefix(packagePath, prefix+"/")
	})
}
