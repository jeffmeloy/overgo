package automationcheck

import (
	"slices"
	"testing"
)

// fixtureDependencies: import edges of a small repository; a resolver walks
// them transitively from every owned package.
var fixtureDependencies = map[string][]string{
	"internal/server":      {"internal/plan", "internal/runrecord"},
	"internal/model":       {"internal/cuda/kernel", "internal/tensor"},
	"internal/cuda/kernel": {"internal/tensor"},
	"internal/projector":   {"internal/model"},
	"internal/gate":        {"internal/loop", "internal/plan"},
	"cmd/loop":             {"internal/loop"},
	"internal/loop":        {"internal/plan"},
}

func fixtureResolver(t *testing.T) DependencyResolver {
	t.Helper()
	var closure func(packagePath string, seen map[string]bool) bool
	return func(ownership Ownership, changed string) bool {
		closure = func(packagePath string, seen map[string]bool) bool {
			if packagePath == changed {
				return true
			}
			if seen[packagePath] {
				return false
			}
			seen[packagePath] = true
			return slices.ContainsFunc(fixtureDependencies[packagePath], func(dependency string) bool { return closure(dependency, seen) })
		}
		for packagePath := range fixtureDependencies {
			if ownership.OwnedPackage(packagePath) && closure(packagePath, map[string]bool{}) {
				return true
			}
		}
		return slices.ContainsFunc(ownership.Packages, func(owned string) bool { return closure(owned, map[string]bool{}) })
	}
}

// TestUntouchedOwnershipExcludesDeviceChecks pins: under symbol uncertainty
// the dependency closure decides package-owned checks; a change confined to
// internal/loop excludes both the device and browser lanes, a change in
// internal/plan keeps the browser lane (the server depends on plan) and
// excludes the device lane, a change under internal/cuda keeps the device
// lane, and a symbol-owned check is never excluded by dependency alone.
func TestUntouchedOwnershipExcludesDeviceChecks(t *testing.T) {
	record := recordingLane(new([]string))
	device := DeviceCheck(".", nil, DeviceLanePackages[1:], record)
	webui := WebUICheck(".", record)
	symbolOwned := Check{
		Descriptor: Descriptor{
			Name: "symbol", Phase: "test", Triggers: []Fact{"owner:symbol"}, Inapplicable: "no symbol changed",
			Ownership: Ownership{Fact: "owner:symbol", Symbols: []Symbol{{Package: "internal/loop", Name: "Run"}}},
		},
		Run: pass,
	}
	checks := []Check{device, webui, symbolOwned}
	resolver := fixtureResolver(t)
	for _, tc := range []struct {
		name     string
		changed  []string
		excluded []string
		facts    []Fact
	}{
		{"loop only", []string{"internal/loop"}, []string{"device", WebUICheckName}, nil},
		{"plan reaches the server", []string{"internal/plan"}, []string{"device"}, []Fact{WebUIImpact}},
		{"cuda kernel", []string{"internal/cuda/kernel"}, []string{WebUICheckName}, []Fact{deviceImpact}},
		{"model", []string{"internal/model"}, []string{WebUICheckName}, []Fact{deviceImpact}},
	} {
		impact := OwnershipByDependency(checks, tc.changed, resolver)
		var excluded []string
		for _, exclusion := range impact.Exclusions {
			excluded = append(excluded, exclusion.Check)
		}
		slices.Sort(excluded)
		slices.Sort(tc.excluded)
		if !slices.Equal(excluded, tc.excluded) || !slices.Equal(impact.Facts, tc.facts) {
			t.Fatalf("%s: excluded=%v facts=%v, want %v / %v", tc.name, excluded, impact.Facts, tc.excluded, tc.facts)
		}
		if _, reason := impact.ExclusionReason("symbol"); reason {
			t.Fatalf("%s: a symbol-owned check was excluded by dependency alone", tc.name)
		}
		if _, err := Plan(checks, impact); err != nil {
			t.Fatalf("%s: plan: %v", tc.name, err)
		}
	}
	if impact := OwnershipByDependency(checks, []string{"internal/loop"}, nil); len(impact.Exclusions) != 0 {
		t.Fatal("a nil resolver excluded a check")
	}
}
