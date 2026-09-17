package automationcheck

import (
	"slices"
	"strings"
)

// DeviceTestScope declares a lane's package and test selection. Named Tests
// are required individually; Run retains existing pattern-based contracts.
type DeviceTestScope struct {
	Package string
	Run     string
	Tests   []string
}

// Keep fixture-free kernel contracts separate from general host tests and
// externally supplied model journeys in the same package.
var deviceLaneScopes = []DeviceTestScope{
	{Package: "./internal/cuda/..."},
	{Package: "./internal/model"},
	{Package: "./internal/projector"},
	{Package: "./internal/optimizer"},
	{Package: "./internal/devicemath"},
	{Package: "./internal/densecausal", Run: "Device"},
	{Package: "./internal/inference", Tests: []string{
		"TestHermeticCUDAContinuousCacheParity",
		"TestHermeticCUDACapacityCachePageBoundary",
		"TestHermeticCUDAContinuationScoringParity",
		"TestDeviceDecodeSpan",
		"TestSpanDecodeSession",
		"TestDecodeSessionServesOnlyItsOwnPastPage",
		"TestRecurrentCheckpoint",
		"TestNextNMTPDeviceSpeculationLossless",
	}},
}

// DeviceLanePackages derives impact ownership from the lane's executed scopes.
// internal/cuda is recursive; all remaining packages are exact owners.
func DeviceLanePackages() []string {
	var packages []string
	for _, scope := range deviceLaneScopes {
		packages = append(packages, strings.TrimSuffix(strings.TrimPrefix(scope.Package, "./"), "/..."))
	}
	return packages
}

// DeviceScopes uses the same test contracts for full and scoped verification.
// Additional launch owners retain package-wide verification.
func DeviceScopes(plan DeviceVerificationPlan) []DeviceTestScope {
	if plan.Full {
		result := slices.Clone(deviceLaneScopes)
		for index := range result {
			result[index].Tests = slices.Clone(result[index].Tests)
		}
		return result
	}
	var result []DeviceTestScope
	for _, packagePath := range plan.Packages {
		scope := DeviceTestScope{Package: packagePath}
		for _, declared := range deviceLaneScopes {
			if declared.Package == packagePath {
				scope.Run, scope.Tests = declared.Run, slices.Clone(declared.Tests)
				break
			}
		}
		result = append(result, scope)
	}
	return result
}
