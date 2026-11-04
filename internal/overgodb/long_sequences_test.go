//go:build integration

package overgodb

import "testing"

// TestLongStoreLifecycles drives the store through crashLongSeeds generated lifecycles, held to the promises
// TestGeneratedStoreLifecycles holds its crashSeeds to. It takes minutes, so the cycle's checks run it, under the
// cycle's tags (landing.CycleLanes), where a full run already spends hours, and a commit's checks never do.
func TestLongStoreLifecycles(t *testing.T) {
	// Serial: it runs the package's tests again in a go test of its own, which compiles the package under an overlay.
	runCrashHarness(t, "TestLongLifecyclesKeepTheStoresPromises")
}
