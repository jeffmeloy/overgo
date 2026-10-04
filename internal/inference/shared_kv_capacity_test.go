package inference

import (
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
)

// sharedKVFixtureContext holds the fixture's page bands 4, 8, 16, 32 and 64.
const sharedKVFixtureContext = uint32(64)

// TestSharedKVFixtureResolvesCapacitySession pins on the host that a Gemma 4
// model whose last layers share the KV of earlier ones compiles under the
// capacity decode session, as a model's recipe declares it when intake derives
// the session from the plan.
func TestSharedKVFixtureResolvesCapacitySession(t *testing.T) {
	path := testutil.HermeticGemma4SharedKVGGUF(t, sharedKVFixtureContext)
	for _, session := range []modelrecipe.DecodeSessionPolicy{
		modelrecipe.DecodeSessionRequest, modelrecipe.DecodeSessionCapacity,
	} {
		loaded, err := servingtest.ResolveActiveGGUFWithPolicy(
			path, recipe.PlacementHybrid, session, recipe.ResidencyDeviceF32,
		)
		if err != nil {
			t.Fatalf("%s session: %v", session, err)
		}
		file, _, _, _, plan, _, _, err := loaded.Take()
		if err != nil {
			t.Fatal(err)
		}
		if closeErr := file.Close(); closeErr != nil {
			t.Fatal(closeErr)
		}
		if plan.Decode.Session != session || !plan.Model.SupportsCapacityCache() {
			t.Fatalf("%s session compiled as %s, capacity support %t",
				session, plan.Decode.Session, plan.Model.SupportsCapacityCache())
		}
	}
}
