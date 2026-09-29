package runrecord

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// gateDoor stands in for the old gate in this package's fixtures: the store
// admits lifecycle records from its capability alone.
var gateDoor = overgodb.NewProducer("gate")

// TestRetiredGateAliasesAtRest admits the old gate's two store-local heads
// across a store rewrite and refuses any other store-local alias.
func TestRetiredGateAliasesAtRest(t *testing.T) {
	t.Parallel()
	for _, name := range []string{gateLifecycleCurrentAlias, overgodb.StoreLocalAliasPrefix + "gate-lanes/current"} {
		if err := RetiredGateAliasesAtRest(name, artifact.ID{}); err != nil {
			t.Fatalf("%s refused: %v", name, err)
		}
	}
	if err := RetiredGateAliasesAtRest(overgodb.StoreLocalAliasPrefix+"unknown/current", artifact.ID{}); err == nil ||
		!strings.Contains(err.Error(), "unknown store-local authority") {
		t.Fatalf("unknown authority error = %v", err)
	}
}
