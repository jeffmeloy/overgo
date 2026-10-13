package trainingworkflow

import (
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/trainingprogram"
)

// TestOrderSeedSetsTheRecordsARunTrainsOn fine-tunes one tiny model under one
// text objective in runs that differ only in the seed their training records
// are read by, a few updates each, fewer than the training text holds
// windows, so the records a run reads first are the ones it trains on. Two
// runs of one seed train the same checkpoint. A run of another seed trains
// another checkpoint, and so does a run with no seed, which reads the records
// in membership order.
func TestOrderSeedSetsTheRecordsARunTrainsOn(t *testing.T) {
	t.Parallel()
	var documents []string
	for index, pattern := range []string{"abcd", "dcba", "acbd", "bdac", "cadb", "dbca", "abdc", "cdab", "badc", "dacb", "cbad", "adcb", "bcda", "cabd", "dbac", "acdb"} {
		// Distinct lengths keep every file its own record, and distinct
		// patterns make what a run learns depend on the records it reads.
		documents = append(documents, strings.Repeat(pattern, 6)[:8+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	// trained runs the objective into a directory of its own under the seed
	// and returns the checkpoint the run published.
	trained := func(directory string, order uint64) artifact.ID {
		t.Helper()
		result, err := RunTextHeldout(t.Context(), text.store, TextHeldoutRequest{
			Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, directory), Steps: 3, MaximumSequence: 8, Host: true, Order: order,
		})
		if err != nil {
			t.Fatalf("the run of order %d into %s: %v", order, directory, err)
		}
		return result.Trained
	}
	first, again, second, membership := trained("first", 1), trained("again", 1), trained("second", 2), trained("membership", 0)
	if again != first {
		t.Errorf("order 1 trained again published %s, want the checkpoint %s its first run published", again, first)
	}
	if second == first {
		t.Errorf("orders 1 and 2 both published %s, want a checkpoint for each seed", first)
	}
	if membership == first || membership == second {
		t.Errorf("membership order published %s, want a checkpoint no seeded run published (%s, %s)", membership, first, second)
	}
}
