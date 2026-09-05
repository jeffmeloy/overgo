package main

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testevidence"
)

// TestAcceptedCapabilityCensus binds the clean producer's immutable before
// picture and checks it against the full live registered denominator. It does
// not execute models or turn historical verification tiers into current claims.
func TestAcceptedCapabilityCensus(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT to check the exact capability census; no models execute")
	}
	roots, err := dataroot.Resolve(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id, err := artifact.ParseID("evidence:sha256:e544d4878f06e2d4279fb1521b417aef5fb9335741273ff01d1ad964aeb3842a")
	if err != nil {
		t.Fatal(err)
	}
	value, err := capabilityCensusCodec.Require(t.Context(), store, id)
	if err != nil {
		t.Fatal(err)
	}
	if value.ProducerCommit != "5d55e6108519e7abf35776892b0043cad030027f" || len(value.Models) != 43 || len(value.Verifications) != 51 {
		t.Fatal("accepted producer, registered denominator or historical record set differs")
	}
	if err := checkCapabilityCensus(t.Context(), store, value, mediaCatalogLimit); err != nil {
		t.Fatal(err)
	}
	wantTasks := map[recipe.Task]int{
		recipe.TaskForecast: 1, recipe.TaskImageGen: 4, recipe.TaskInference: 9, recipe.TaskProjection: 1,
		recipe.TaskSeq2Seq: 1, recipe.TaskSpeech: 1, recipe.TaskTabular: 1, recipe.TaskTraining: 6,
		recipe.TaskVideoGen: 3, recipe.TaskVQA: 1,
	}
	tasks := map[recipe.Task]int{}
	activeModels, missing, stale, text, dna := 0, 0, 0, 0, 0
	for _, model := range value.Models {
		if len(model.Capabilities) != 0 {
			activeModels++
		}
		if !model.Present {
			missing++
			if len(model.Capabilities) != 0 {
				t.Fatalf("activated identity has no available recorded bytes: %s", model.Model)
			}
		}
		for _, activation := range model.Capabilities {
			tasks[activation.Task]++
			if activation.Stale != "" {
				stale++
			}
			if activation.Task != recipe.TaskInference {
				continue
			}
			domains, declared, err := evaluation.EvalDomains(t.Context(), store, model.Model)
			if err != nil {
				t.Fatal(err)
			}
			if !declared || slices.Contains(domains, evaluation.DomainText) {
				text++
			} else if slices.Equal(domains, []string{"dna"}) {
				dna++
			} else {
				t.Fatalf("unexpected inference domain for %s: %v", model.Model, domains)
			}
		}
	}
	if !maps.Equal(tasks, wantTasks) || activeModels != 26 || missing != 6 || stale != 0 || text != 8 || dna != 1 {
		t.Fatalf("census differs: tasks=%v active=%d unavailable=%d stale=%d text=%d DNA=%d", tasks, activeModels, missing, stale, text, dna)
	}
	t.Logf("census=%s: 43 identities, 26 activated identities, 28 task/recipe pairs in 10 task kinds, 6 unavailable inactive identities, 0 stale activations, 51 historical records; 8 text + 1 DNA inference activations. No model execution, quality promotion or historical evidence reuse was asserted.", id)
}
