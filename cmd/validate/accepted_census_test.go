package main

import (
	"errors"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
)

// Preserve the historical census and admit the current denominator through
// an explicit disposition. Registration never grants new verification credit.
func TestAcceptedCapabilityCensus(t *testing.T) {
	t.Parallel()
	testskip.Short(t)
	testskip.StoreAcceptance(t)
	if os.Getenv(dataroot.Env) == "" {
		testskip.NotApplicable(t, "integration: set OVERGO_DATA_ROOT to check the exact capability census; no models execute")
	}
	roots, err := dataroot.Resolve(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	// The binding file names its own base, the historical census held below
	// by its producer and counts, so no identity is typed here or by whoever
	// accepts a new census into the file.
	var binding acceptedCensus
	if err := jsonfile.DecodeStrict("testdata/accepted_census.json", &binding); err != nil {
		t.Fatal(err)
	}
	id := binding.Base
	// The census binds the producer's store lineage: its document and the
	// checkpoint it names exist only there. A lane store that never carried
	// the document cannot check the census, and says so rather than failing.
	if found, err := store.HasContent(t.Context(), id); err != nil {
		t.Fatal(err)
	} else if !found {
		t.Skip("integration: the bound producer census document is absent from this store; only the producer's store lineage carries it")
	}
	value, err := capabilityCensusCodec.Require(t.Context(), store, id)
	if err != nil {
		t.Fatal(err)
	}
	if value.ProducerCommit != "5d55e6108519e7abf35776892b0043cad030027f" || len(value.Models) != 43 || len(value.Verifications) != 51 {
		t.Fatal("accepted producer, registered denominator or historical record set differs")
	}
	// The base counts its verifications up to the store commit it was taken
	// at. A store rebuilt since holds the base's document and none of those
	// commits: there the census accepted on the rebuilt store, held below,
	// answers for the checkpoint.
	if _, err := readCensusVerifications(t.Context(), store, value); err != nil && !errors.Is(err, errCensusCheckpointAbsent) {
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
			domains, declared, err := modelartifact.EvalDomains(t.Context(), store, model.Model)
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
	if err := checkCensusDisposition(t.Context(), store, binding); err != nil {
		t.Fatal(err)
	}
	current, err := capabilityCensusCodec.Require(t.Context(), store, binding.Current)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkCapabilityCensus(t.Context(), store, current, catalogLimit); err != nil {
		t.Fatal(err)
	}
	for _, invalid := range []acceptedCensus{
		{Base: id, Current: id, Finding: binding.Finding, Disposition: binding.Disposition},
		{Base: id, Current: binding.Current, Finding: binding.Finding, Disposition: binding.Finding},
		{Base: id, Current: binding.Current, Finding: id, Disposition: binding.Disposition},
	} {
		if err := checkCensusDisposition(t.Context(), store, invalid); err == nil {
			t.Fatal("unbound census, open finding or substituted authority admitted")
		}
	}
	t.Logf("current census=%s: %d registered identities, %d historical records; exact disposition=%s; current quality remains separate", current.ID, len(current.Models), len(current.Verifications), binding.Disposition)
}
