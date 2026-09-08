package main

import (
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
)

// TestCurrentGuardControls readmits exact isolated repeats while retaining the
// historical guard as an independent comparison. It never measures a model.
func TestCurrentGuardControls(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT to check the six exact current control records; no models execute")
	}
	roots, err := dataroot.Resolve(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	surface, err := longform.Surface(t.Context(), filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	fixtures := []struct {
		name, historical string
		repeats          [3]string
	}{
		{
			name:       "Qwen capacity session",
			historical: "evidence:sha256:0a6076ca812d7b2d0eb8bc17eef3fda6913ecec8dfbfd0e366e14f42eba12e51",
			repeats: [3]string{
				"evidence:sha256:aa7b3a8942e7f37208060608df1a70a72d99d34a7be75f9c6942223c4ce5feec",
				"evidence:sha256:8e14c6b03cd8bab1b396bda8fbd1e9f52e7e26206789aa1adcb63d29eefbca4f",
				"evidence:sha256:52edee0ae6b2bdabea6c6b402c9ac83fe8e6be6338acfec54c0503d726f096b1",
			},
		},
		{
			name:       "E4B request session",
			historical: "evidence:sha256:4d51503d37ae7b10d80a3cb939777390473324d59a814383fcfdb48cc7af635a",
			repeats: [3]string{
				"evidence:sha256:07d45dfe0ac68309d4f312fd7b5b709ef303e96f075a2acf9fc9535368c1aecd",
				"evidence:sha256:f2ab8f7d98662be4524f2245ed0dab4298e1237b15103c5c4c5c86cf3fc48060",
				"evidence:sha256:06b89e81154d0b60dffc9190c919150609acae72e2044430eabea2690780882f",
			},
		},
	}
	requireStoreLineage(t, store, fixtures[0].historical)
	seen := make(map[artifact.ID]bool)
	opts := options{Root: filepath.Join("..", ".."), Repository: roots.Store, Corpus: "testdata/guard-corpus.txt", ValidateBaselines: true}
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			historical := measuredGuardRecord(t, store, fixture.historical)
			if err := validateGuard(historical.Result); err != nil {
				t.Fatal(err)
			}
			var first longform.Result
			for index, text := range fixture.repeats {
				id, err := artifact.ParseID(text)
				if err != nil || seen[id] || id == historical.Record {
					t.Fatalf("missing, duplicate or historical repeat %q: %v", text, err)
				}
				seen[id] = true
				accepted, err := longform.ReadBaseline(t.Context(), store, id)
				if err != nil {
					t.Fatal(err)
				}
				fresh := accepted.Result
				if fresh.Surface != surface || fresh.Commit != "1b7ac58150c36e83faaaf016cdd03d2a0aefd8ab" {
					t.Fatal("control does not identify the current inference surface and clean measured producer")
				}
				if err := validateGuard(fresh); err != nil {
					t.Fatal(err)
				}
				if verdict := longform.Compare(historical.Result, fresh, fresh.Floors, fresh.Floors.CheckRungCeiling); !verdict.Passed {
					t.Fatalf("historical comparison: %s", verdict)
				}
				if index == 0 {
					first = fresh
					opts.Models = append(opts.Models, fresh.ModelPath)
					opts.Baselines = append(opts.Baselines, text)
				} else if verdict := longform.Compare(first, fresh, fresh.Floors, fresh.Floors.CheckRungCeiling); !verdict.Passed {
					t.Fatalf("first-repeat comparison: %s", verdict)
				}
				t.Logf("record=%s judged_decode=%.1f wall=%.3fs; short shape and all %d rungs match tokens, quality, rate and allocation limits", id, fresh.Measure.DecodeTokensPerSecond, float64(fresh.WallNS)/1e9, len(fresh.Rungs))
			}
		})
	}
	if len(seen) != 6 {
		t.Fatalf("accepted control denominator = %d, want six distinct records", len(seen))
	}
	if t.Failed() {
		return
	}
	targets, err := listTargets(t.Context(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(targets) != len(fixtures) {
		t.Fatal("live control selection differs from the accepted denominator")
	}
	if err := bindBaselines(t.Context(), opts, targets); err != nil {
		t.Fatal(err)
	}
	if err := validateSelectedBaselines(io.Discard, targets, surface); err != nil {
		t.Fatal(err)
	}
	t.Log("control readmission: 2 exact models, 3 isolated repeats each; historical controls retained. Six other text cohorts, chat, modalities and full benchmark suites are excluded.")
}
