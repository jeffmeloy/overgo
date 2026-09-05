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
			historical: "evidence:sha256:0190b70b93219859525aff79e925f30ccc8b6be5647cb36eba2838ca56bef2c8",
			repeats: [3]string{
				"evidence:sha256:0a6076ca812d7b2d0eb8bc17eef3fda6913ecec8dfbfd0e366e14f42eba12e51",
				"evidence:sha256:c672001f407f6025292e19885227dc51c9e0ff9228a6e839b2dfa205ddb1a7c7",
				"evidence:sha256:cd5fe11911f9f07daf8afd32cab656551b0534be5ee1e726f77c8638a39db4d3",
			},
		},
		{
			name:       "E4B request session",
			historical: "evidence:sha256:3d6b27ffaf0fa747799b64f85eda10e5261cf09465c0457652e0a2d4f422e928",
			repeats: [3]string{
				"evidence:sha256:4d51503d37ae7b10d80a3cb939777390473324d59a814383fcfdb48cc7af635a",
				"evidence:sha256:8f3fbbfa4ea87c39d9b0c246a16053aa39b6cc9f40faae8fdad2e840980ac6ad",
				"evidence:sha256:c62797260111a90da085b14c77f72303742751ca3f2fdaffac1149ee93baa22b",
			},
		},
	}
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
				if fresh.Surface != surface || fresh.Commit != "2b8737ac4f2bc42533f7c19e19bbf94c36b36aff" {
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
