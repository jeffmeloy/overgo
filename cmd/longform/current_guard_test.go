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

type guardCohort struct {
	name, historical string
	repeats          [3]string
}

// TestCurrentGuardControls checks retained controls without measuring models.
func TestCurrentGuardControls(t *testing.T) {
	fixtures := []guardCohort{
		{
			name:       "Qwen capacity session",
			historical: "evidence:sha256:0a6076ca812d7b2d0eb8bc17eef3fda6913ecec8dfbfd0e366e14f42eba12e51",
			repeats: [3]string{
				"evidence:sha256:8216acc2fdbc4705d6acead094d3253c176aa9c029683bc4e4480d19774d410c",
				"evidence:sha256:98cb6f0b40bd3ece9d1cf1815b4a5255fd6e26fa550dc57806cd82f5f7d8634a",
				"evidence:sha256:8a3933043176ed794d5fcad18c647dfedf583fb59a8a1bbe1e6aec0dead657d7",
			},
		},
		{
			name:       "E4B request session",
			historical: "evidence:sha256:4d51503d37ae7b10d80a3cb939777390473324d59a814383fcfdb48cc7af635a",
			repeats: [3]string{
				"evidence:sha256:5e665ff150fad0fca8ae13a3de609f555ed6b67046c6697a45bfbb48bc9f6cd5",
				"evidence:sha256:64557c547d3c281bbb4ee77174f28d2c8ba19519197d9d21b54afd54d0abc469",
				"evidence:sha256:edf1b490201bba3344c73a361a4c8af2db1af0654258e8fd6e3786b11805ec8c",
			},
		},
	}
	requireGuardCohorts(t, fixtures, "95c0ac02654d9e9034451f9ab1a4d4b2abd0646b")
	t.Log("control readmission: 2 exact models, 3 isolated repeats each; historical controls retained. Six other text cohorts, chat, modalities and full benchmark suites are excluded.")
}

func requireGuardCohorts(t *testing.T, fixtures []guardCohort, producer string) {
	t.Helper()
	if len(fixtures) == 0 {
		t.Fatal("empty guard cohort")
	}
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT to check exact guard cohorts; no models execute")
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
				if fresh.Surface != surface || fresh.Commit != producer {
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
	if len(seen) != len(fixtures)*len(fixtures[0].repeats) {
		t.Fatalf("accepted cohort denominator differs: %d distinct records", len(seen))
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
}
