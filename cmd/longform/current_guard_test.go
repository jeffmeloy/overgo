package main

import (
	"cmp"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/longform"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
)

type guardCohort struct {
	name, historical  string
	initialModel      string
	retired           bool
	repeats           [3]string
	priorProducer     string
	prior             [3]string
	historicalSurface string
}

// TestCurrentGuardControls checks retained controls without measuring models.
func TestCurrentGuardControls(t *testing.T) {
	selected := readGuardCatalog(t)
	fixtures := []guardCohort{
		{
			name:       "Qwen 0.5B",
			historical: "evidence:sha256:0a6076ca812d7b2d0eb8bc17eef3fda6913ecec8dfbfd0e366e14f42eba12e51",
		},
		{
			name:       "E4B",
			historical: "evidence:sha256:4d51503d37ae7b10d80a3cb939777390473324d59a814383fcfdb48cc7af635a",
		},
	}
	for index := range fixtures {
		fixture := &fixtures[index]
		records := selected.Cohorts[fixture.name]
		if len(records) != len(fixture.repeats) {
			t.Fatalf("%s requires three complete immutable records", fixture.name)
		}
		fixture.repeats = [3]string(records)
	}
	requireGuardCohorts(t, fixtures, selected.Producer, false)
	t.Log("control readmission: 2 exact models, 3 isolated repeats each; historical controls retained. Six other text cohorts, chat, modalities and full benchmark suites are excluded.")
}

func requireGuardCohorts(t *testing.T, fixtures []guardCohort, producer string, catalog bool) {
	t.Helper()
	if os.Getenv(testskip.StoreAcceptanceEnv) == "" {
		t.Skip(testskip.StoreAcceptance)
	}
	if len(fixtures) == 0 {
		t.Fatal("empty guard cohort")
	}
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
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

	lineage := cmp.Or(fixtures[0].historical, fixtures[0].repeats[0])
	requireStoreLineage(t, store, lineage)
	seen := make(map[artifact.ID]bool)
	live := 0
	opts := options{Root: filepath.Join("..", ".."), Repository: roots.Store, Corpus: "testdata/guard-corpus.txt", ValidateBaselines: true}
	for _, fixture := range fixtures {
		if catalog && fixture.historicalSurface != "" {
			t.Fatal("historical source binding cannot grant current catalog credit")
		}
		if !fixture.retired && fixture.historicalSurface == "" {
			live++
		}
		t.Run(fixture.name, func(t *testing.T) {
			var prior []longform.Result
			priorSeen := make(map[artifact.ID]bool)
			for _, text := range fixture.prior {
				if text == "" {
					continue
				}
				id, err := artifact.ParseID(text)
				if err != nil || priorSeen[id] {
					t.Fatalf("invalid or duplicate prior record %q: %v", text, err)
				}
				priorSeen[id] = true
				accepted, err := longform.ReadBaseline(t.Context(), store, id)
				if err != nil || accepted.Result.Commit != fixture.priorProducer {
					t.Fatalf("prior record lost its admitted producer: %v", err)
				}
				prior = append(prior, accepted.Result)
			}
			var historical longform.Summary
			if fixture.initialModel != "" {
				id, err := artifact.ParseID(fixture.initialModel)
				if err != nil || id.Kind() != artifact.KindModel || fixture.historical != "" {
					t.Fatal("initial calibration requires one exact model and no substituted historical reference")
				}
				t.Logf("exact serving model %s; retained full-budget references=%d", fixture.initialModel, len(prior))
			} else {
				historical = measuredGuardRecord(t, store, fixture.historical)
				if err := validateGuard(historical.Result); err != nil {
					t.Fatal(err)
				}
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
				if !fixture.retired && (fresh.Surface != cmp.Or(fixture.historicalSurface, surface) || fresh.Commit != producer) {
					t.Fatal("control does not identify the required inference surface and clean measured producer")
				}
				if !fixture.retired && fixture.historicalSurface == "" && fresh.Shape.WarmupOutputTokens != longform.WarmupOutputTokens {
					t.Fatal("current short measurement lacks its exact-shape initialization contract")
				}
				if fixture.initialModel != "" && fresh.Program.Model.String() != fixture.initialModel {
					t.Fatal("initial calibration substituted another model")
				}
				if err := validateGuard(fresh); err != nil {
					t.Fatal(err)
				}
				if !fixture.retired {
					if err := checkGuardShortBenchmark(t.Context(), store, fresh); err != nil {
						t.Fatal(err)
					}
				}
				if historical.Record.Valid() {
					if verdict := longform.Compare(historical.Result, fresh, fresh.Floors, fresh.Floors.CheckRungCeiling); !verdict.Passed {
						t.Fatalf("historical comparison: %s", verdict)
					}
				}
				for _, previous := range prior {
					if verdict := longform.Compare(previous, fresh, fresh.Floors, fresh.Floors.CheckRungCeiling); !verdict.Passed {
						t.Fatalf("prior accepted cohort comparison: %s", verdict)
					}
				}
				if index == 0 {
					first = fresh
					if fixture.retired {
						active, err := modelrecipe.HasActiveRecipe(t.Context(), store, fresh.Program.Model, recipe.TaskInference)
						if err != nil || active {
							t.Fatalf("retired control is still active: %v", err)
						}
					} else if fixture.historicalSurface == "" {
						opts.Models = append(opts.Models, fresh.ModelPath)
						opts.Baselines = append(opts.Baselines, text)
					}
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
	if live == 0 && !catalog {
		t.Logf("historical evidence: %d records; no live model credit", len(seen))
		return
	}
	if catalog {
		opts.All, opts.Models = true, nil
	}
	targets, err := listTargets(t.Context(), opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(targets) != live {
		t.Fatal("live control selection differs from the accepted denominator")
	}
	if catalog {
		report, err := loadGuardCoverage(t.Context(), opts, targets, surface)
		if err != nil || report.Covered != live {
			t.Fatalf("live execution coverage is incomplete: %d/%d: %v", report.Covered, report.Selected, err)
		}
		t.Logf("catalog: %d exact live models, %d retired cohorts, %d distinct complete records; recipe, definition, profile, corpus and inference surface bound; no model executions", report.Covered, len(fixtures)-live, len(seen))
		return
	}
	if err := bindBaselines(t.Context(), opts, targets); err != nil {
		t.Fatal(err)
	}
	if err := validateSelectedBaselines(io.Discard, targets, surface); err != nil {
		t.Fatal(err)
	}
}
