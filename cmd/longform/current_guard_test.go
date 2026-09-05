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
				"evidence:sha256:14c8126978a28677c330fdda7a1eb80d4c103ed312f08c3546e54d47b31dbb36",
				"evidence:sha256:ec2dd785dcbb0212b458a419e8dc13724341a2ae19f435ca3deca12bbbb10bb7",
				"evidence:sha256:6de760ae212837fb2ed28c65c104284f25c6af9466d14ae1d75a6e1518c1a67e",
			},
		},
		{
			name:       "E4B request session",
			historical: "evidence:sha256:4d51503d37ae7b10d80a3cb939777390473324d59a814383fcfdb48cc7af635a",
			repeats: [3]string{
				"evidence:sha256:691583cd94967c89162c90b5c9cb5ef9463c892e5c060d77c5eeccb2ddbbe3b0",
				"evidence:sha256:b6e7ae4ae005fc650babfa7eb5e530a59a10abc540b407bacb95fc689a6ebc63",
				"evidence:sha256:467b065ec9bc86f2bd938bff6d88c0f1079bc2e2e3a003f9a0ba11bd61fea356",
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
				if fresh.Surface != surface || fresh.Commit != "519a2690db1a39e16b6e3254b7556c56dbc549ea" {
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
