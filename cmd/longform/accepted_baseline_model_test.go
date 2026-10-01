//go:build modeltest

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
)

// TestAcceptedBaselineEvidence audits immutable historical guard records but
// refuses to promote them without measured realized residency. The old Qwen
// and E4B results remain diagnostic evidence; a current baseline must be
// reacquired on the same observed weight path as its comparison.
func TestAcceptedBaselineEvidence(t *testing.T) {
	testskip.Short(t)
	if os.Getenv(dataroot.Env) == "" {
		testskip.NotApplicable(t, "integration: set OVERGO_DATA_ROOT to audit the exact stored guard records; no measurement runs")
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
	const acceptedSurface = "d3ff45edf061f7a2f8833d9625a6154532e1760fd08878cb89343f46a728341f"
	const acceptedCommit = "e4b9d1e6b1b46626f1b934a7360db76c960277ce"
	fixtures := []struct {
		name, before, refused string
		after                 [3]string // Three sequential isolated samples; first is reference, never best-run selection.
	}{
		{
			name:    "Qwen capacity session",
			before:  "evidence:sha256:8fe6efde3661a9f24a481002fbd5696536e52f782e00580e0ab6d991d3b06e39",
			refused: "evidence:sha256:1c61e2821f010632b9b7a99b05261d7ccad9649929a9bfbc1a923d4430bfd8e7",
			after: [3]string{
				"evidence:sha256:0190b70b93219859525aff79e925f30ccc8b6be5647cb36eba2838ca56bef2c8",
				"evidence:sha256:7475932433954855147a950a1aff5db66006d18225266cdae90dd5c1fd4cf81b",
				"evidence:sha256:2a0f7c99ed9d4c8d4dfeca916c494b371c3cf413bfa5f20acfbb21248856c445",
			},
		},
		{
			name:    "E4B request session",
			before:  "evidence:sha256:2da44cefe6b7db08e49315acf50b727421fb2fe212197ee4cba16db680bcdd43",
			refused: "evidence:sha256:83a69ff733e8cd2f52388bb94e28eb870fdd8642e10b189a37923c24b15a0d2a",
			after: [3]string{
				"evidence:sha256:3d6b27ffaf0fa747799b64f85eda10e5261cf09465c0457652e0a2d4f422e928",
				"evidence:sha256:a7804c9c52b752a5b3b80979b741d1662fd3baf8f18733517c1923d3593472ca",
				"evidence:sha256:3497b3b65674f032badd1df1f2395c0dd3c166981d66bed7c5e242455c28eff8",
			},
		},
	}
	requireStoreLineage(t, store, fixtures[0].before)
	seen := make(map[artifact.ID]bool)
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			before := measuredGuardRecord(t, store, fixture.before)
			refused := measuredGuardRecord(t, store, fixture.refused)
			if err := validateGuard(refused.Result); err == nil {
				t.Fatal("original failed or incomplete measurement accepted")
			}
			if !before.Result.Verdict.Passed {
				if _, err := longform.ReadBaseline(t.Context(), store, before.Record); err == nil {
					t.Fatal("failed pre-fix measurement accepted as a healthy baseline")
				}
			}
			var first longform.Result
			for index, text := range fixture.after {
				id, err := artifact.ParseID(text)
				if err != nil || seen[id] || id == before.Record || id == refused.Record {
					t.Fatalf("invalid, duplicate or self-selected record %s: %v", text, err)
				}
				seen[id] = true
				fresh := measuredGuardRecord(t, store, text).Result
				if fresh.Surface != acceptedSurface || fresh.Commit != acceptedCommit {
					t.Fatal("record does not identify the committed measured implementation")
				}
				if err := validateGuard(fresh); err == nil || !strings.Contains(err.Error(), "comparison lacks matching realized residency") {
					t.Fatal(err)
				}
				if fresh.RealizedResidency != "" {
					t.Fatal("historical record unexpectedly declares an observed residency")
				}
				if _, err := longform.ReadBaseline(t.Context(), store, id); err == nil || !strings.Contains(err.Error(), "realized residency") {
					t.Fatalf("legacy record was promoted or refused for another reason: %v", err)
				}
				if verdict := longform.Compare(before.Result, fresh, longform.DeclaredFloors(), fresh.Floors.CheckRungCeiling); verdict.Passed || !strings.Contains(strings.Join(verdict.Reasons, " "), "realized residency") {
					t.Fatalf("legacy pre-fix comparison was promoted: %s", verdict)
				}
				if index > 0 {
					if verdict := longform.Compare(first, fresh, fresh.Floors, fresh.Floors.CheckRungCeiling); verdict.Passed || !strings.Contains(strings.Join(verdict.Reasons, " "), "realized residency") {
						t.Fatalf("legacy repeat comparison was promoted: %s", verdict)
					}
				} else {
					first = fresh
				}
				t.Logf("historical record=%s judged_decode=%.1f short=1 rungs=%d; current comparison requires residency reacquisition", id, fresh.Measure.DecodeTokensPerSecond, len(fresh.Rungs))
			}
		})
	}
	if len(seen) != len(fixtures)*len(fixtures[0].after) {
		t.Fatal("missing historical model from the declared guard denominator")
	}
	t.Logf("historical guard audit: models=%d repeated_records=%d; immutable results retained but no baseline is promoted without realized residency", len(fixtures), len(seen))
}
