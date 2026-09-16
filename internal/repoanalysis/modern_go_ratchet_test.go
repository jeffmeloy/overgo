package repoanalysis

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestModernGoRatchetCannotIncreaseDebt(t *testing.T) {
	census := modernGoRatchetCensus()
	baseline, err := BuildModernGoBaseline(census)
	if err != nil {
		t.Fatal(err)
	}
	census.Findings[0].Candidates = append(census.Findings[0].Candidates, ModernGoSite{
		Path: "internal/sample/sample.go", Line: 2, Package: "example/internal/sample", Symbol: "second",
	})
	if err := AdmitModernGoRatchet(baseline, census, time.Now()); err == nil || !strings.Contains(err.Error(), "debt increased") {
		t.Fatalf("increased debt result = %v", err)
	}
}

func TestModernGoRatchetCannotLoseCoverage(t *testing.T) {
	census := modernGoRatchetCensus()
	baseline, err := BuildModernGoBaseline(census)
	if err != nil {
		t.Fatal(err)
	}
	census.Findings[0].Measured = false
	if err := AdmitModernGoRatchet(baseline, census, time.Now()); err == nil || !strings.Contains(err.Error(), "coverage lost") {
		t.Fatalf("lost measurement result = %v", err)
	}
	census = modernGoRatchetCensus()
	census.Findings[0].TypedFiles--
	if err := AdmitModernGoRatchet(baseline, census, time.Now()); err == nil || !strings.Contains(err.Error(), "typed coverage fell") {
		t.Fatalf("lost typed coverage result = %v", err)
	}
}

func TestModernGoRatchetRejectsExpiredExceptions(t *testing.T) {
	census := modernGoRatchetCensus()
	baseline, err := BuildModernGoBaseline(census)
	if err != nil {
		t.Fatal(err)
	}
	baseline.Exceptions = []ModernGoException{{
		Guideline: "any", Path: "internal/sample/sample.go", Symbol: "first", Owner: "sample",
		Reason: "compatibility", Oracle: "go test ./internal/sample", Retirement: "remove legacy API",
		Expires: "2026-01-01", CandidateCeiling: 1,
	}}
	if err := AdmitModernGoRatchet(baseline, census, time.Date(2026, time.August, 29, 0, 0, 0, 0, time.UTC)); err == nil || !strings.Contains(err.Error(), "expired") {
		t.Fatalf("expired exception result = %v", err)
	}
}

func TestModernGoRatchetRejectsIntroducedDebt(t *testing.T) {
	previous := modernGoRatchetCensus()
	baseline, err := BuildModernGoBaseline(previous)
	if err != nil {
		t.Fatal(err)
	}
	candidate := modernGoRatchetCensus()
	candidate.Findings[0].Candidates = []ModernGoSite{{
		Path: "internal/sample/sample.go", Line: 2, Package: "example/internal/sample", Symbol: "replacement",
	}}
	if err := AdmitModernGoDelta(baseline, previous, candidate, []string{"internal/sample/sample.go"}); err == nil || !strings.Contains(err.Error(), "introduced") {
		t.Fatalf("introduced debt result = %v", err)
	}
}

func TestModernGoRatchetSurvivesWorktreeRelocation(t *testing.T) {
	measure := func(root string) (ModernGoCensus, ModernGoBaseline) {
		t.Helper()
		for name, content := range map[string]string{
			"go.mod":                    "module example\n\ngo 1.26\n",
			"internal/sample/sample.go": "package sample\n\nfunc Values() []int { return []int{1, 2, 3} }\n",
		} {
			path := filepath.Join(root, filepath.FromSlash(name))
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		census, err := BuildModernGoCensus(root, ModernGoTargetVersion)
		if err != nil {
			t.Fatal(err)
		}
		baseline, err := BuildModernGoBaseline(census)
		if err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(baseline)
		if err != nil {
			t.Fatal(err)
		}
		baselinePath := filepath.Join(root, filepath.FromSlash(ModernGoBaselineFile))
		if err := os.MkdirAll(filepath.Dir(baselinePath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(baselinePath, raw, 0o600); err != nil {
			t.Fatal(err)
		}
		loaded, err := LoadModernGoBaseline(baselinePath)
		if err != nil {
			t.Fatal(err)
		}
		if err := AdmitModernGoRatchet(loaded, census, time.Time{}); err != nil {
			t.Fatal(err)
		}
		return census, loaded
	}
	firstCensus, firstBaseline := measure(filepath.Join(t.TempDir(), "first-worktree"))
	secondCensus, secondBaseline := measure(filepath.Join(t.TempDir(), "relocated-worktree"))
	if !reflect.DeepEqual(firstCensus, secondCensus) || !reflect.DeepEqual(firstBaseline, secondBaseline) {
		t.Fatal("modern-Go census or ratchet authority depends on the worktree path")
	}
}

func TestModernGoBaselineLoweringIsMonotonic(t *testing.T) {
	census := modernGoRatchetCensus()
	baseline, err := BuildModernGoBaseline(census)
	if err != nil {
		t.Fatal(err)
	}
	census.Findings[0].Candidates = nil
	lowered, err := LowerModernGoBaseline(baseline, census)
	if err != nil {
		t.Fatal(err)
	}
	if lowered.Guidelines[0].CandidateCeiling != 0 {
		t.Fatalf("lowered ceiling = %d", lowered.Guidelines[0].CandidateCeiling)
	}
	census.Findings[0].Candidates = append(census.Findings[0].Candidates, ModernGoSite{}, ModernGoSite{})
	if _, err := LowerModernGoBaseline(lowered, census); err == nil {
		t.Fatal("baseline lowering admitted increased debt")
	}
}

func modernGoRatchetCensus() ModernGoCensus {
	return ModernGoCensus{
		TargetGo: ModernGoTargetVersion, SourceIdentity: "source", BuildContext: "test/test",
		CatalogCommit: ModernGoCatalogCommit, CatalogSHA256: ModernGoCatalogSHA256,
		Findings: []ModernGoFinding{{
			ID: "any", Risk: ModernGoRiskMechanical, Measured: true, InspectedFiles: 10, TypedFiles: 9,
			Candidates: []ModernGoSite{{
				Path: "internal/sample/sample.go", Line: 1, Package: "example/internal/sample", Symbol: "first",
			}},
		}},
	}
}

func TestModernGoBaselineRetiresResolvedExceptions(t *testing.T) {
	initial := modernGoRatchetCensus()
	first := initial.Findings[0].Candidates[0]
	second := first
	second.Symbol = "second"
	initial.Findings[0].Candidates = []ModernGoSite{first, first, second}
	authority, err := BuildModernGoBaseline(initial)
	if err != nil {
		t.Fatal(err)
	}
	authority.Guidelines[0].CandidateCeiling = 0
	for _, site := range []ModernGoSite{first, second} {
		authority.Exceptions = append(authority.Exceptions, ModernGoException{
			Guideline: initial.Findings[0].ID, Path: site.Path, Symbol: site.Symbol, Owner: "sample", Reason: "fixture compatibility",
			Oracle: "go test ./internal/sample", Retirement: "remove legacy fixture", Expires: "2099-01-01", CandidateCeiling: 1,
		})
	}
	authority.Exceptions[0].CandidateCeiling = 2
	authority.ExceptionSHA256, err = ModernGoExceptionIdentity(authority.Exceptions)
	if err != nil {
		t.Fatal(err)
	}
	foreign := first
	foreign.Symbol = "new-debt"
	for _, test := range []struct {
		name   string
		sites  []ModernGoSite
		counts []int
		reject bool
	}{
		{"unchanged", []ModernGoSite{first, first, second}, []int{2, 1}, false},
		{"partial", []ModernGoSite{first, second}, []int{1, 1}, false},
		{"scope removed", []ModernGoSite{second}, []int{1}, false},
		{"all resolved", nil, nil, false},
		{"expanded", []ModernGoSite{first, first, first, second}, nil, true},
		{"moved debt", []ModernGoSite{foreign, second}, nil, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			current := modernGoRatchetCensus()
			current.Findings[0].Candidates = test.sites
			if _, err := BuildModernGoPublishedCensus(current, authority); (err == nil) != (test.name == "unchanged") {
				t.Fatalf("publication admitted stale or expanded authority: %v", err)
			}
			before, err := json.Marshal(authority)
			if err != nil {
				t.Fatal(err)
			}
			lowered, err := LowerModernGoBaseline(authority, current)
			after, marshalErr := json.Marshal(authority)
			if marshalErr != nil {
				t.Fatal(marshalErr)
			}
			if string(before) != string(after) {
				t.Fatal("lowering mutated authority")
			}
			if (err != nil) != test.reject {
				t.Fatalf("rejected=%v want %v: %v", err != nil, test.reject, err)
			}
			if test.reject {
				return
			}
			if _, err := BuildModernGoPublishedCensus(current, lowered); err != nil {
				t.Fatalf("resolved authority does not satisfy publication: %v", err)
			}
			if len(lowered.Exceptions) != len(test.counts) {
				t.Fatalf("exceptions=%d want %d", len(lowered.Exceptions), len(test.counts))
			}
			for i, exception := range lowered.Exceptions {
				if exception.CandidateCeiling != test.counts[i] {
					t.Fatalf("ceiling=%d want %d", exception.CandidateCeiling, test.counts[i])
				}
				index := 0
				if exception.Symbol == second.Symbol {
					index = 1
				}
				original := authority.Exceptions[index]
				original.CandidateCeiling = exception.CandidateCeiling
				if !reflect.DeepEqual(original, exception) {
					t.Fatal("changed retained policy")
				}
			}
			identity, err := ModernGoExceptionIdentity(lowered.Exceptions)
			if err != nil || identity != lowered.ExceptionSHA256 {
				t.Fatalf("stale exception identity: %v", err)
			}
			again, err := LowerModernGoBaseline(lowered, current)
			if err != nil || !reflect.DeepEqual(again, lowered) {
				t.Fatalf("not idempotent: %v", err)
			}
		})
	}
	for _, name := range []string{"tampered authority", "lost coverage"} {
		t.Run(name, func(t *testing.T) {
			baseline := authority
			current := modernGoRatchetCensus()
			current.Findings[0].Candidates = nil
			if name == "tampered authority" {
				baseline.ExceptionSHA256 = "tampered"
			} else {
				current.Findings[0].Measured = false
			}
			if _, err := LowerModernGoBaseline(baseline, current); err == nil {
				t.Fatal("invalid authority or coverage admitted while retiring all scopes")
			}
		})
	}
}
