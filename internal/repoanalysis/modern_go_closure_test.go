package repoanalysis

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestEveryApplicableModernGoGuidelineResolved(t *testing.T) {
	census, baseline := modernGoRepositoryExceptionAuthority(t)
	published, err := BuildModernGoPublishedCensus(census, baseline)
	if err != nil {
		t.Fatal(err)
	}
	if published.Applicable != len(census.Findings) || published.Measured != published.Applicable ||
		len(published.Resolutions) != published.Applicable || published.Unresolved != 0 ||
		published.ExceptedCandidates != published.Candidates {
		t.Fatalf("incomplete modern-Go resolution: %+v", published)
	}
	for _, resolution := range published.Resolutions {
		if !resolution.Measured || resolution.Unresolved != 0 ||
			resolution.Candidates != resolution.Excepted && resolution.Candidates != 0 {
			t.Errorf("guideline %s is unresolved: %+v", resolution.ID, resolution)
		}
	}
}

// TestModernGoBaselineHoldsOnlyReviewedThresholds holds the ratchet file to
// reviewed floors and nothing measured from the tree. It once carried a hash
// of the whole source tree, so every landing rewrote it and a reviewer could
// not tell a moved floor from a touched file. A tree that changes and grows
// without bettering a floor leaves the lowered baseline byte-identical; a tree
// with fewer untyped files moves the typed-coverage floor, and until the file
// is lowered the gate's rule refuses it as loose. The closure report is
// computed on demand: no copy of it is kept in the tree to be checked against
// the computation it came from.
func TestModernGoBaselineHoldsOnlyReviewedThresholds(t *testing.T) {
	census, baseline := modernGoRepositoryExceptionAuthority(t)
	if err := ModernGoBaselineHolds(baseline, census); err != nil {
		t.Fatal(err)
	}
	written, err := json.Marshal(baseline)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(written), "source_identity") {
		t.Fatal("the baseline carries a source identity again")
	}
	grown := census
	grown.SourceIdentity = strings.Repeat("0", len(census.SourceIdentity))
	grown.Findings = slices.Clone(census.Findings)
	for index := range grown.Findings {
		grown.Findings[index].InspectedFiles += 7
		grown.Findings[index].TypedFiles += 7
	}
	lowered, err := LowerModernGoBaseline(baseline, grown)
	if err != nil {
		t.Fatal(err)
	}
	if rewritten, err := json.Marshal(lowered); err != nil || !bytes.Equal(rewritten, written) {
		t.Fatalf("a tree that changed and grew rewrote the baseline (%v):\n%s\nwas:\n%s", err, rewritten, written)
	}
	if err := ModernGoBaselineHolds(baseline, grown); err != nil {
		t.Fatalf("a tree that changed and grew made the baseline stale: %v", err)
	}
	typed := census
	typed.Findings = slices.Clone(census.Findings)
	for index := range typed.Findings {
		typed.Findings[index].TypedFiles++
	}
	if err := ModernGoBaselineHolds(baseline, typed); err == nil || !strings.Contains(err.Error(), "loose") {
		t.Fatalf("a bettered typed-coverage floor did not make the baseline loose: %v", err)
	}
	if _, err := os.Stat(filepath.Join("..", "..", "docs", "modern_go_census.json")); !os.IsNotExist(err) {
		t.Fatalf("a copy of the closure report is kept in the tree: %v", err)
	}
}

func TestModernGoRatchetAtClosure(t *testing.T) {
	census, baseline := modernGoRepositoryExceptionAuthority(t)
	if err := ModernGoBaselineHolds(baseline, census); err != nil {
		t.Fatal(err)
	}
	for _, guideline := range baseline.Guidelines {
		if guideline.CandidateCeiling != 0 {
			t.Errorf("guideline %s retains unresolved ceiling %d", guideline.ID, guideline.CandidateCeiling)
		}
	}
	if err := AdmitModernGoRatchet(baseline, census, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	applicable, err := ModernGoApplicableGuidelines(ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	published, err := BuildModernGoPublishedCensus(census, baseline)
	if err != nil {
		t.Fatal(err)
	}
	if len(published.Excluded) != len(ModernGoCatalog())-len(applicable) {
		t.Fatalf("later-version exclusions = %d, want %d", len(published.Excluded), len(ModernGoCatalog())-len(applicable))
	}
}

// TestModernGoGateSnapshotReachesClosure holds the census the gate computes
// from its own snapshot to the same closure as the census command.
func TestModernGoGateSnapshotReachesClosure(t *testing.T) {
	t.Parallel()
	census, err := modernGoSnapshotCensus()
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := LoadModernGoBaseline(filepath.Join("..", "..", filepath.FromSlash(ModernGoBaselineFile)))
	if err != nil {
		t.Fatal(err)
	}
	if err := AdmitModernGoRatchet(baseline, census, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	if err := ModernGoBaselineHolds(baseline, census); err != nil {
		t.Fatal(err)
	}
	if _, err := BuildModernGoPublishedCensus(census, baseline); err != nil {
		t.Fatal(err)
	}
}
