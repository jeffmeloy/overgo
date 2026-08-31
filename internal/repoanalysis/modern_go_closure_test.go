package repoanalysis

import (
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/jsonfile"
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

func TestModernGoPublishedCensusMatchesSource(t *testing.T) {
	census, baseline := modernGoRepositoryExceptionAuthority(t)
	expected, err := BuildModernGoPublishedCensus(census, baseline)
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join("..", "..")
	var published ModernGoPublishedCensus
	if err := jsonfile.DecodeStrict(filepath.Join(root, filepath.FromSlash(ModernGoPublishedCensusFile)), &published); err != nil {
		t.Fatal(err)
	}
	if err := ValidateModernGoPublishedCensus(published, expected); err != nil {
		t.Fatal(err)
	}
}

func TestModernGoRatchetAtClosure(t *testing.T) {
	census, baseline := modernGoRepositoryExceptionAuthority(t)
	if baseline.SourceIdentity != census.SourceIdentity {
		t.Fatalf("ratchet source = %s, current source = %s", baseline.SourceIdentity, census.SourceIdentity)
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

func TestModernGoGateSnapshotMatchesPublishedCensus(t *testing.T) {
	root := filepath.Join("..", "..")
	snapshot, err := DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	selection, err := HostBuildSelection(root, "./cmd/...", "./internal/...")
	if err != nil {
		t.Fatal(err)
	}
	census, err := ModernGoCensusSnapshot(snapshot, selection, ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := LoadModernGoBaseline(filepath.Join(root, filepath.FromSlash(ModernGoBaselineFile)))
	if err != nil {
		t.Fatal(err)
	}
	if err := AdmitModernGoRatchet(baseline, census, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	var published ModernGoPublishedCensus
	if err := jsonfile.DecodeStrict(filepath.Join(root, filepath.FromSlash(ModernGoPublishedCensusFile)), &published); err != nil {
		t.Fatal(err)
	}
	expected, err := BuildModernGoPublishedCensus(census, baseline)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateModernGoPublishedCensus(published, expected); err != nil {
		t.Fatal(err)
	}
}
