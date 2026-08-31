package gate

import (
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/repoanalysis"
)

func TestModernGoNoBroadSuppressions(t *testing.T) {
	root := filepath.Join("..", "..")
	census, err := repoanalysis.BuildModernGoCensus(root, repoanalysis.ModernGoTargetVersion)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := repoanalysis.LoadModernGoBaseline(
		filepath.Join(root, filepath.FromSlash(repoanalysis.ModernGoBaselineFile)),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := repoanalysis.AdmitModernGoRatchet(baseline, census, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}
	counts := map[string]int{}
	for _, finding := range census.Findings {
		for _, site := range finding.Candidates {
			counts[finding.ID+"\x00"+site.Path+"\x00"+site.Symbol]++
		}
	}
	covered := map[string]int{}
	for _, exception := range baseline.Exceptions {
		key := exception.Guideline + "\x00" + exception.Path + "\x00" + exception.Symbol
		if counts[key] == 0 || exception.CandidateCeiling != counts[key] {
			t.Errorf("broad or stale exception %s %s:%s covers %d, exact candidates %d",
				exception.Guideline, exception.Path, exception.Symbol, exception.CandidateCeiling, counts[key])
		}
		covered[exception.Guideline] += exception.CandidateCeiling
	}
	for _, finding := range census.Findings {
		if covered[finding.ID] != len(finding.Candidates) {
			t.Errorf("guideline %s exception coverage = %d, want %d", finding.ID, covered[finding.ID], len(finding.Candidates))
		}
	}
	for _, guideline := range baseline.Guidelines {
		if guideline.CandidateCeiling != 0 {
			t.Errorf("guideline %s retains broad aggregate ceiling %d", guideline.ID, guideline.CandidateCeiling)
		}
	}
}
