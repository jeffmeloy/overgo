package gate

import (
	"strings"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestModernGoNoBroadSuppressions(t *testing.T) {
	for _, test := range []struct {
		name   string
		change func(*repoanalysis.ModernGoBaseline, *repoanalysis.ModernGoCensus)
		want   string
	}{
		{name: "exact coverage"},
		{name: "empty candidate set", change: func(b *repoanalysis.ModernGoBaseline, c *repoanalysis.ModernGoCensus) {
			b.Exceptions = nil
			c.Findings[0].Candidates = nil
		}},
		{name: "aggregate ceiling", change: func(b *repoanalysis.ModernGoBaseline, _ *repoanalysis.ModernGoCensus) {
			b.Guidelines[0].CandidateCeiling = 1
		}, want: "broad aggregate ceiling"},
		{name: "stale exception", change: func(_ *repoanalysis.ModernGoBaseline, c *repoanalysis.ModernGoCensus) {
			c.Findings[0].Candidates = nil
		}, want: "broad or stale exception"},
		{name: "excess ceiling", change: func(b *repoanalysis.ModernGoBaseline, _ *repoanalysis.ModernGoCensus) {
			b.Exceptions[0].CandidateCeiling++
		}, want: "broad or stale exception"},
		{name: "insufficient ceiling", change: func(b *repoanalysis.ModernGoBaseline, _ *repoanalysis.ModernGoCensus) {
			b.Exceptions[0].CandidateCeiling--
		}, want: "broad or stale exception"},
		{name: "wrong path", change: func(b *repoanalysis.ModernGoBaseline, _ *repoanalysis.ModernGoCensus) {
			b.Exceptions[0].Path = "internal/other.go"
		}, want: "broad or stale exception"},
		{name: "wrong symbol", change: func(b *repoanalysis.ModernGoBaseline, _ *repoanalysis.ModernGoCensus) {
			b.Exceptions[0].Symbol = "other"
		}, want: "broad or stale exception"},
		{name: "wrong guideline", change: func(b *repoanalysis.ModernGoBaseline, _ *repoanalysis.ModernGoCensus) {
			b.Exceptions[0].Guideline = "other"
		}, want: "broad or stale exception"},
		{name: "duplicate exception", change: func(b *repoanalysis.ModernGoBaseline, _ *repoanalysis.ModernGoCensus) {
			b.Exceptions = append(b.Exceptions, b.Exceptions[0])
		}, want: "broad or stale exception"},
		{name: "uncovered site", change: func(_ *repoanalysis.ModernGoBaseline, c *repoanalysis.ModernGoCensus) {
			c.Findings[0].Candidates = append(c.Findings[0].Candidates, repoanalysis.ModernGoSite{Path: "internal/new.go", Symbol: "new"})
		}, want: "uncovered"},
	} {
		t.Run(test.name, func(t *testing.T) {
			baseline := repoanalysis.ModernGoBaseline{
				Guidelines: []repoanalysis.ModernGoBaselineGuideline{{ID: "rule"}},
				Exceptions: []repoanalysis.ModernGoException{{Guideline: "rule", Path: "internal/example.go", Symbol: "example", CandidateCeiling: 2}},
			}
			census := repoanalysis.ModernGoCensus{Findings: []repoanalysis.ModernGoFinding{{ID: "rule", Candidates: []repoanalysis.ModernGoSite{
				{Path: "internal/example.go", Symbol: "example", Line: 10},
				{Path: "internal/example.go", Symbol: "example", Line: 20},
			}}}}
			if test.change != nil {
				test.change(&baseline, &census)
			}
			err := admitModernGoExactExceptions(baseline, census)
			if test.want == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
}
