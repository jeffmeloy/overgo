package repoanalysis

import (
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestModernGoExceptionsAreExactOwnedAndTested(t *testing.T) {
	census, baseline := modernGoRepositoryExceptionAuthority(t)
	type group struct {
		owner string
		count int
	}
	groups := map[string]group{}
	for _, finding := range census.Findings {
		for _, site := range finding.Candidates {
			key := finding.ID + "\x00" + site.Path + "\x00" + site.Symbol
			current := groups[key]
			if current.owner != "" && current.owner != site.Package {
				t.Fatalf("candidate group %q crosses owners %q and %q", key, current.owner, site.Package)
			}
			current.owner = site.Package
			current.count++
			groups[key] = current
		}
	}
	if len(baseline.Exceptions) != len(groups) {
		t.Fatalf("exception groups = %d, want %d", len(baseline.Exceptions), len(groups))
	}
	for _, exception := range baseline.Exceptions {
		key := exception.Guideline + "\x00" + exception.Path + "\x00" + exception.Symbol
		want, ok := groups[key]
		if !ok {
			t.Errorf("exception has no exact candidate group: %s %s:%s", exception.Guideline, exception.Path, exception.Symbol)
			continue
		}
		if exception.CandidateCeiling != want.count || exception.Owner != want.owner {
			t.Errorf("exception %s %s:%s = count %d owner %q, want %d %q",
				exception.Guideline, exception.Path, exception.Symbol,
				exception.CandidateCeiling, exception.Owner, want.count, want.owner)
		}
		packagePath := filepath.ToSlash(filepath.Dir(exception.Path))
		if exception.Oracle != "go test ./"+packagePath+" -count=1" {
			t.Errorf("exception %s %s:%s oracle = %q, want exact owner package",
				exception.Guideline, exception.Path, exception.Symbol, exception.Oracle)
		}
	}
}

func TestModernGoExceptionsHaveRetirementTriggers(t *testing.T) {
	_, baseline := modernGoRepositoryExceptionAuthority(t)
	today := time.Now().UTC()
	latest := today.AddDate(1, 0, 1)
	for _, exception := range baseline.Exceptions {
		policy, ok := modernGoExceptionPolicies[exception.Guideline]
		if !ok || exception.Reason != policy.reason || exception.Retirement != policy.retirement {
			t.Errorf("exception %s %s:%s does not use its reviewed reason and retirement trigger",
				exception.Guideline, exception.Path, exception.Symbol)
		}
		expires, err := time.Parse(time.DateOnly, exception.Expires)
		if err != nil || !expires.After(today) || expires.After(latest) {
			t.Errorf("exception %s %s:%s expiry = %q, want future within one year",
				exception.Guideline, exception.Path, exception.Symbol, exception.Expires)
		}
	}
	want, err := ModernGoExceptionIdentity(baseline.Exceptions)
	if err != nil {
		t.Fatal(err)
	}
	if baseline.ExceptionSHA256 != want {
		t.Fatalf("exception authority = %s, want %s", baseline.ExceptionSHA256, want)
	}
}

// Each caller reads this census; policy and time-sensitive admission stay fresh.
var modernGoRepositoryCensus = sync.OnceValues(func() (ModernGoCensus, error) {
	return BuildModernGoCensus(filepath.Join("..", ".."), ModernGoTargetVersion)
})

func modernGoRepositoryExceptionAuthority(t *testing.T) (ModernGoCensus, ModernGoBaseline) {
	t.Helper()
	root := filepath.Join("..", "..")
	census, err := modernGoRepositoryCensus()
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
	return census, baseline
}

func TestModernGoNoBroadSuppressions(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name   string
		change func(*ModernGoBaseline, *ModernGoCensus)
		want   string
	}{
		{name: "exact coverage"},
		{name: "empty candidate set", change: func(b *ModernGoBaseline, c *ModernGoCensus) {
			b.Exceptions = nil
			c.Findings[0].Candidates = nil
		}},
		{name: "aggregate ceiling", change: func(b *ModernGoBaseline, _ *ModernGoCensus) {
			b.Guidelines[0].CandidateCeiling = 1
		}, want: "broad aggregate ceiling"},
		{name: "stale exception", change: func(_ *ModernGoBaseline, c *ModernGoCensus) {
			c.Findings[0].Candidates = nil
		}, want: "broad or stale exception"},
		{name: "excess ceiling", change: func(b *ModernGoBaseline, _ *ModernGoCensus) {
			b.Exceptions[0].CandidateCeiling++
		}, want: "broad or stale exception"},
		{name: "insufficient ceiling", change: func(b *ModernGoBaseline, _ *ModernGoCensus) {
			b.Exceptions[0].CandidateCeiling--
		}, want: "broad or stale exception"},
		{name: "wrong path", change: func(b *ModernGoBaseline, _ *ModernGoCensus) {
			b.Exceptions[0].Path = "internal/other.go"
		}, want: "broad or stale exception"},
		{name: "wrong symbol", change: func(b *ModernGoBaseline, _ *ModernGoCensus) {
			b.Exceptions[0].Symbol = "other"
		}, want: "broad or stale exception"},
		{name: "wrong guideline", change: func(b *ModernGoBaseline, _ *ModernGoCensus) {
			b.Exceptions[0].Guideline = "other"
		}, want: "broad or stale exception"},
		{name: "duplicate exception", change: func(b *ModernGoBaseline, _ *ModernGoCensus) {
			b.Exceptions = append(b.Exceptions, b.Exceptions[0])
		}, want: "broad or stale exception"},
		{name: "uncovered site", change: func(_ *ModernGoBaseline, c *ModernGoCensus) {
			c.Findings[0].Candidates = append(c.Findings[0].Candidates, ModernGoSite{Path: "internal/new.go", Symbol: "new"})
		}, want: "uncovered"},
	} {
		t.Run(test.name, func(t *testing.T) {
			baseline := ModernGoBaseline{
				Guidelines: []ModernGoBaselineGuideline{{ID: "rule"}},
				Exceptions: []ModernGoException{{Guideline: "rule", Path: "internal/example.go", Symbol: "example", CandidateCeiling: 2}},
			}
			census := ModernGoCensus{Findings: []ModernGoFinding{{ID: "rule", Candidates: []ModernGoSite{
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
