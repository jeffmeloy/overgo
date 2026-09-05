package gate

import (
	"fmt"

	"overgo/internal/repoanalysis"
)

// admitModernGoExactExceptions supplements the general ratchet with the
// repository's zero aggregate debt and exact site exception policy. The gate
// supplies its already computed live census; tests need no repository scan.
func admitModernGoExactExceptions(baseline repoanalysis.ModernGoBaseline, census repoanalysis.ModernGoCensus) error {
	type siteKey struct{ guideline, path, symbol string }
	counts := map[siteKey]int{}
	for _, finding := range census.Findings {
		for _, site := range finding.Candidates {
			counts[siteKey{finding.ID, site.Path, site.Symbol}]++
		}
	}
	for _, exception := range baseline.Exceptions {
		key := siteKey{exception.Guideline, exception.Path, exception.Symbol}
		if counts[key] == 0 || exception.CandidateCeiling != counts[key] {
			return fmt.Errorf("broad or stale exception %s %s:%s covers %d, exact candidates %d",
				exception.Guideline, exception.Path, exception.Symbol, exception.CandidateCeiling, counts[key])
		}
		delete(counts, key)
	}
	if len(counts) != 0 {
		return fmt.Errorf("modern-Go exceptions leave %d candidate sites uncovered", len(counts))
	}
	for _, guideline := range baseline.Guidelines {
		if guideline.CandidateCeiling != 0 {
			return fmt.Errorf("guideline %s retains broad aggregate ceiling %d", guideline.ID, guideline.CandidateCeiling)
		}
	}
	return nil
}
