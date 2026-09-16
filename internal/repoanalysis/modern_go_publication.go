package repoanalysis

import (
	"fmt"
	"reflect"
	"time"
)

const (
	// ModernGoPublishedCensusFile is the repository-relative closure evidence.
	ModernGoPublishedCensusFile = "docs/modern_go_census.json"
	modernGoPublishedSchema     = "overgo-modern-go-census/v1"
)

// ModernGoPublishedResolution is the complete disposition of one applicable
// guideline without duplicating its exact source sites or exception records.
type ModernGoPublishedResolution struct {
	ID          string       `json:"id"`
	Since       string       `json:"since"`
	Risk        ModernGoRisk `json:"risk"`
	Measured    bool         `json:"measured"`
	Candidates  int          `json:"candidates"`
	Adopted     int          `json:"adopted"`
	Excepted    int          `json:"excepted"`
	Unresolved  int          `json:"unresolved"`
	Disposition string       `json:"disposition"`
}

// ModernGoPublishedExclusion records a catalog rule above the module floor.
type ModernGoPublishedExclusion struct {
	ID     string `json:"id"`
	Since  string `json:"since"`
	Reason string `json:"reason"`
}

// ModernGoPublishedCensus is the compact, reproducible closure report.
type ModernGoPublishedCensus struct {
	Schema             string                        `json:"schema"`
	Doc                string                        `json:"doc"`
	TargetGo           string                        `json:"target_go"`
	BuildContext       string                        `json:"build_context"`
	SourceIdentity     string                        `json:"source_identity"`
	CatalogCommit      string                        `json:"catalog_commit"`
	CatalogSHA256      string                        `json:"catalog_sha256"`
	ExceptionSHA256    string                        `json:"exceptions_sha256"`
	Applicable         int                           `json:"applicable"`
	Measured           int                           `json:"measured"`
	Candidates         int                           `json:"candidates"`
	Adopted            int                           `json:"adopted"`
	ExceptionGroups    int                           `json:"exception_groups"`
	ExceptedCandidates int                           `json:"excepted_candidates"`
	Unresolved         int                           `json:"unresolved"`
	Resolutions        []ModernGoPublishedResolution `json:"resolutions"`
	Excluded           []ModernGoPublishedExclusion  `json:"excluded"`
}

// BuildModernGoPublishedCensus derives closure evidence from exact current
// census and ratchet authority. It refuses partially resolved guidelines.
func BuildModernGoPublishedCensus(census ModernGoCensus, baseline ModernGoBaseline) (ModernGoPublishedCensus, error) {
	if err := AdmitModernGoRatchet(baseline, census, time.Time{}); err != nil {
		return ModernGoPublishedCensus{}, err
	}
	if err := admitModernGoExactExceptions(baseline, census); err != nil {
		return ModernGoPublishedCensus{}, err
	}
	published := ModernGoPublishedCensus{
		Schema:   modernGoPublishedSchema,
		Doc:      "Source-bound Go 1.26 guideline closure. Exact sites and owned exception oracles remain in the ratchet authority.",
		TargetGo: census.TargetGo, BuildContext: census.BuildContext, SourceIdentity: census.SourceIdentity,
		CatalogCommit: census.CatalogCommit, CatalogSHA256: census.CatalogSHA256,
		ExceptionSHA256: baseline.ExceptionSHA256, Applicable: len(census.Findings),
		Candidates: census.CandidateCount(), Adopted: census.AdoptedCount(), ExceptionGroups: len(baseline.Exceptions),
	}
	applicable := make(map[string]bool, len(census.Findings))
	for _, finding := range census.Findings {
		applicable[finding.ID] = true
		excepted, err := modernGoExceptionCount(baseline.Exceptions, finding)
		if err != nil {
			return ModernGoPublishedCensus{}, err
		}
		unresolved := len(finding.Candidates) - excepted
		disposition := "applied-or-not-present"
		if excepted != 0 {
			disposition = "exact-expiring-exceptions"
		}
		published.Resolutions = append(published.Resolutions, ModernGoPublishedResolution{
			ID: finding.ID, Since: finding.SinceVersion, Risk: finding.Risk, Measured: finding.Measured,
			Candidates: len(finding.Candidates), Adopted: len(finding.Adopted), Excepted: excepted,
			Unresolved: unresolved, Disposition: disposition,
		})
		if finding.Measured {
			published.Measured++
		}
		published.ExceptedCandidates += excepted
		published.Unresolved += unresolved
	}
	for _, guideline := range ModernGoCatalog() {
		if applicable[guideline.ID] {
			continue
		}
		published.Excluded = append(published.Excluded, ModernGoPublishedExclusion{
			ID:     guideline.ID,
			Since:  guideline.SinceVersion,
			Reason: fmt.Sprintf("requires Go %s; repository target is Go %s", guideline.SinceVersion, census.TargetGo),
		})
	}
	if published.Measured != published.Applicable || published.Unresolved != 0 {
		return ModernGoPublishedCensus{}, fmt.Errorf("modern-Go closure is incomplete: measured=%d/%d unresolved=%d",
			published.Measured, published.Applicable, published.Unresolved)
	}
	return published, nil
}

// ValidateModernGoPublishedCensus rejects any published/source drift.
func ValidateModernGoPublishedCensus(published, expected ModernGoPublishedCensus) error {
	if !reflect.DeepEqual(published, expected) {
		return fmt.Errorf("published modern-Go census differs from current source, catalog, or exception authority")
	}
	return nil
}

// Closed census publication requires exact exceptions and zero aggregate debt.
func admitModernGoExactExceptions(baseline ModernGoBaseline, census ModernGoCensus) error {
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
