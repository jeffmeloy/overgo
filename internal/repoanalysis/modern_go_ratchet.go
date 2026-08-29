package repoanalysis

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/jsonfile"
)

const (
	// ModernGoBaselineFile is the repository-relative permanent ratchet manifest.
	ModernGoBaselineFile = "docs/modern_go_baseline.json"
	modernGoSchema       = "overgo-modern-go-ratchet/v2"
)

// ModernGoBaselineCoverage records the complete census coverage that produced
// the published ceilings.
type ModernGoBaselineCoverage struct {
	Applicable     int `json:"applicable"`
	Measured       int `json:"measured"`
	InspectedFiles int `json:"inspected_files"`
	TypedFiles     int `json:"typed_files"`
}

// ModernGoBaselineGuideline is the reviewed candidate ceiling for one rule.
type ModernGoBaselineGuideline struct {
	ID               string       `json:"id"`
	Risk             ModernGoRisk `json:"risk"`
	CandidateCeiling int          `json:"candidate_ceiling"`
}

// ModernGoException is a bounded retained legacy form. Its oracle is an exact
// executable command, and its expiry and retirement fields prevent permanent
// convenience suppressions.
type ModernGoException struct {
	Guideline        string `json:"guideline"`
	Path             string `json:"path"`
	Symbol           string `json:"symbol"`
	Owner            string `json:"owner"`
	Reason           string `json:"reason"`
	Oracle           string `json:"oracle"`
	Retirement       string `json:"retirement"`
	Expires          string `json:"expires"`
	CandidateCeiling int    `json:"candidate_ceiling"`
}

// ModernGoBaseline is the strict, relocatable modernization gate authority.
type ModernGoBaseline struct {
	Schema          string                      `json:"schema"`
	Doc             string                      `json:"doc"`
	TargetGo        string                      `json:"target_go"`
	CatalogCommit   string                      `json:"catalog_commit"`
	CatalogSHA256   string                      `json:"catalog_sha256"`
	SourceIdentity  string                      `json:"source_identity"`
	Coverage        ModernGoBaselineCoverage    `json:"coverage"`
	Guidelines      []ModernGoBaselineGuideline `json:"guidelines"`
	Exceptions      []ModernGoException         `json:"exceptions"`
	ExceptionSHA256 string                      `json:"exceptions_sha256"`
}

// BuildModernGoBaseline creates the deterministic initial ceiling document
// from a complete census. Existing exceptions are never inferred.
func BuildModernGoBaseline(census ModernGoCensus) (ModernGoBaseline, error) {
	baseline := ModernGoBaseline{
		Schema:   modernGoSchema,
		Doc:      "Pinned per-guideline candidate ceilings. The gate requires complete coverage, refuses increases, and rejects new findings in changed source even when aggregate debt falls.",
		TargetGo: census.TargetGo, CatalogCommit: census.CatalogCommit,
		CatalogSHA256: census.CatalogSHA256, SourceIdentity: census.SourceIdentity,
		Exceptions: []ModernGoException{},
	}
	baseline.Coverage.Applicable = len(census.Findings)
	for _, finding := range census.Findings {
		if finding.Measured {
			baseline.Coverage.Measured++
		}
		baseline.Coverage.InspectedFiles = max(baseline.Coverage.InspectedFiles, finding.InspectedFiles)
		baseline.Coverage.TypedFiles = max(baseline.Coverage.TypedFiles, finding.TypedFiles)
		baseline.Guidelines = append(baseline.Guidelines, ModernGoBaselineGuideline{
			ID: finding.ID, Risk: finding.Risk, CandidateCeiling: len(finding.Candidates),
		})
	}
	identity, err := ModernGoExceptionIdentity(baseline.Exceptions)
	if err != nil {
		return ModernGoBaseline{}, err
	}
	baseline.ExceptionSHA256 = identity
	if err := validateModernGoBaseline(baseline, time.Time{}); err != nil {
		return ModernGoBaseline{}, err
	}
	return baseline, nil
}

// LowerModernGoBaseline refreshes census provenance while permitting only
// lower guideline ceilings. It is the repeatable sweep operation; a caller
// cannot use it to normalize newly introduced debt into the authority.
func LowerModernGoBaseline(baseline ModernGoBaseline, census ModernGoCensus) (ModernGoBaseline, error) {
	if err := AdmitModernGoRatchet(baseline, census, time.Time{}); err != nil {
		return ModernGoBaseline{}, err
	}
	measured, err := BuildModernGoBaseline(census)
	if err != nil {
		return ModernGoBaseline{}, err
	}
	measured.Doc = baseline.Doc
	measured.Exceptions = slices.Clone(baseline.Exceptions)
	measured.ExceptionSHA256 = baseline.ExceptionSHA256
	for index := range measured.Guidelines {
		excepted, err := modernGoExceptionCount(baseline.Exceptions, census.Findings[index])
		if err != nil {
			return ModernGoBaseline{}, err
		}
		measured.Guidelines[index].CandidateCeiling = min(
			baseline.Guidelines[index].CandidateCeiling,
			len(census.Findings[index].Candidates)-excepted,
		)
	}
	return measured, nil
}

// LoadModernGoBaseline strictly decodes the permanent ratchet manifest.
func LoadModernGoBaseline(name string) (ModernGoBaseline, error) {
	var baseline ModernGoBaseline
	if err := jsonfile.DecodeStrict(name, &baseline); err != nil {
		return ModernGoBaseline{}, err
	}
	return baseline, nil
}

// AdmitModernGoRatchet checks catalog identity, coverage, exceptions, and every
// per-rule ceiling. The caller supplies today so expiry checks are testable.
func AdmitModernGoRatchet(baseline ModernGoBaseline, census ModernGoCensus, today time.Time) error {
	if err := validateModernGoBaseline(baseline, today); err != nil {
		return err
	}
	if census.TargetGo != baseline.TargetGo || census.CatalogCommit != baseline.CatalogCommit || census.CatalogSHA256 != baseline.CatalogSHA256 {
		return fmt.Errorf("modern-Go ratchet catalog identity differs from %s", ModernGoBaselineFile)
	}
	if len(census.Findings) != baseline.Coverage.Applicable || len(census.Findings) != len(baseline.Guidelines) {
		return fmt.Errorf("modern-Go ratchet coverage lost: findings=%d applicable=%d", len(census.Findings), baseline.Coverage.Applicable)
	}
	for index, ceiling := range baseline.Guidelines {
		finding := census.Findings[index]
		if finding.ID != ceiling.ID || finding.Risk != ceiling.Risk || !finding.Measured {
			return fmt.Errorf("modern-Go ratchet coverage lost at %s", ceiling.ID)
		}
		if finding.InspectedFiles == 0 || baseline.Coverage.InspectedFiles*finding.TypedFiles < baseline.Coverage.TypedFiles*finding.InspectedFiles {
			return fmt.Errorf("modern-Go ratchet typed coverage fell at %s: %d/%d below %d/%d",
				finding.ID, finding.TypedFiles, finding.InspectedFiles,
				baseline.Coverage.TypedFiles, baseline.Coverage.InspectedFiles)
		}
		excepted, err := modernGoExceptionCount(baseline.Exceptions, finding)
		if err != nil {
			return err
		}
		debt := len(finding.Candidates) - excepted
		if debt > ceiling.CandidateCeiling {
			return fmt.Errorf("modern-Go debt increased for %s: %d exceeds ceiling %d after %d excepted candidates",
				finding.ID, debt, ceiling.CandidateCeiling, excepted)
		}
	}
	return nil
}

// AdmitModernGoDelta rejects newly introduced findings in changed source even
// when the aggregate manifest ceiling still has headroom.
func AdmitModernGoDelta(baseline ModernGoBaseline, previous, candidate ModernGoCensus, changedPaths []string) error {
	changed := map[string]bool{}
	for _, name := range changedPaths {
		changed[filepath.ToSlash(filepath.Clean(name))] = true
	}
	previousByID := make(map[string]ModernGoFinding, len(previous.Findings))
	for _, finding := range previous.Findings {
		previousByID[finding.ID] = finding
	}
	for _, finding := range candidate.Findings {
		before := modernGoSiteCounts(previousByID[finding.ID].Candidates)
		after := modernGoSiteCounts(finding.Candidates)
		for key, count := range after {
			path, symbol, _ := strings.Cut(key, "\x00")
			if !changed[path] || count <= before[key] || modernGoExceptionAllows(baseline.Exceptions, finding.ID, path, symbol, count) {
				continue
			}
			return fmt.Errorf("modern-Go debt introduced for %s at %s:%s: %d exceeds HEAD count %d",
				finding.ID, path, symbol, count, before[key])
		}
	}
	return nil
}

func validateModernGoBaseline(baseline ModernGoBaseline, today time.Time) error {
	if baseline.Schema != modernGoSchema || baseline.TargetGo == "" || baseline.CatalogCommit == "" || baseline.CatalogSHA256 == "" || baseline.SourceIdentity == "" {
		return fmt.Errorf("modern-Go baseline has incomplete schema or provenance")
	}
	if baseline.Coverage.Applicable == 0 || baseline.Coverage.Measured != baseline.Coverage.Applicable || baseline.Coverage.InspectedFiles == 0 || baseline.Coverage.TypedFiles == 0 {
		return fmt.Errorf("modern-Go baseline coverage is incomplete: %+v", baseline.Coverage)
	}
	if len(baseline.Guidelines) != baseline.Coverage.Applicable {
		return fmt.Errorf("modern-Go baseline guideline count differs from coverage")
	}
	seen := map[string]bool{}
	for _, guideline := range baseline.Guidelines {
		if guideline.ID == "" || guideline.Risk == "" || guideline.CandidateCeiling < 0 || seen[guideline.ID] {
			return fmt.Errorf("modern-Go baseline has invalid or duplicate guideline %q", guideline.ID)
		}
		seen[guideline.ID] = true
	}
	exceptionKeys := map[string]bool{}
	for _, exception := range baseline.Exceptions {
		if err := validateModernGoException(exception, seen, today); err != nil {
			return err
		}
		key := exception.Guideline + "\x00" + exception.Path + "\x00" + exception.Symbol
		if exceptionKeys[key] {
			return fmt.Errorf("modern-Go baseline repeats exception for %s at %s:%s", exception.Guideline, exception.Path, exception.Symbol)
		}
		exceptionKeys[key] = true
	}
	identity, err := ModernGoExceptionIdentity(baseline.Exceptions)
	if err != nil {
		return err
	}
	if baseline.ExceptionSHA256 != identity {
		return fmt.Errorf("modern-Go exception authority identity differs: have %q want %q", baseline.ExceptionSHA256, identity)
	}
	return nil
}

// ModernGoExceptionIdentity commits the complete ordered exception authority.
func ModernGoExceptionIdentity(exceptions []ModernGoException) (string, error) {
	encoded, err := json.Marshal(exceptions)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

func validateModernGoException(exception ModernGoException, guidelines map[string]bool, today time.Time) error {
	clean := filepath.ToSlash(filepath.Clean(exception.Path))
	tool, remainder, hasSubcommand := strings.Cut(strings.TrimSpace(exception.Oracle), " ")
	subcommand, arguments, hasArguments := strings.Cut(strings.TrimSpace(remainder), " ")
	executableOracle := hasSubcommand && hasArguments && strings.TrimSpace(arguments) != "" &&
		tool == "go" && (subcommand == "test" || subcommand == "run") &&
		!strings.Contains(exception.Oracle, "...") && !strings.ContainsAny(exception.Oracle, "&|;")
	if !guidelines[exception.Guideline] || clean != exception.Path || filepath.IsAbs(exception.Path) ||
		exception.Path == "." || strings.ContainsAny(exception.Path, "*?[") || !strings.HasSuffix(exception.Path, ".go") ||
		exception.Symbol == "" || exception.Owner == "" || exception.Reason == "" || exception.Oracle == "" ||
		exception.Retirement == "" || exception.CandidateCeiling <= 0 || !executableOracle {
		return fmt.Errorf("modern-Go exception is broad or incomplete: %+v", exception)
	}
	expires, err := time.Parse(time.DateOnly, exception.Expires)
	if err != nil {
		return fmt.Errorf("modern-Go exception expiry %q: %w", exception.Expires, err)
	}
	if !today.IsZero() && !expires.After(today) {
		return fmt.Errorf("modern-Go exception expired for %s at %s:%s", exception.Guideline, exception.Path, exception.Symbol)
	}
	return nil
}

func modernGoExceptionCount(exceptions []ModernGoException, finding ModernGoFinding) (int, error) {
	count := 0
	for _, exception := range exceptions {
		if exception.Guideline != finding.ID {
			continue
		}
		matches := 0
		var matchedSites []string
		for _, site := range finding.Candidates {
			if site.Path == exception.Path && site.Symbol == exception.Symbol {
				matches++
				matchedSites = append(matchedSites, fmt.Sprintf("%s:%d", site.Path, site.Line))
			}
		}
		if matches > exception.CandidateCeiling {
			return 0, fmt.Errorf("modern-Go exception expanded for %s at %s:%s: %d exceeds %d; sites=%v",
				exception.Guideline, exception.Path, exception.Symbol, matches, exception.CandidateCeiling, matchedSites)
		}
		count += matches
	}
	return count, nil
}

func modernGoExceptionAllows(exceptions []ModernGoException, guideline, path, symbol string, count int) bool {
	return slices.ContainsFunc(exceptions, func(exception ModernGoException) bool {
		return exception.Guideline == guideline && exception.Path == path && exception.Symbol == symbol && count <= exception.CandidateCeiling
	})
}

func modernGoSiteCounts(sites []ModernGoSite) map[string]int {
	counts := map[string]int{}
	for _, site := range sites {
		counts[site.Path+"\x00"+site.Symbol]++
	}
	return counts
}
