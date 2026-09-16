package repoanalysis

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"overgo/internal/gosource"
)

// ModernGoWorkRequest is the plan-owned rule slice and maximum number of
// source sites to return. The caller chooses rules; the census only derives
// facts and therefore cannot become a competing work dispatcher.
type ModernGoWorkRequest struct {
	Guidelines []string `json:"guidelines"`
	Limit      int      `json:"limit"`
}

// ModernGoWorkCoverage reports every denominator behind a bounded selection.
type ModernGoWorkCoverage struct {
	Inspected  int `json:"inspected"`
	Matched    int `json:"matched"`
	Selected   int `json:"selected"`
	Excluded   int `json:"excluded"`
	Unmeasured int `json:"unmeasured"`
}

// ModernGoWorkFinding is one risk-classified rule and its selected sites.
type ModernGoWorkFinding struct {
	ID    string         `json:"id"`
	Risk  ModernGoRisk   `json:"risk"`
	Sites []ModernGoSite `json:"sites"`
}

// ModernGoVerification is the exact package closure and command input derived
// from repository package metadata for the selected source owners.
type ModernGoVerification struct {
	DirectPackages    []string `json:"direct_packages"`
	DependentPackages []string `json:"dependent_packages"`
	DirectArgs        []string `json:"direct_args"`
	ClosureArgs       []string `json:"closure_args"`
}

// ModernGoWorkSelection binds a bounded work slice and its verification to the
// source, build context, and catalog that produced it.
type ModernGoWorkSelection struct {
	TargetGo       string                `json:"target_go"`
	SourceIdentity string                `json:"source_identity"`
	BuildContext   string                `json:"build_context"`
	CatalogCommit  string                `json:"catalog_commit"`
	CatalogSHA256  string                `json:"catalog_sha256"`
	Request        ModernGoWorkRequest   `json:"request"`
	Coverage       ModernGoWorkCoverage  `json:"coverage"`
	Findings       []ModernGoWorkFinding `json:"findings"`
	Verification   ModernGoVerification  `json:"verification"`
}

// BuildModernGoWorkSelection measures the repository and derives a bounded,
// deterministic selection plus the exact repository package test closure.
func BuildModernGoWorkSelection(root, targetVersion string, request ModernGoWorkRequest) (ModernGoWorkSelection, error) {
	if request.Limit <= 0 {
		return ModernGoWorkSelection{}, fmt.Errorf("modern-Go work limit must be positive")
	}
	request.Guidelines = normalizeModernGoRuleIDs(request.Guidelines)
	if len(request.Guidelines) == 0 {
		return ModernGoWorkSelection{}, fmt.Errorf("modern-Go work selection requires plan-owned guideline IDs")
	}
	census, err := BuildModernGoCensus(root, targetVersion)
	if err != nil {
		return ModernGoWorkSelection{}, err
	}
	packages, err := gosource.HostBuildSelection(root, "./...")
	if err != nil {
		return ModernGoWorkSelection{}, err
	}
	return modernGoSelectWork(census, packages.Dependencies, request)
}

func modernGoSelectWork(census ModernGoCensus, packages map[string][]string, request ModernGoWorkRequest) (ModernGoWorkSelection, error) {
	requested := make(map[string]bool, len(request.Guidelines))
	for _, id := range request.Guidelines {
		requested[id] = true
	}
	known := make(map[string]bool, len(census.Findings))
	coverage := ModernGoWorkCoverage{}
	for _, finding := range census.Findings {
		known[finding.ID] = true
		coverage.Inspected = max(coverage.Inspected, finding.InspectedFiles)
		coverage.Matched += len(finding.Candidates)
		if !finding.Measured {
			coverage.Unmeasured++
		}
	}
	for _, id := range request.Guidelines {
		if !known[id] {
			return ModernGoWorkSelection{}, fmt.Errorf("modern-Go guideline %q is not measured by this census", id)
		}
	}

	selection := ModernGoWorkSelection{
		TargetGo: census.TargetGo, SourceIdentity: census.SourceIdentity,
		BuildContext: census.BuildContext, CatalogCommit: census.CatalogCommit,
		CatalogSHA256: census.CatalogSHA256, Request: request, Coverage: coverage,
	}
	remaining := request.Limit
	direct := map[string]bool{}
	for _, finding := range census.Findings {
		if !requested[finding.ID] {
			continue
		}
		if !finding.Measured {
			return ModernGoWorkSelection{}, fmt.Errorf("modern-Go guideline %q has no complete measurement", finding.ID)
		}
		for _, site := range finding.Candidates {
			if err := validateModernGoSite(finding.ID, site); err != nil {
				return ModernGoWorkSelection{}, err
			}
		}
		count := min(remaining, len(finding.Candidates))
		if count == 0 {
			continue
		}
		sites := slices.Clone(finding.Candidates[:count])
		selection.Findings = append(selection.Findings, ModernGoWorkFinding{
			ID: finding.ID, Risk: finding.Risk, Sites: sites,
		})
		for _, site := range sites {
			direct[site.Package] = true
		}
		remaining -= count
	}
	selection.Coverage.Selected = request.Limit - remaining
	selection.Coverage.Excluded = selection.Coverage.Matched - selection.Coverage.Selected
	selection.Verification = deriveModernGoVerification(packages, direct)
	return selection, nil
}

func normalizeModernGoRuleIDs(ids []string) []string {
	seen := map[string]bool{}
	result := make([]string, 0, len(ids))
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		result = append(result, id)
	}
	slices.Sort(result)
	return result
}

func validateModernGoSite(id string, site ModernGoSite) error {
	if site.Path == "" || site.Line <= 0 || site.Package == "" || site.Symbol == "" {
		return fmt.Errorf("modern-Go guideline %s has incomplete source ownership: %+v", id, site)
	}
	return nil
}

func deriveModernGoVerification(packages map[string][]string, directSet map[string]bool) ModernGoVerification {
	direct := slices.Sorted(maps.Keys(directSet))
	dependentSet := map[string]bool{}
	for path, dependencies := range packages {
		if directSet[path] {
			continue
		}
		if slices.ContainsFunc(dependencies, func(dependency string) bool { return directSet[dependency] }) {
			dependentSet[path] = true
		}
	}
	dependent := slices.Sorted(maps.Keys(dependentSet))
	closure := append(slices.Clone(direct), dependent...)
	slices.Sort(closure)
	verification := ModernGoVerification{DirectPackages: direct, DependentPackages: dependent}
	if len(direct) != 0 {
		verification.DirectArgs = append([]string{"test"}, direct...)
		verification.ClosureArgs = append([]string{"test"}, closure...)
	}
	return verification
}
