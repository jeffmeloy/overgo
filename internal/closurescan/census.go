package closurescan

import (
	"sort"

	"overgo/internal/repoanalysis"
)

const CensusSchema = "overgo/magic-census/v1"

type CensusCounts struct {
	ProductionFiles  int `json:"production_files"`
	TestFiles        int `json:"test_files"`
	NamedConstants   int `json:"named_constants"`
	InlineLiterals   int `json:"inline_literals"`
	AssumptionHints  int `json:"assumption_hints"`
	TestLiterals     int `json:"test_literals"`
	TestFixtures     int `json:"test_fixtures"`
	TestAssertions   int `json:"test_assertions"`
	TestPolicyCopies int `json:"test_policy_copies"`
	RepeatedGroups   int `json:"repeated_groups"`
	RepeatedSites    int `json:"repeated_sites"`
}

type OwnerPressure struct {
	Package          string `json:"package"`
	DecisionSurfaces int    `json:"decision_surfaces"`
	NamedConstants   int    `json:"named_constants"`
	InlineLiterals   int    `json:"inline_literals"`
	AssumptionHints  int    `json:"assumption_hints"`
	TestPolicyCopies int    `json:"test_policy_copies"`
	RepeatedGroups   int    `json:"repeated_groups"`
	RepeatedSites    int    `json:"repeated_sites"`
}

type Census struct {
	Schema   string             `json:"schema"`
	Source   string             `json:"source"`
	Counts   CensusCounts       `json:"counts"`
	Owners   []OwnerPressure    `json:"owners"`
	Repeated []RawPolicyLiteral `json:"repeated"`
}

// BuildCensus measures source pressure; findings grant no disposition.
func BuildCensus(snapshot repoanalysis.SourceSnapshot) (Census, error) {
	named, err := ScanSnapshot(snapshot, nil)
	if err != nil {
		return Census{}, err
	}
	inline, err := CensusLiterals(snapshot, nil)
	if err != nil {
		return Census{}, err
	}
	assumptions, err := CensusAssumptions(snapshot, nil)
	if err != nil {
		return Census{}, err
	}
	repeated, err := RepeatedPolicyLiterals(snapshot)
	if err != nil {
		return Census{}, err
	}
	tests, testOwners, err := summarizeTestLiterals(snapshot)
	if err != nil {
		return Census{}, err
	}

	result := Census{
		Schema: CensusSchema, Source: snapshot.Identity(), Repeated: repeated,
		Counts: CensusCounts{
			NamedConstants: len(named), InlineLiterals: len(inline), AssumptionHints: len(assumptions),
			TestLiterals: tests.Total, TestFixtures: tests.Fixtures,
			TestAssertions: tests.Assertions, TestPolicyCopies: tests.PolicyCopies,
			RepeatedGroups: len(repeated),
		},
	}
	owners := map[string]*OwnerPressure{}
	owner := func(pkg string) *OwnerPressure {
		if owners[pkg] == nil {
			owners[pkg] = &OwnerPressure{Package: pkg}
		}
		return owners[pkg]
	}
	for _, candidate := range named {
		owner(candidate.Package).NamedConstants++
	}
	for _, site := range inline {
		owner(site.Package).InlineLiterals++
	}
	for _, hint := range assumptions {
		owner(hint.Package).AssumptionHints++
	}
	for pkg, count := range testOwners {
		owner(pkg).TestPolicyCopies += count
	}
	for _, group := range repeated {
		entry := owner(group.Package)
		entry.RepeatedGroups++
		entry.RepeatedSites += group.Count
		result.Counts.RepeatedSites += group.Count
	}
	for _, source := range snapshot.Files {
		generated, err := source.Generated()
		if err != nil {
			return Census{}, err
		}
		if generated {
			continue
		}
		if source.Test {
			result.Counts.TestFiles++
		} else {
			result.Counts.ProductionFiles++
		}
	}
	for _, entry := range owners {
		entry.DecisionSurfaces = entry.NamedConstants + entry.InlineLiterals + entry.AssumptionHints + entry.TestPolicyCopies
		result.Owners = append(result.Owners, *entry)
	}
	sort.Slice(result.Owners, func(i, j int) bool {
		left, right := result.Owners[i], result.Owners[j]
		if left.DecisionSurfaces != right.DecisionSurfaces {
			return left.DecisionSurfaces > right.DecisionSurfaces
		}
		if left.RepeatedSites != right.RepeatedSites {
			return left.RepeatedSites > right.RepeatedSites
		}
		return left.Package < right.Package
	})
	return result, nil
}

type testLiteralCounts struct {
	Total, Fixtures, Assertions, PolicyCopies int
}

func summarizeTestLiterals(snapshot repoanalysis.SourceSnapshot) (testLiteralCounts, map[string]int, error) {
	var counts testLiteralCounts
	owners := map[string]int{}
	err := visitTestLiterals(snapshot, func(site TestLiteralSite) {
		counts.Total++
		switch site.Class {
		case TestFixture:
			counts.Fixtures++
		case TestAssertion:
			counts.Assertions++
		case TestPolicyCopy:
			counts.PolicyCopies++
			owners[site.Package]++
		}
	})
	return counts, owners, err
}
