package closurescan

import (
	"path/filepath"
	"sort"

	"overgo/internal/repoanalysis"
)

const CensusSchema = "overgo/magic-census/v2"

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

type FilePressure struct {
	File             string `json:"file"`
	Package          string `json:"package"`
	Test             bool   `json:"test"`
	DecisionSurfaces int    `json:"decision_surfaces"`
	NamedConstants   int    `json:"named_constants"`
	InlineLiterals   int    `json:"inline_literals"`
	AssumptionHints  int    `json:"assumption_hints"`
	TestLiterals     int    `json:"test_literals"`
	TestFixtures     int    `json:"test_fixtures"`
	TestAssertions   int    `json:"test_assertions"`
	TestPolicyCopies int    `json:"test_policy_copies"`
	RepeatedGroups   int    `json:"repeated_groups"`
}

type Census struct {
	Schema   string             `json:"schema"`
	Source   string             `json:"source"`
	Counts   CensusCounts       `json:"counts"`
	Owners   []OwnerPressure    `json:"owners"`
	Files    []FilePressure     `json:"files"`
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
	testSites, err := CensusTestLiterals(snapshot)
	if err != nil {
		return Census{}, err
	}

	result := Census{
		Schema: CensusSchema, Source: snapshot.Identity(), Repeated: repeated,
		Counts: CensusCounts{NamedConstants: len(named), InlineLiterals: len(inline),
			AssumptionHints: len(assumptions), RepeatedGroups: len(repeated)},
	}
	owners := map[string]*OwnerPressure{}
	files := map[string]*FilePressure{}
	owner := func(pkg string) *OwnerPressure {
		if owners[pkg] == nil {
			owners[pkg] = &OwnerPressure{Package: pkg}
		}
		return owners[pkg]
	}
	for _, source := range snapshot.Files {
		generated, err := source.Generated()
		if err != nil {
			return Census{}, err
		}
		if generated {
			continue
		}
		files[source.Path] = &FilePressure{File: source.Path, Package: packagePath(source.Path), Test: source.Test}
		if source.Test {
			result.Counts.TestFiles++
		} else {
			result.Counts.ProductionFiles++
		}
	}
	for _, candidate := range named {
		owner(candidate.Package).NamedConstants++
		files[candidate.File].NamedConstants++
	}
	for _, site := range inline {
		owner(site.Package).InlineLiterals++
		files[site.File].InlineLiterals++
	}
	for _, hint := range assumptions {
		owner(hint.Package).AssumptionHints++
		files[hint.File].AssumptionHints++
	}
	for _, site := range testSites {
		entry := files[site.File]
		entry.TestLiterals++
		result.Counts.TestLiterals++
		switch site.Class {
		case TestFixture:
			entry.TestFixtures++
			result.Counts.TestFixtures++
		case TestAssertion:
			entry.TestAssertions++
			result.Counts.TestAssertions++
		case TestPolicyCopy:
			entry.TestPolicyCopies++
			result.Counts.TestPolicyCopies++
			owner(site.Package).TestPolicyCopies++
		}
	}
	for _, group := range repeated {
		entry := owner(group.Package)
		entry.RepeatedGroups++
		entry.RepeatedSites += group.Count
		result.Counts.RepeatedSites += group.Count
		for _, file := range group.Files {
			files[file].RepeatedGroups++
		}
	}
	for _, entry := range files {
		entry.DecisionSurfaces = entry.NamedConstants + entry.InlineLiterals + entry.AssumptionHints + entry.TestPolicyCopies
		result.Files = append(result.Files, *entry)
	}
	sort.Slice(result.Files, func(i, j int) bool {
		left, right := result.Files[i], result.Files[j]
		if left.DecisionSurfaces != right.DecisionSurfaces {
			return left.DecisionSurfaces > right.DecisionSurfaces
		}
		if left.RepeatedGroups != right.RepeatedGroups {
			return left.RepeatedGroups > right.RepeatedGroups
		}
		return left.File < right.File
	})
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

func packagePath(file string) string {
	packageName := filepath.ToSlash(filepath.Dir(file))
	if packageName == "." {
		return ""
	}
	return packageName
}
