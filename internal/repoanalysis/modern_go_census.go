package repoanalysis

import (
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"path"
	"slices"
	"strings"
)

// ModernGoRisk is the strongest proof required before changing a finding.
type ModernGoRisk string

const (
	// ModernGoRiskMechanical permits an exact behavior-preserving rewrite.
	ModernGoRiskMechanical ModernGoRisk = "mechanical"
	// ModernGoRiskBehavioral requires a contract-specific behavior oracle.
	ModernGoRiskBehavioral ModernGoRisk = "behavioral"
	// ModernGoRiskConcurrency requires lifecycle, ordering, and race evidence.
	ModernGoRiskConcurrency ModernGoRisk = "concurrency"
	// ModernGoRiskWire requires byte-level protocol or document evidence.
	ModernGoRiskWire ModernGoRisk = "wire"
	// ModernGoRiskNumerical requires numerical parity and performance evidence.
	ModernGoRiskNumerical ModernGoRisk = "numerical"
)

// ModernGoSite binds one observed legacy or adopted form to source ownership.
type ModernGoSite struct {
	Path           string `json:"path"`
	Line           int    `json:"line"`
	Package        string `json:"package"`
	Symbol         string `json:"symbol"`
	Build          string `json:"build,omitzero"`
	Kind           string `json:"kind"`
	Test           bool   `json:"test"`
	Generated      bool   `json:"generated"`
	TypeChecked    bool   `json:"type_checked"`
	NumericRuntime bool   `json:"numeric_runtime"`
}

// ModernGoFinding is the complete measured result for one applicable rule.
type ModernGoFinding struct {
	ID             string         `json:"id"`
	SinceVersion   string         `json:"since_version"`
	Risk           ModernGoRisk   `json:"risk"`
	Measured       bool           `json:"measured"`
	InspectedFiles int            `json:"inspected_files"`
	TypedFiles     int            `json:"typed_files"`
	Candidates     []ModernGoSite `json:"candidates,omitempty"`
	Adopted        []ModernGoSite `json:"adopted,omitempty"`
}

// ModernGoCensus is one source-, build-, and catalog-bound measurement.
type ModernGoCensus struct {
	TargetGo       string            `json:"target_go"`
	SourceIdentity string            `json:"source_identity"`
	BuildContext   string            `json:"build_context"`
	CatalogCommit  string            `json:"catalog_commit"`
	CatalogSHA256  string            `json:"catalog_sha256"`
	Findings       []ModernGoFinding `json:"findings"`
}

// CandidateCount returns the complete broad-match denominator.
func (c ModernGoCensus) CandidateCount() int {
	total := 0
	for _, finding := range c.Findings {
		total += len(finding.Candidates)
	}
	return total
}

// AdoptedCount returns the complete adopted-form denominator.
func (c ModernGoCensus) AdoptedCount() int {
	total := 0
	for _, finding := range c.Findings {
		total += len(finding.Adopted)
	}
	return total
}

type modernGoParsedFile struct {
	source    GoFile
	syntax    *ast.File
	fileSet   *token.FileSet
	info      *types.Info
	packageID string
	build     string
	generated bool
	typed     bool
}

// BuildModernGoCensus discovers source and host build ownership through the
// repository's existing authorities, then computes the typed census.
func BuildModernGoCensus(root, targetVersion string) (ModernGoCensus, error) {
	snapshot, err := DiscoverGo(root, "cmd", "internal")
	if err != nil {
		return ModernGoCensus{}, err
	}
	selection, err := HostBuildSelection(root, "./cmd/...", "./internal/...")
	if err != nil {
		return ModernGoCensus{}, err
	}
	return ModernGoCensusSnapshot(snapshot, selection, targetVersion)
}

// ModernGoCensusSnapshot computes a deterministic census over an explicit
// snapshot and toolchain-derived build selection.
func ModernGoCensusSnapshot(snapshot SourceSnapshot, selection BuildSelection, targetVersion string) (ModernGoCensus, error) {
	guidelines, err := ModernGoApplicableGuidelines(targetVersion)
	if err != nil {
		return ModernGoCensus{}, err
	}
	parsed, err := parseModernGoFiles(snapshot, selection)
	if err != nil {
		return ModernGoCensus{}, err
	}
	findings := make([]ModernGoFinding, len(guidelines))
	index := make(map[string]int, len(guidelines))
	for guidelineIndex, guideline := range guidelines {
		findings[guidelineIndex] = ModernGoFinding{
			ID: guideline.ID, SinceVersion: guideline.SinceVersion,
			Risk: modernGoGuidelineRisk(guideline.ID), Measured: true,
		}
		index[guideline.ID] = guidelineIndex
	}
	for _, file := range parsed {
		for findingIndex := range findings {
			findings[findingIndex].InspectedFiles++
			if file.typed {
				findings[findingIndex].TypedFiles++
			}
		}
		measureModernGoFile(file, findings, index)
	}
	for findingIndex := range findings {
		sortModernGoSites(findings[findingIndex].Candidates)
		sortModernGoSites(findings[findingIndex].Adopted)
	}
	return ModernGoCensus{
		TargetGo: targetVersion, SourceIdentity: snapshot.Identity(), BuildContext: selection.Context,
		CatalogCommit: ModernGoCatalogCommit, CatalogSHA256: ModernGoCatalogSHA256, Findings: findings,
	}, nil
}

func parseModernGoFiles(snapshot SourceSnapshot, selection BuildSelection) ([]modernGoParsedFile, error) {
	type group struct {
		fileSet *token.FileSet
		files   []*ast.File
		parsed  []*modernGoParsedFile
	}
	groups := map[string]*group{}
	var unselected []*modernGoParsedFile
	for _, source := range snapshot.Files {
		fileSet := token.NewFileSet()
		syntax, err := parser.ParseFile(fileSet, source.Path, source.blob.data, parser.ParseComments)
		if err != nil {
			return nil, fmt.Errorf("modern-Go parse %s: %w", source.Path, err)
		}
		build, err := source.BuildExpression()
		if err != nil {
			return nil, fmt.Errorf("modern-Go build expression %s: %w", source.Path, err)
		}
		generated, err := source.Generated()
		if err != nil {
			return nil, err
		}
		packageID := selection.Packages[source.Path]
		if packageID == "" {
			packageID = path.Dir(source.Path)
		}
		parsed := &modernGoParsedFile{
			source: source, syntax: syntax, fileSet: fileSet, packageID: packageID,
			build: build, generated: generated,
		}
		if !selection.Files[source.Path] {
			unselected = append(unselected, parsed)
			continue
		}
		key := packageID + "#" + syntax.Name.Name
		owner := groups[key]
		if owner == nil {
			owner = &group{fileSet: token.NewFileSet()}
			groups[key] = owner
		}
		groupSyntax, err := parser.ParseFile(owner.fileSet, source.Path, source.blob.data, parser.ParseComments)
		if err != nil {
			return nil, err
		}
		parsed.syntax, parsed.fileSet = groupSyntax, owner.fileSet
		owner.files = append(owner.files, groupSyntax)
		owner.parsed = append(owner.parsed, parsed)
	}
	var result []modernGoParsedFile
	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	// Package checks share immutable imported types within this census.
	sharedImporter := importer.Default()
	for _, key := range keys {
		owner := groups[key]
		info := &types.Info{
			Types: make(map[ast.Expr]types.TypeAndValue), Defs: make(map[*ast.Ident]types.Object),
			Uses: make(map[*ast.Ident]types.Object), Selections: make(map[*ast.SelectorExpr]*types.Selection),
			Scopes: make(map[ast.Node]*types.Scope),
		}
		config := types.Config{Importer: sharedImporter, Error: func(error) {}}
		packagePath, _ := strings.CutSuffix(key, "#"+owner.files[0].Name.Name)
		_, _ = config.Check(packagePath, owner.fileSet, owner.files, info)
		for _, parsed := range owner.parsed {
			parsed.info, parsed.typed = info, true
			result = append(result, *parsed)
		}
	}
	for _, parsed := range unselected {
		result = append(result, *parsed)
	}
	slices.SortFunc(result, func(left, right modernGoParsedFile) int {
		return strings.Compare(left.source.Path, right.source.Path)
	})
	return result, nil
}

func measureModernGoFile(file modernGoParsedFile, findings []ModernGoFinding, index map[string]int) {
	imports := importBases(file.syntax)
	scan := func(symbol string, root ast.Node) {
		for node := range ast.Preorder(root) {
			for id, findingIndex := range index {
				candidate, adopted := modernGoNodeMatch(id, node, file.info, imports, file.source.Test)
				if !candidate && !adopted {
					continue
				}
				site := modernGoSite(file, node, symbol)
				if candidate {
					site.Kind = "candidate"
					findings[findingIndex].Candidates = append(findings[findingIndex].Candidates, site)
				}
				if adopted {
					site.Kind = "adopted"
					findings[findingIndex].Adopted = append(findings[findingIndex].Adopted, site)
				}
			}
		}
	}
	for _, declaration := range file.syntax.Decls {
		symbol := "package"
		if function, ok := declaration.(*ast.FuncDecl); ok {
			symbol = function.Name.Name
		}
		scan(symbol, declaration)
	}
}

func modernGoSite(file modernGoParsedFile, node ast.Node, symbol string) ModernGoSite {
	return ModernGoSite{
		Path: file.source.Path, Line: file.fileSet.Position(node.Pos()).Line,
		Package: file.packageID, Symbol: symbol, Build: file.build,
		Test: file.source.Test, Generated: file.generated, TypeChecked: file.typed,
		NumericRuntime: modernGoNumericRuntimePath(file.source.Path),
	}
}

func sortModernGoSites(sites []ModernGoSite) {
	slices.SortFunc(sites, func(left, right ModernGoSite) int {
		if left.Path != right.Path {
			return strings.Compare(left.Path, right.Path)
		}
		if left.Line != right.Line {
			return left.Line - right.Line
		}
		if left.Symbol != right.Symbol {
			return strings.Compare(left.Symbol, right.Symbol)
		}
		return strings.Compare(left.Kind, right.Kind)
	})
}

func modernGoGuidelineRisk(id string) ModernGoRisk {
	switch id {
	case "sync_waitgroup_go", "sync_once_func", "sync_once_value", "context_after_func",
		"context_timeout_deadline_cause", "context_cancel_cause", "atomic_types":
		return ModernGoRiskConcurrency
	case "json_omitzero", "http_servemux_patterns", "new_expression":
		return ModernGoRiskWire
	case "min_max", "slices_max_min", "range_over_int":
		return ModernGoRiskNumerical
	case "errors_as_type", "errors_join", "errors_is", "time_tick_gc":
		return ModernGoRiskBehavioral
	default:
		return ModernGoRiskMechanical
	}
}
