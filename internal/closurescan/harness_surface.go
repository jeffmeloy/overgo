package closurescan

import (
	"errors"
	"go/ast"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/codeprofile"
	"overgo/internal/repoanalysis"
)

// HarnessLayerViolation records one import pointing from a lower harness
// layer to a higher one.
type HarnessLayerViolation struct {
	Importer string `json:"importer"`
	Imported string `json:"imported"`
}

// HarnessSurface is the measured production complexity footprint of the
// agent-harness packages.
type HarnessSurface struct {
	Packages             int                     `json:"packages"`
	ProductionFiles      int                     `json:"production_files"`
	ProductionNodes      int                     `json:"production_nodes"`
	MaxFileNodes         int                     `json:"max_file_nodes"`
	MaxFunctionNodes     int                     `json:"max_function_nodes"`
	GuardedScalarFields  int                     `json:"guarded_scalar_fields"`
	RepeatedPolicyGroups int                     `json:"repeated_policy_groups"`
	RepeatedPolicySites  int                     `json:"repeated_policy_sites"`
	LayerViolations      []HarnessLayerViolation `json:"layer_violations"`
}

// HarnessSurfaceRegression names one surface metric that grew past its
// baseline, or one new layer violation.
type HarnessSurfaceRegression struct {
	Metric string `json:"metric"`
	Base   int    `json:"base"`
	Value  int    `json:"value"`
	Detail string `json:"detail,omitempty"`
}

type harnessLayer uint8

const (
	harnessIntent harnessLayer = iota
	harnessAuthority
	harnessLifecycle
	harnessExecution
	harnessComposition
)

var harnessLayers = map[string]harnessLayer{
	"internal/recipe":          harnessIntent,
	"internal/agenttool":       harnessAuthority,
	"internal/runrecord":       harnessAuthority,
	"internal/plan":            harnessAuthority,
	"internal/operatoraction":  harnessAuthority,
	"internal/operation":       harnessLifecycle,
	"internal/workflowruntime": harnessExecution,
	"internal/agentworkflow":   harnessExecution,
	"internal/agentloop":       harnessComposition,
}

// BuildAgentHarnessSurface measures the harness-owned packages: node counts,
// guarded scalar fields, repeated policy literals, and layer violations.
func BuildAgentHarnessSurface(snapshot repoanalysis.SourceSnapshot) (HarnessSurface, error) {
	profile, err := codeprofile.Build(snapshot)
	if err != nil {
		return HarnessSurface{}, err
	}
	repeated, err := RepeatedPolicyLiterals(snapshot)
	if err != nil {
		return HarnessSurface{}, err
	}
	result := HarnessSurface{}
	packages := map[string]bool{}
	for _, source := range snapshot.Files {
		pkg := packagePath(source.Path)
		layer, owned := harnessLayers[pkg]
		if !owned || source.Test {
			continue
		}
		generated, generatedErr := source.Generated()
		if generatedErr != nil {
			return HarnessSurface{}, generatedErr
		}
		if generated {
			continue
		}
		parsed, parseErr := source.Syntax()
		if parseErr != nil {
			return HarnessSurface{}, parseErr
		}
		packages[pkg] = true
		result.ProductionFiles++
		nodes := codeprofile.NodeCount(parsed)
		result.ProductionNodes += nodes
		result.MaxFileNodes = max(result.MaxFileNodes, nodes)
		result.GuardedScalarFields += guardedScalarFields(parsed)
		for _, imported := range parsed.Imports {
			path, unquoteErr := strconv.Unquote(imported.Path.Value)
			if unquoteErr != nil {
				return HarnessSurface{}, unquoteErr
			}
			importedPkg := internalPackage(path)
			importedLayer, harnessImport := harnessLayers[importedPkg]
			if harnessImport && importedLayer > layer {
				result.LayerViolations = append(result.LayerViolations, HarnessLayerViolation{
					Importer: pkg, Imported: importedPkg,
				})
			}
		}
	}
	result.Packages = len(packages)
	for _, function := range profile.Functions {
		if _, owned := harnessLayers[packagePath(function.File)]; owned && !strings.HasSuffix(function.File, "_test.go") {
			result.MaxFunctionNodes = max(result.MaxFunctionNodes, function.Nodes)
		}
	}
	for _, group := range repeated {
		if _, owned := harnessLayers[group.Package]; !owned {
			continue
		}
		result.RepeatedPolicyGroups++
		result.RepeatedPolicySites += group.Count
	}
	sort.Slice(result.LayerViolations, func(left, right int) bool {
		l, r := result.LayerViolations[left], result.LayerViolations[right]
		return l.Importer < r.Importer || l.Importer == r.Importer && l.Imported < r.Imported
	})
	result.LayerViolations = slices.Compact(result.LayerViolations)
	return result, nil
}

// AgentHarnessSurfaceRegressions compares candidate against base and returns
// every grown metric and every new layer violation.
func AgentHarnessSurfaceRegressions(base, candidate HarnessSurface) []HarnessSurfaceRegression {
	metrics := []struct {
		name        string
		base, value int
	}{
		{"production_files", base.ProductionFiles, candidate.ProductionFiles},
		{"production_nodes", base.ProductionNodes, candidate.ProductionNodes},
		{"max_file_nodes", base.MaxFileNodes, candidate.MaxFileNodes},
		{"max_function_nodes", base.MaxFunctionNodes, candidate.MaxFunctionNodes},
		{"guarded_scalar_fields", base.GuardedScalarFields, candidate.GuardedScalarFields},
		{"repeated_policy_groups", base.RepeatedPolicyGroups, candidate.RepeatedPolicyGroups},
		{"repeated_policy_sites", base.RepeatedPolicySites, candidate.RepeatedPolicySites},
	}
	var regressions []HarnessSurfaceRegression
	for _, metric := range metrics {
		if metric.value > metric.base {
			regressions = append(regressions, HarnessSurfaceRegression{
				Metric: metric.name, Base: metric.base, Value: metric.value,
			})
		}
	}
	known := make(map[HarnessLayerViolation]bool, len(base.LayerViolations))
	for _, violation := range base.LayerViolations {
		known[violation] = true
	}
	for _, violation := range candidate.LayerViolations {
		if known[violation] {
			continue
		}
		regressions = append(regressions, HarnessSurfaceRegression{
			Metric: "layer_violation", Detail: violation.Importer + " -> " + violation.Imported,
		})
	}
	return regressions
}

func validateHarnessSurface(value HarnessSurface) error {
	counts := []int{
		value.Packages, value.ProductionFiles, value.ProductionNodes, value.MaxFileNodes,
		value.MaxFunctionNodes, value.GuardedScalarFields, value.RepeatedPolicyGroups, value.RepeatedPolicySites,
	}
	if slices.ContainsFunc(counts, func(count int) bool { return count < 0 }) {
		return errors.New("closure scan: invalid agent harness surface")
	}
	for index, violation := range value.LayerViolations {
		if _, ok := harnessLayers[violation.Importer]; !ok {
			return errors.New("closure scan: invalid agent harness layer importer")
		}
		if _, ok := harnessLayers[violation.Imported]; !ok {
			return errors.New("closure scan: invalid agent harness layer import")
		}
		if index > 0 {
			previous := value.LayerViolations[index-1]
			if previous.Importer > violation.Importer || previous.Importer == violation.Importer && previous.Imported >= violation.Imported {
				return errors.New("closure scan: unordered agent harness layer violations")
			}
		}
	}
	return nil
}

func internalPackage(importPath string) string {
	const marker = "/internal/"
	index := strings.Index(importPath, marker)
	if index < 0 {
		return filepath.ToSlash(importPath)
	}
	return "internal/" + strings.TrimPrefix(filepath.ToSlash(importPath[index+len(marker):]), "/")
}

func guardedScalarFields(file *ast.File) int {
	total := 0
	ast.Inspect(file, func(node ast.Node) bool {
		structure, ok := node.(*ast.StructType)
		if !ok || structure.Fields == nil || !guardedStructure(structure) {
			return true
		}
		for _, field := range structure.Fields.List {
			if scalarField(field.Type) {
				fieldCount := len(field.Names)
				if fieldCount == 0 {
					fieldCount++
				}
				total += fieldCount
			}
		}
		return false
	})
	return total
}

func guardedStructure(structure *ast.StructType) bool {
	return slices.ContainsFunc(structure.Fields.List, func(field *ast.Field) bool {
		selector, ok := field.Type.(*ast.SelectorExpr)
		qualifier, qualified := selectorQualifier(selector)
		return ok && qualified && (qualifier == "sync" && (selector.Sel.Name == "Mutex" || selector.Sel.Name == "RWMutex") ||
			qualifier == "atomic")
	})
}

func selectorQualifier(selector *ast.SelectorExpr) (string, bool) {
	if selector == nil {
		return "", false
	}
	identifier, ok := selector.X.(*ast.Ident)
	if !ok {
		return "", false
	}
	return identifier.Name, true
}

func scalarField(expression ast.Expr) bool {
	identifier, ok := expression.(*ast.Ident)
	if !ok {
		return false
	}
	switch identifier.Name {
	case "bool", "byte", "rune", "string", "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", "uintptr", "float32", "float64", "complex64", "complex128":
		return true
	default:
		return false
	}
}
