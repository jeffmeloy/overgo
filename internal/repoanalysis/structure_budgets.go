package repoanalysis

import (
	"bytes"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/jsonfile"
)

// StructureBudgetsFile is the reviewed size-and-coupling budget manifest.
const StructureBudgetsFile = "docs/structure_budgets.json"

// StructureBudgetException records one reviewed exceedance: the exact
// subject and the reason it may stand above the shared budget.
type StructureBudgetException struct {
	Budget  string `json:"budget"`
	Subject string `json:"subject"`
	Limit   int    `json:"limit"`
	Reason  string `json:"reason"`
}

// StructureBudgets is the enforceable ceiling set: maximum production file
// size, maximum direct internal imports per package, maximum production
// files per command package, and the advisory staged-path bound per
// ordinary gate commit. Ceilings only tighten; standing exceedances live
// as reviewed exceptions with their reasons.
type StructureBudgets struct {
	Version                  int                        `json:"version"`
	Doc                      string                     `json:"doc"`
	MaxProductionFileLines   int                        `json:"max_production_file_lines"`
	MaxDirectInternalImports int                        `json:"max_direct_internal_imports"`
	MaxCommandPackageFiles   int                        `json:"max_command_package_files"`
	MaxStagedPathsPerCommit  int                        `json:"max_staged_paths_per_commit"`
	Exceptions               []StructureBudgetException `json:"exceptions,omitempty"`
}

// LoadStructureBudgets strictly decodes the budget manifest.
func LoadStructureBudgets(path string) (StructureBudgets, error) {
	var budgets StructureBudgets
	if err := jsonfile.DecodeStrict(path, &budgets); err != nil {
		return StructureBudgets{}, err
	}
	if budgets.MaxProductionFileLines <= 0 || budgets.MaxDirectInternalImports <= 0 ||
		budgets.MaxCommandPackageFiles <= 0 || budgets.MaxStagedPathsPerCommit <= 0 {
		return StructureBudgets{}, fmt.Errorf("structure budgets: every ceiling must be positive")
	}
	return budgets, nil
}

// StructureFinding names one measured exceedance of a shared budget.
type StructureFinding struct {
	Budget  string
	Subject string
	Value   int
	Limit   int
}

// MeasureStructureBudgets audits the snapshot's production surface against
// the budgets: per-file line counts, per-package direct internal imports,
// and per-command-package production file counts. Reviewed exceptions carry
// their own limits; everything else holds the shared ceiling.
func MeasureStructureBudgets(snapshot SourceSnapshot, budgets StructureBudgets) []StructureFinding {
	exceptionLimit := func(budget, subject string) int {
		for _, exception := range budgets.Exceptions {
			if exception.Budget == budget && exception.Subject == subject {
				return exception.Limit
			}
		}
		return 0
	}
	var findings []StructureFinding
	packageImports := map[string]map[string]bool{}
	commandFiles := map[string]int{}
	for _, file := range snapshot.Files {
		if file.Test {
			continue
		}
		if generated, err := file.Generated(); err != nil || generated {
			continue
		}
		lines := bytes.Count(file.blob.data, []byte{'\n'}) + 1
		limit := max(budgets.MaxProductionFileLines, exceptionLimit("file-lines", file.Path))
		if lines > limit {
			findings = append(findings, StructureFinding{
				Budget: "file-lines", Subject: file.Path, Value: lines, Limit: limit,
			})
		}
		packagePath := packageOf(file.Path)
		if strings.HasPrefix(packagePath, "cmd/") {
			commandFiles[packagePath]++
		}
		syntax, err := file.Syntax()
		if err != nil {
			continue
		}
		for _, imported := range syntax.Imports {
			path := strings.Trim(imported.Path.Value, `"`)
			if !strings.HasPrefix(path, "overgo/internal/") {
				continue
			}
			if packageImports[packagePath] == nil {
				packageImports[packagePath] = map[string]bool{}
			}
			packageImports[packagePath][path] = true
		}
	}
	for packagePath, imports := range packageImports {
		limit := max(budgets.MaxDirectInternalImports, exceptionLimit("internal-imports", packagePath))
		if len(imports) > limit {
			findings = append(findings, StructureFinding{
				Budget: "internal-imports", Subject: packagePath, Value: len(imports), Limit: limit,
			})
		}
	}
	for packagePath, count := range commandFiles {
		limit := max(budgets.MaxCommandPackageFiles, exceptionLimit("command-files", packagePath))
		if count > limit {
			findings = append(findings, StructureFinding{
				Budget: "command-files", Subject: packagePath, Value: count, Limit: limit,
			})
		}
	}
	slices.SortFunc(findings, func(left, right StructureFinding) int {
		if order := strings.Compare(left.Budget, right.Budget); order != 0 {
			return order
		}
		return strings.Compare(left.Subject, right.Subject)
	})
	return findings
}

func packageOf(path string) string {
	slash := strings.LastIndexByte(path, '/')
	if slash < 0 {
		return path
	}
	return path[:slash]
}
