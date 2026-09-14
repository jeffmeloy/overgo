package repoanalysis

import (
	"go/ast"
	"go/token"
	"regexp"
	"slices"
	"strings"
)

// FamilyBranch is one family-named branch remaining in shared execution
// code: a comparison or switch case whose string literal names a model
// family instead of consulting a declared profile policy.
type FamilyBranch struct {
	File    string `json:"file"`
	Literal string `json:"literal"`
	Count   int    `json:"count"`
}

// familyStemPattern matches family-named string literals: a known family
// stem optionally followed by a version suffix.
var familyStemPattern = regexp.MustCompile(`^(llama|qwen|gemma|phi|glm|deepseek|mistral|mixtral|bert|t5|gpt2|granite|olmo|minicpm|starcoder|falcon|sensenova|wan)[0-9.]*$`)

// familyBranchExcluded names the file classes the census deliberately leaves
// out: per-family catalogs and converters (family facts belong there),
// generated code, fixtures, and presentation.
func familyBranchExcluded(path string) bool {
	for _, prefix := range []string{
		"internal/hfgguf/", "internal/hfconvert/", "internal/gemma4convert/",
		"internal/server/webui/", "internal/testutil/",
	} {
		if strings.HasPrefix(path, prefix) {
			return true
		}
	}
	return strings.HasSuffix(path, "_test.go") || strings.Contains(path, "/testdata/") ||
		path == "internal/model/architecture_catalog.go"
}

// FamilyBranchCensus finds family-named comparisons in the shared snapshot.
func FamilyBranchCensus(snapshot SourceSnapshot) ([]FamilyBranch, error) {
	counts := map[[2]string]int{}
	for _, file := range snapshot.Files {
		relative := file.Path
		if familyBranchExcluded(relative) {
			continue
		}
		parsed, err := file.Syntax()
		if err != nil {
			return nil, err
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			switch typed := node.(type) {
			case *ast.BinaryExpr:
				if typed.Op == token.EQL || typed.Op == token.NEQ {
					for _, side := range []ast.Expr{typed.X, typed.Y} {
						if literal := familyLiteral(side); literal != "" {
							counts[[2]string{relative, literal}]++
						}
					}
				}
			case *ast.CaseClause:
				for _, value := range typed.List {
					if literal := familyLiteral(value); literal != "" {
						counts[[2]string{relative, literal}]++
					}
				}
			}
			return true
		})
	}
	branches := make([]FamilyBranch, 0, len(counts))
	for key, count := range counts {
		branches = append(branches, FamilyBranch{File: key[0], Literal: key[1], Count: count})
	}
	slices.SortFunc(branches, func(left, right FamilyBranch) int {
		if order := strings.Compare(left.File, right.File); order != 0 {
			return order
		}
		return strings.Compare(left.Literal, right.Literal)
	})
	return branches, nil
}

func familyLiteral(expression ast.Expr) string {
	literal, ok := expression.(*ast.BasicLit)
	if !ok || literal.Kind != token.STRING {
		return ""
	}
	value := strings.Trim(literal.Value, `"`)
	if familyStemPattern.MatchString(value) {
		return value
	}
	return ""
}
