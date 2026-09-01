package model

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
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

// FamilyBranchCensus walks the production tree under root and enumerates
// every family-named branch in shared execution code: string literals
// matching a family stem inside equality comparisons or switch cases. A new
// model expressible through existing primitives must introduce none; the
// remaining set is pinned by its reviewed baseline.
func FamilyBranchCensus(root string) ([]FamilyBranch, error) {
	counts := map[[2]string]int{}
	for _, top := range []string{"internal", "cmd"} {
		err := filepath.WalkDir(filepath.Join(root, top), func(path string, entry fs.DirEntry, walkErr error) error {
			if walkErr != nil || entry.IsDir() || !strings.HasSuffix(path, ".go") {
				return walkErr
			}
			relative, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			relative = filepath.ToSlash(relative)
			if familyBranchExcluded(relative) {
				return nil
			}
			parsed, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
			if err != nil {
				return err
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
			return nil
		})
		if err != nil {
			return nil, err
		}
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
