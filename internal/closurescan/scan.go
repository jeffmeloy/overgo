// Package closurescan owns the constant-scanning core shared by the
// closure-scan CLI and the gate's magic step: parse production Go, collect
// numeric constants (iota enumerations, tests, testdata, and generated code
// excluded -- enumerations are not magics), and rank them for triage. The
// score orders reports and admits nothing; judgment happens at triage.
package closurescan

import (
	"go/ast"
	"go/token"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/repoanalysis"
)

type Candidate struct {
	Name    string `json:"name"`
	File    string `json:"file"`
	Value   string `json:"value"`
	Doc     string `json:"doc,omitempty"`
	Score   int    `json:"score"`
	Package string `json:"package"`
}

// ScanRoot walks internal/ and cmd/ under root.
func ScanRoot(root string) ([]Candidate, error) {
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		return nil, err
	}
	return ScanSnapshot(snapshot, nil)
}

// ScanFiles scans exactly the given root-relative production files; non-Go,
// test, and missing files are skipped so callers can pass a commit's path
// set unfiltered.
func ScanFiles(root string, relatives []string) ([]Candidate, error) {
	snapshot, err := repoanalysis.LoadGo(root, relatives)
	if err != nil {
		return nil, err
	}
	return ScanSnapshot(snapshot, nil)
}

// ScanSnapshot reuses parsed source and optionally limits findings to paths.
func ScanSnapshot(snapshot repoanalysis.SourceSnapshot, relatives []string) ([]Candidate, error) {
	var out []Candidate
	wanted := map[string]bool{}
	for _, relative := range relatives {
		wanted[filepath.ToSlash(relative)] = true
	}
	for _, source := range snapshot.Files {
		if source.Test || len(wanted) > 0 && !wanted[source.Path] {
			continue
		}
		generated, err := source.Generated()
		if err != nil {
			return nil, err
		}
		if generated {
			continue
		}
		parsed, err := source.Syntax()
		if err != nil {
			return nil, err
		}
		collect(parsed, source.Path, &out)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Name < out[j].Name
	})
	return out, nil
}

func collect(file *ast.File, relative string, out *[]Candidate) {
	for _, declaration := range file.Decls {
		generic, ok := declaration.(*ast.GenDecl)
		if !ok || generic.Tok != token.CONST {
			continue
		}
		blockDoc := ""
		if generic.Doc != nil {
			blockDoc = strings.TrimSpace(generic.Doc.Text())
		}
		for _, spec := range generic.Specs {
			value, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			if usesIota(value) {
				continue
			}
			doc := blockDoc
			if value.Doc != nil {
				doc = strings.TrimSpace(value.Doc.Text())
			}
			for index, name := range value.Names {
				if name.Name == "_" || index >= len(value.Values) {
					continue
				}
				literal := numericLiteral(value.Values[index])
				if literal == "" {
					continue
				}
				*out = append(*out, Candidate{
					Name: name.Name, File: relative, Value: literal,
					Doc: doc, Score: score(name.Name, doc, literal),
					Package: filepath.ToSlash(filepath.Dir(relative)),
				})
			}
		}
	}
}

func usesIota(spec *ast.ValueSpec) bool {
	for _, value := range spec.Values {
		found := false
		ast.Inspect(value, func(node ast.Node) bool {
			if identifier, ok := node.(*ast.Ident); ok && identifier.Name == "iota" {
				found = true
			}
			return !found
		})
		if found {
			return true
		}
	}
	return len(spec.Values) == 0
}

func numericLiteral(expression ast.Expr) string {
	numeric := true
	ast.Inspect(expression, func(node ast.Node) bool {
		switch leaf := node.(type) {
		case *ast.BasicLit:
			if leaf.Kind != token.INT && leaf.Kind != token.FLOAT {
				numeric = false
			}
		case *ast.Ident, *ast.CallExpr, *ast.SelectorExpr:
			numeric = false
		}
		return numeric
	})
	if !numeric {
		return ""
	}
	return render(expression)
}

func render(expression ast.Expr) string {
	switch typed := expression.(type) {
	case *ast.BasicLit:
		return typed.Value
	case *ast.BinaryExpr:
		return render(typed.X) + typed.Op.String() + render(typed.Y)
	case *ast.UnaryExpr:
		return typed.Op.String() + render(typed.X)
	case *ast.ParenExpr:
		return "(" + render(typed.X) + ")"
	default:
		return ""
	}
}

func score(name, doc, value string) int {
	text := strings.ToLower(name + " " + doc)
	total := 0
	for _, hot := range []string{"max", "min", "limit", "bound", "budget", "threshold", "depth", "width", "iter", "retry", "timeout", "window", "sample", "multiplier", "seq", "batch"} {
		if strings.Contains(text, hot) {
			total += 2
		}
	}
	for _, cold := range []string{"offset", "version", "magic-number", "header", "byte", "kind", "schema"} {
		if strings.Contains(text, cold) {
			total--
		}
	}
	if value != "0" && value != "1" {
		total++
	}
	return total
}
