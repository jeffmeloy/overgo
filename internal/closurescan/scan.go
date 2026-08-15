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
	"strconv"
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

// RawPolicyLiteral is an advisory group of the same raw literal repeated in
// multiple functions in one package. Functions and files are ownership
// evidence; the score ranks inspection and never authorizes extraction.
type RawPolicyLiteral struct {
	Value     string   `json:"value"`
	Package   string   `json:"package"`
	Functions []string `json:"functions"`
	Files     []string `json:"files"`
	Count     int      `json:"count"`
	Score     int      `json:"score"`
}

type rawLiteralGroup struct {
	functions map[string]bool
	files     map[string]bool
	count     int
	policy    int
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

// RankRawPolicyLiterals finds repeated non-trivial numeric and string literals
// in function bodies. Indexes, slice/array extents, arithmetic factors, tests,
// and generated files are excluded to avoid recommending constants for local
// math or structure facts.
func RankRawPolicyLiterals(snapshot repoanalysis.SourceSnapshot) ([]RawPolicyLiteral, error) {
	groups := map[string]*rawLiteralGroup{}
	for _, source := range snapshot.Files {
		if source.Test {
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
		pkg := filepath.ToSlash(filepath.Dir(source.Path))
		for _, declaration := range parsed.Decls {
			function, ok := declaration.(*ast.FuncDecl)
			if !ok || function.Body == nil {
				continue
			}
			collectRawLiterals(function, pkg, source.Path, groups)
		}
	}
	var ranked []RawPolicyLiteral
	for key, group := range groups {
		if len(group.functions) < 2 {
			continue
		}
		parts := strings.SplitN(key, "\x00", 2)
		row := RawPolicyLiteral{
			Package: parts[0], Value: parts[1], Count: group.count,
			Functions: sortedKeys(group.functions), Files: sortedKeys(group.files),
		}
		row.Score = row.Count + 2*len(row.Functions) + len(row.Files) + 2*group.policy
		if !strings.HasPrefix(row.Value, "\"") && !strings.HasPrefix(row.Value, "`") {
			row.Score++
		}
		ranked = append(ranked, row)
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].Score != ranked[j].Score {
			return ranked[i].Score > ranked[j].Score
		}
		if ranked[i].Package != ranked[j].Package {
			return ranked[i].Package < ranked[j].Package
		}
		return ranked[i].Value < ranked[j].Value
	})
	return ranked, nil
}

func collectRawLiterals(function *ast.FuncDecl, pkg, file string, groups map[string]*rawLiteralGroup) {
	var stack []ast.Node
	ast.Inspect(function.Body, func(node ast.Node) bool {
		if node == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		var parent ast.Node
		if len(stack) != 0 {
			parent = stack[len(stack)-1]
		}
		stack = append(stack, node)
		literal, ok := node.(*ast.BasicLit)
		if !ok || !rawPolicyValue(literal) || structuralLiteral(parent) {
			return true
		}
		key := pkg + "\x00" + literal.Value
		group := groups[key]
		if group == nil {
			group = &rawLiteralGroup{functions: map[string]bool{}, files: map[string]bool{}}
			groups[key] = group
		}
		group.functions[file+"#"+function.Name.Name] = true
		group.files[file] = true
		group.count++
		if comparisonLiteral(parent) {
			group.policy++
		}
		return true
	})
}

func rawPolicyValue(literal *ast.BasicLit) bool {
	switch literal.Kind {
	case token.INT:
		value, err := strconv.ParseInt(literal.Value, 0, 64)
		return err != nil || value > 16
	case token.FLOAT:
		return literal.Value != "0.0" && literal.Value != "1.0" && literal.Value != "2.0"
	case token.STRING:
		value, err := strconv.Unquote(literal.Value)
		return err == nil && len(value) >= 4 && !strings.ContainsAny(value, "%\r\n \t")
	default:
		return false
	}
}

func structuralLiteral(parent ast.Node) bool {
	switch typed := parent.(type) {
	case *ast.IndexExpr, *ast.IndexListExpr, *ast.SliceExpr, *ast.ArrayType, *ast.Field:
		return true
	case *ast.BinaryExpr:
		switch typed.Op {
		case token.ADD, token.SUB, token.MUL, token.QUO, token.REM, token.SHL, token.SHR, token.AND, token.OR, token.XOR, token.AND_NOT:
			return true
		}
	}
	return false
}

func comparisonLiteral(parent ast.Node) bool {
	binary, ok := parent.(*ast.BinaryExpr)
	return ok && binary.Op >= token.EQL && binary.Op <= token.GEQ
}

func sortedKeys(values map[string]bool) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
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
