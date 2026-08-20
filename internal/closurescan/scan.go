// Package closurescan owns the constant-scanning core shared by the
// closure-scan CLI and the gate's magic step: parse production Go, collect
// numeric constants (iota enumerations, tests, testdata, and generated code
// excluded -- enumerations are not magics), and rank them for triage. The
// score orders reports and admits nothing; judgment happens at triage.
package closurescan

import (
	"bytes"
	"go/ast"
	"go/constant"
	"go/format"
	"go/token"
	"maps"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/repoanalysis"
)

type Candidate struct {
	Name       string `json:"name"`
	File       string `json:"file"`
	Package    string `json:"package"`
	Scope      string `json:"scope"`
	Line       int    `json:"line"`
	Expression string `json:"expression"`
	Value      string `json:"value"`
	SourceID   string `json:"source_id"`
	Doc        string `json:"doc,omitempty"`
	Score      int    `json:"score"`
}

func (c Candidate) DeclarationKey() string {
	return c.File + "\x00" + c.Scope + "\x00" + strconv.Itoa(c.Line) + "\x00" + c.Name
}

func (c Candidate) ExactKey() string {
	return c.DeclarationKey() + "\x00" + c.Expression + "\x00" + c.Value + "\x00" + c.SourceID
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

type LiteralContext string

const (
	LiteralComparison LiteralContext = "comparison"
	LiteralCall       LiteralContext = "call_argument"
	LiteralComposite  LiteralContext = "composite_value"
	LiteralIndex      LiteralContext = "index"
	LiteralSlice      LiteralContext = "slice_bound"
	LiteralExtent     LiteralContext = "array_extent"
	LiteralArithmetic LiteralContext = "arithmetic_operand"
	LiteralAssignment LiteralContext = "assignment"
	LiteralReturn     LiteralContext = "return"
	LiteralCase       LiteralContext = "case"
	LiteralOther      LiteralContext = "other"
)

type LiteralSite struct {
	File       string         `json:"file"`
	Package    string         `json:"package"`
	Scope      string         `json:"scope"`
	Line       int            `json:"line"`
	Offset     int            `json:"offset"`
	Kind       string         `json:"kind"`
	Expression string         `json:"expression"`
	Value      string         `json:"value"`
	Context    LiteralContext `json:"context"`
	SourceID   string         `json:"source_id"`
}

// CensusLiterals classifies non-const numeric source literals.
func CensusLiterals(snapshot repoanalysis.SourceSnapshot, relatives []string) ([]LiteralSite, error) {
	var sites []LiteralSite
	err := visitProduction(snapshot, relatives, func(source repoanalysis.GoFile, parsed *ast.File) {
		collectLiteralSites(parsed, source, &sites)
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(sites, func(i, j int) bool {
		if sites[i].File != sites[j].File {
			return sites[i].File < sites[j].File
		}
		return sites[i].Offset < sites[j].Offset
	})
	return sites, nil
}

func collectLiteralSites(file *ast.File, source repoanalysis.GoFile, out *[]LiteralSite) {
	parents := map[ast.Node]ast.Node{}
	var stack []ast.Node
	ast.Inspect(file, func(node ast.Node) bool {
		if node == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		if len(stack) > 0 {
			parents[node] = stack[len(stack)-1]
		}
		stack = append(stack, node)
		literal, ok := node.(*ast.BasicLit)
		if !ok || insideConst(node, parents) || literal.Kind != token.INT && literal.Kind != token.FLOAT {
			return true
		}
		expression, parent := ast.Expr(literal), parents[node]
		if unary, ok := parent.(*ast.UnaryExpr); ok && unary.X == literal {
			expression, parent = unary, parents[unary]
		}
		value, ok := evaluateConstant(expression, nil, nil)
		if !ok {
			return true
		}
		*out = append(*out, LiteralSite{
			File: source.Path, Package: filepath.ToSlash(filepath.Dir(source.Path)),
			Scope: literalScope(node, parents), Line: source.Line(expression.Pos()), Offset: int(expression.Pos()) - 1,
			Kind: literal.Kind.String(), Expression: formatExpression(expression), Value: value.ExactString(),
			Context: literalContext(parent, parents), SourceID: source.ContentID,
		})
		return true
	})
}

func insideConst(node ast.Node, parents map[ast.Node]ast.Node) bool {
	for node != nil {
		if generic, ok := node.(*ast.GenDecl); ok {
			return generic.Tok == token.CONST
		}
		node = parents[node]
	}
	return false
}

func literalScope(node ast.Node, parents map[ast.Node]ast.Node) string {
	for node != nil {
		if function, ok := node.(*ast.FuncDecl); ok {
			return functionIdentity(function)
		}
		node = parents[node]
	}
	return "package"
}

func literalContext(parent ast.Node, parents map[ast.Node]ast.Node) LiteralContext {
	for {
		switch parent.(type) {
		case *ast.ParenExpr, *ast.UnaryExpr:
			parent = parents[parent]
			continue
		}
		break
	}
	switch typed := parent.(type) {
	case *ast.BinaryExpr:
		if typed.Op >= token.EQL && typed.Op <= token.GEQ {
			return LiteralComparison
		}
		return LiteralArithmetic
	case *ast.CallExpr:
		return LiteralCall
	case *ast.CompositeLit, *ast.KeyValueExpr:
		return LiteralComposite
	case *ast.IndexExpr, *ast.IndexListExpr:
		return LiteralIndex
	case *ast.SliceExpr:
		return LiteralSlice
	case *ast.ArrayType:
		return LiteralExtent
	case *ast.AssignStmt, *ast.ValueSpec:
		return LiteralAssignment
	case *ast.ReturnStmt:
		return LiteralReturn
	case *ast.CaseClause:
		return LiteralCase
	default:
		return LiteralOther
	}
}

// ScanRoot walks internal/ and cmd/ under root.
func ScanRoot(root string) ([]Candidate, error) {
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		return nil, err
	}
	return ScanSnapshot(snapshot, nil)
}

// ScanSnapshot reuses parsed source and optionally limits findings to paths.
func ScanSnapshot(snapshot repoanalysis.SourceSnapshot, relatives []string) ([]Candidate, error) {
	var out []Candidate
	err := visitProduction(snapshot, relatives, func(source repoanalysis.GoFile, parsed *ast.File) {
		collect(parsed, source, &out)
	})
	if err != nil {
		return nil, err
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].DeclarationKey() < out[j].DeclarationKey()
	})
	return out, nil
}

func visitProduction(snapshot repoanalysis.SourceSnapshot, relatives []string, visit func(repoanalysis.GoFile, *ast.File)) error {
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
			return err
		}
		if generated {
			continue
		}
		parsed, err := source.Syntax()
		if err != nil {
			return err
		}
		visit(source, parsed)
	}
	return nil
}

// RankRawPolicyLiterals finds repeated non-trivial numeric and string literals
// in function bodies. Indexes, slice/array extents, arithmetic factors, tests,
// and generated files are excluded to avoid recommending constants for local
// math or structure facts.
func RankRawPolicyLiterals(snapshot repoanalysis.SourceSnapshot) ([]RawPolicyLiteral, error) {
	groups := map[string]*rawLiteralGroup{}
	err := visitProduction(snapshot, nil, func(source repoanalysis.GoFile, parsed *ast.File) {
		pkg := filepath.ToSlash(filepath.Dir(source.Path))
		for _, declaration := range parsed.Decls {
			function, ok := declaration.(*ast.FuncDecl)
			if !ok || function.Body == nil {
				continue
			}
			collectRawLiterals(function, pkg, source.Path, groups)
		}
	})
	if err != nil {
		return nil, err
	}
	var ranked []RawPolicyLiteral
	for key, group := range groups {
		if len(group.functions) < 2 {
			continue
		}
		parts := strings.SplitN(key, "\x00", 2)
		row := RawPolicyLiteral{
			Package: parts[0], Value: parts[1], Count: group.count,
			Functions: slices.Sorted(maps.Keys(group.functions)), Files: slices.Sorted(maps.Keys(group.files)),
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
	case token.INT, token.FLOAT:
		return true
	case token.STRING:
		value, err := strconv.Unquote(literal.Value)
		return err == nil && value != "" && !strings.ContainsAny(value, "%\r\n\t")
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

type constBinding struct {
	name       *ast.Ident
	expression ast.Expr
	doc        string
	scope      string
}

func collect(file *ast.File, source repoanalysis.GoFile, out *[]Candidate) {
	bindings := map[*ast.Object]ast.Expr{}
	var declarations []constBinding
	for _, declaration := range file.Decls {
		switch typed := declaration.(type) {
		case *ast.GenDecl:
			collectConstBindings(typed, "package", bindings, &declarations)
		case *ast.FuncDecl:
			scope := functionIdentity(typed)
			ast.Inspect(typed.Body, func(node ast.Node) bool {
				generic, ok := node.(*ast.GenDecl)
				if ok {
					collectConstBindings(generic, scope, bindings, &declarations)
				}
				return true
			})
		}
	}
	for _, declaration := range declarations {
		value, ok := evaluateConstant(declaration.expression, bindings, map[*ast.Object]bool{})
		if !ok || value.Kind() != constant.Int && value.Kind() != constant.Float {
			continue
		}
		expression := formatExpression(declaration.expression)
		evaluated := value.ExactString()
		*out = append(*out, Candidate{
			Name: declaration.name.Name, File: source.Path,
			Package: filepath.ToSlash(filepath.Dir(source.Path)),
			Scope:   declaration.scope, Line: source.Line(declaration.name.Pos()),
			Expression: expression, Value: evaluated, SourceID: source.ContentID,
			Doc: declaration.doc, Score: score(declaration.name.Name, declaration.doc, evaluated),
		})
	}
}

func collectConstBindings(generic *ast.GenDecl, scope string, bindings map[*ast.Object]ast.Expr, out *[]constBinding) {
	if generic == nil || generic.Tok != token.CONST {
		return
	}
	blockDoc := ""
	if generic.Doc != nil {
		blockDoc = strings.TrimSpace(generic.Doc.Text())
	}
	var inherited []ast.Expr
	for _, spec := range generic.Specs {
		value, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}
		if len(value.Values) > 0 {
			inherited = value.Values
		}
		doc := blockDoc
		if value.Doc != nil {
			doc = strings.TrimSpace(value.Doc.Text())
		}
		for index, name := range value.Names {
			if name.Name == "_" || index >= len(inherited) {
				continue
			}
			expression := inherited[index]
			if containsIota(expression) {
				continue
			}
			if name.Obj != nil {
				bindings[name.Obj] = expression
			}
			*out = append(*out, constBinding{name: name, expression: expression, doc: doc, scope: scope})
		}
	}
}

func containsIota(expression ast.Expr) bool {
	found := false
	ast.Inspect(expression, func(node ast.Node) bool {
		if identifier, ok := node.(*ast.Ident); ok && identifier.Name == "iota" {
			found = true
		}
		return !found
	})
	return found
}

func functionIdentity(function *ast.FuncDecl) string {
	if function.Recv == nil || len(function.Recv.List) == 0 {
		return function.Name.Name
	}
	return formatExpression(function.Recv.List[0].Type) + "." + function.Name.Name
}

func formatExpression(expression ast.Expr) string {
	var buffer bytes.Buffer
	if err := format.Node(&buffer, token.NewFileSet(), expression); err != nil {
		return ""
	}
	return buffer.String()
}

func evaluateConstant(expression ast.Expr, bindings map[*ast.Object]ast.Expr, visiting map[*ast.Object]bool) (constant.Value, bool) {
	switch typed := expression.(type) {
	case *ast.BasicLit:
		if typed.Kind != token.INT && typed.Kind != token.FLOAT {
			return nil, false
		}
		value := constant.MakeFromLiteral(typed.Value, typed.Kind, 0)
		return value, value.Kind() != constant.Unknown
	case *ast.ParenExpr:
		return evaluateConstant(typed.X, bindings, visiting)
	case *ast.Ident:
		if typed.Obj == nil || visiting[typed.Obj] {
			return nil, false
		}
		bound, ok := bindings[typed.Obj]
		if !ok {
			return nil, false
		}
		visiting[typed.Obj] = true
		value, ok := evaluateConstant(bound, bindings, visiting)
		delete(visiting, typed.Obj)
		return value, ok
	case *ast.UnaryExpr:
		value, ok := evaluateConstant(typed.X, bindings, visiting)
		if !ok || typed.Op != token.ADD && typed.Op != token.SUB && typed.Op != token.XOR {
			return nil, false
		}
		return constant.UnaryOp(typed.Op, value, 0), true
	case *ast.BinaryExpr:
		left, ok := evaluateConstant(typed.X, bindings, visiting)
		if !ok {
			return nil, false
		}
		right, ok := evaluateConstant(typed.Y, bindings, visiting)
		if !ok {
			return nil, false
		}
		if typed.Op == token.SHL || typed.Op == token.SHR {
			shift, exact := constant.Uint64Val(constant.ToInt(right))
			if !exact {
				return nil, false
			}
			return constant.Shift(left, typed.Op, uint(shift)), true
		}
		switch typed.Op {
		case token.ADD, token.SUB, token.MUL, token.QUO, token.REM,
			token.AND, token.OR, token.XOR, token.AND_NOT:
			return constant.BinaryOp(left, typed.Op, right), true
		default:
			return nil, false
		}
	default:
		return nil, false
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
