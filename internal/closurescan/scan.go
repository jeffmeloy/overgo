// Package closurescan owns exact literal and assumption source bindings.
package closurescan

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/constant"
	"go/format"
	"go/token"
	"hash"
	"io"
	"maps"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/repoanalysis"
)

type Candidate struct {
	Kind         closureledger.BindingKind `json:"kind"`
	Name         string                    `json:"name"`
	File         string                    `json:"file"`
	Package      string                    `json:"package"`
	Scope        string                    `json:"scope"`
	Line         int                       `json:"line"`
	Expression   string                    `json:"expression"`
	Value        string                    `json:"value"`
	StructuralID string                    `json:"structural_id"`
	SourceID     string                    `json:"source_id"`
	CallsiteID   string                    `json:"callsite_id"`
	OwnerID      string                    `json:"owner_id"`
	Policy       bool                      `json:"policy"`
	Doc          string                    `json:"doc,omitempty"`
	Score        int                       `json:"score"`
}

type CandidateKinds uint8

const (
	candidateNone      CandidateKinds = iota
	CandidateConstants CandidateKinds = 1 << (iota - 1)
	CandidateLiterals
	CandidateAssumptions
	CandidateAll CandidateKinds = (1 << (iota - 1)) - 1
)

func (k CandidateKinds) valid() bool {
	return k != candidateNone && k&^CandidateAll == candidateNone
}

func (k CandidateKinds) includes(candidate CandidateKinds) bool {
	return k&candidate != candidateNone
}

func (c Candidate) DeclarationKey() string {
	if c.StructuralID != "" {
		return string(c.Kind) + "\x00" + c.Package + "\x00" + c.File + "\x00" + c.Scope + "\x00" + c.StructuralID
	}
	return c.LegacyDeclarationKey()
}

// LegacyDeclarationKey identifies the old source-location representation for migration and triage.
func (c Candidate) LegacyDeclarationKey() string {
	return string(c.Kind) + "\x00" + c.File + "\x00" + c.Scope + "\x00" + strconv.Itoa(c.Line) + "\x00" + c.Name
}

func (c Candidate) ExactKey() string {
	return c.DeclarationKey() + "\x00" + c.Expression + "\x00" + c.Value + "\x00" + c.SourceID + "\x00" + c.CallsiteID
}

func (c Candidate) DecisionKey() string {
	return string(c.Kind) + "\x00" + c.Package + "\x00" + c.File + "\x00" + c.Scope + "\x00" + c.Name + "\x00" + c.Expression + "\x00" + c.Value
}

func (c Candidate) ValueJSON() json.RawMessage {
	value, err := json.Marshal(json.Number(c.Value))
	if err == nil {
		return value
	}
	value, _ = json.Marshal(c.Value)
	return value
}

func (c Candidate) Binding() (closureledger.SourceBinding, error) {
	ownerID := c.OwnerID
	if ownerID == "" {
		ownerID = c.SourceID
	}
	decoded, err := hex.DecodeString(ownerID)
	if err != nil || len(decoded) != sha256.Size {
		return closureledger.SourceBinding{}, fmt.Errorf("closure scan: invalid source identity for %s", c.Name)
	}
	var digest [sha256.Size]byte
	copy(digest[:], decoded)
	owner, err := artifact.NewID(artifact.KindFile, digest)
	if err != nil {
		return closureledger.SourceBinding{}, err
	}
	return closureledger.SourceBinding{
		Kind: c.Kind, Package: c.Package, File: c.File,
		Scope: c.Scope, Name: c.Name, Line: c.Line, Expression: c.Expression,
		StructuralID: c.StructuralID, SourceID: c.SourceID, CallsiteID: c.CallsiteID, Owner: owner,
	}, nil
}

func (s LiteralSite) Candidate() Candidate {
	return Candidate{
		Kind: closureledger.BindingLiteral, Name: "literal." + strconv.Itoa(s.Offset),
		File: s.File, Package: s.Package, Scope: s.Scope, Line: s.Line,
		Expression: s.Expression, Value: s.Value, SourceID: s.SourceID,
		OwnerID:    s.SourceID,
		CallsiteID: s.SourceID,
		Policy:     s.Policy,
	}
}

func (s AssumptionHint) Candidate() Candidate {
	return Candidate{
		Kind: closureledger.BindingAssumption,
		Name: "assumption." + string(s.Kind) + "." + strconv.Itoa(s.Offset),
		File: s.File, Package: s.Package, Scope: s.Scope, Line: s.Line,
		Expression: s.Expression, Value: string(s.Kind), SourceID: s.SourceID,
		OwnerID:    s.SourceID,
		CallsiteID: s.SourceID,
		Policy:     s.Policy,
	}
}

func runtimeLiteralContext(context LiteralContext) bool {
	switch context {
	case LiteralIndex, LiteralSlice, LiteralExtent, LiteralArithmetic:
		return false
	default:
		return true
	}
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
	Policy     bool           `json:"policy"`
}

type TestLiteralClass string

const (
	TestFixture    TestLiteralClass = "fixture"
	TestAssertion  TestLiteralClass = "assertion"
	TestPolicyCopy TestLiteralClass = "policy_copy"
)

type TestLiteralSite struct {
	LiteralSite
	Named             bool             `json:"named"`
	Class             TestLiteralClass `json:"class"`
	ProductionMatches []string         `json:"production_matches,omitempty"`
}

type AssumptionKind string

const (
	AssumptionMoment       AssumptionKind = "moment_scale"
	AssumptionQuantile     AssumptionKind = "quantile"
	AssumptionDistribution AssumptionKind = "distribution_family"
	AssumptionGeometry     AssumptionKind = "geometry"
	AssumptionShape        AssumptionKind = "shape"
	AssumptionIndependence AssumptionKind = "independence"
)

type AssumptionHint struct {
	File       string         `json:"file"`
	Package    string         `json:"package"`
	Scope      string         `json:"scope"`
	Line       int            `json:"line"`
	Offset     int            `json:"offset"`
	Kind       AssumptionKind `json:"kind"`
	Expression string         `json:"expression"`
	SourceID   string         `json:"source_id"`
	Policy     bool           `json:"policy"`
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

// CensusTestLiterals separates fixtures, assertions, and production overlaps.
func CensusTestLiterals(snapshot repoanalysis.SourceSnapshot) ([]TestLiteralSite, error) {
	var out []TestLiteralSite
	err := visitTestLiterals(snapshot, func(site TestLiteralSite) { out = append(out, site) })
	if err != nil {
		return nil, err
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].File != out[j].File {
			return out[i].File < out[j].File
		}
		return out[i].Offset < out[j].Offset || out[i].Offset == out[j].Offset && out[i].Line < out[j].Line
	})
	return out, nil
}

func CountTestPolicyLiterals(snapshot repoanalysis.SourceSnapshot) (int, error) {
	var count int
	err := visitTestLiterals(snapshot, func(site TestLiteralSite) {
		if site.Class == TestPolicyCopy {
			count++
		}
	})
	return count, err
}

func visitTestLiterals(snapshot repoanalysis.SourceSnapshot, visit func(TestLiteralSite)) error {
	production := map[string][]string{}
	err := visitProduction(snapshot, nil, func(source repoanalysis.GoFile, file *ast.File) {
		var candidates []Candidate
		collect(file, source, &candidates)
		for _, candidate := range candidates {
			addProductionValue(production, candidate.Package, candidate.Value, candidate.File, candidate.Line)
		}
	})
	if err != nil {
		return err
	}
	err = visitSources(snapshot, nil, func(source repoanalysis.GoFile) bool { return source.Test }, func(source repoanalysis.GoFile, file *ast.File) {
		var literals []LiteralSite
		collectLiteralSites(file, source, &literals)
		for _, site := range literals {
			visit(classifyTestLiteral(site, false, production))
		}
		var named []Candidate
		collect(file, source, &named)
		for _, candidate := range named {
			visit(classifyTestLiteral(LiteralSite{
				File: candidate.File, Package: candidate.Package, Scope: candidate.Scope,
				Line: candidate.Line, Expression: candidate.Expression, Value: candidate.Value,
				Context: LiteralOther, SourceID: candidate.SourceID,
			}, true, production))
		}
	})
	return err
}

func addProductionValue(values map[string][]string, pkg, value, file string, line int) {
	key := pkg + "\x00" + value
	values[key] = append(values[key], file+":"+strconv.Itoa(line))
}

func classifyTestLiteral(site LiteralSite, named bool, production map[string][]string) TestLiteralSite {
	matches := slices.Clone(production[site.Package+"\x00"+site.Value])
	class := TestFixture
	if site.Context == LiteralComparison {
		class = TestAssertion
	}
	if len(matches) > 0 && (named || site.Context == LiteralComparison) {
		class = TestPolicyCopy
	}
	return TestLiteralSite{LiteralSite: site, Named: named, Class: class, ProductionMatches: matches}
}

// CensusAssumptions reports syntax-derived decision hints; it proves none.
func CensusAssumptions(snapshot repoanalysis.SourceSnapshot, relatives []string) ([]AssumptionHint, error) {
	var out []AssumptionHint
	err := visitProduction(snapshot, relatives, func(source repoanalysis.GoFile, file *ast.File) {
		collectAssumptionHints(file, source, &out)
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].File != out[j].File {
			return out[i].File < out[j].File
		}
		return out[i].Offset < out[j].Offset || out[i].Offset == out[j].Offset && out[i].Kind < out[j].Kind
	})
	return out, nil
}

var identifierWords = regexp.MustCompile(`[A-Z]+(?:[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+`)

var assumptionWordKinds = map[string]AssumptionKind{
	"mean": AssumptionMoment, "variance": AssumptionMoment, "stddev": AssumptionMoment, "sigma": AssumptionMoment,
	"quantile": AssumptionQuantile, "percentile": AssumptionQuantile,
	"gaussian": AssumptionDistribution, "normal": AssumptionDistribution, "poisson": AssumptionDistribution,
	"euclidean": AssumptionGeometry, "cosine": AssumptionGeometry, "distance": AssumptionGeometry, "l2": AssumptionGeometry,
	"shape": AssumptionShape, "rank": AssumptionShape, "dim": AssumptionShape, "rows": AssumptionShape,
	"cols": AssumptionShape, "width": AssumptionShape, "height": AssumptionShape, "channels": AssumptionShape,
	"iid": AssumptionIndependence, "independent": AssumptionIndependence, "shuffle": AssumptionIndependence,
}

func assumptionKinds(expression ast.Expr) []AssumptionKind {
	kinds := map[AssumptionKind]bool{}
	ast.Inspect(expression, func(node ast.Node) bool {
		identifier, ok := node.(*ast.Ident)
		if !ok {
			return true
		}
		for _, word := range identifierWords.FindAllString(identifier.Name, -1) {
			if kind, ok := assumptionWordKinds[strings.ToLower(word)]; ok {
				kinds[kind] = true
			}
		}
		return true
	})
	return slices.Sorted(maps.Keys(kinds))
}

func collectLiteralSites(file *ast.File, source repoanalysis.GoFile, out *[]LiteralSite) {
	inspectWithParents(file, func(node ast.Node, parents map[ast.Node]ast.Node) {
		literal, ok := node.(*ast.BasicLit)
		if !ok || insideConst(node, parents) || literal.Kind != token.INT && literal.Kind != token.FLOAT {
			return
		}
		expression, parent := ast.Expr(literal), parents[node]
		if unary, ok := parent.(*ast.UnaryExpr); ok && unary.X == literal {
			expression, parent = unary, parents[unary]
		}
		value, ok := evaluateConstant(expression, nil, nil)
		if !ok {
			return
		}
		context := literalContext(parent, parents)
		*out = append(*out, LiteralSite{
			File: source.Path, Package: filepath.ToSlash(filepath.Dir(source.Path)),
			Scope: literalScope(node, parents), Line: source.Line(expression.Pos()), Offset: int(expression.Pos()) - 1,
			Kind: literal.Kind.String(), Expression: formatExpression(expression), Value: value.ExactString(),
			Context: context, SourceID: source.ContentID,
			Policy: runtimeLiteralContext(context) && !structuralIdentity(literal, parent, parents),
		})
	})
}

func structuralIdentity(literal *ast.BasicLit, parent ast.Node, parents map[ast.Node]ast.Node) bool {
	if literal.Value != "0" && literal.Value != "1" {
		return false
	}
	switch value := parent.(type) {
	case *ast.BinaryExpr:
		if value.Op == token.EQL || value.Op == token.NEQ || value.Op == token.LSS ||
			value.Op == token.LEQ || value.Op == token.GTR || value.Op == token.GEQ {
			return true
		}
		other := value.X
		if other == literal {
			other = value.Y
		}
		call, ok := other.(*ast.CallExpr)
		if ok {
			name, ok := call.Fun.(*ast.Ident)
			return ok && (name.Name == "len" || name.Name == "cap")
		}
		index, ok := other.(*ast.Ident)
		return ok && enclosingRangeKey(value, index, parents)
	case *ast.CallExpr:
		name, ok := value.Fun.(*ast.Ident)
		return ok && (name.Name == "make" || numericBuiltin(name.Name))
	case *ast.ReturnStmt:
		for _, result := range value.Results {
			if result != literal && errorLike(result) {
				return true
			}
		}
		return false
	case *ast.AssignStmt:
		if literal.Value == "0" && value.Tok == token.DEFINE {
			return true
		}
		loop, ok := parents[value].(*ast.ForStmt)
		return literal.Value == "0" && ok && loop.Init == value
	default:
		return false
	}
}

func errorLike(expression ast.Expr) bool {
	switch value := expression.(type) {
	case *ast.Ident:
		return value.Name == "nil" || value.Name == "err" || strings.HasSuffix(value.Name, "Err")
	case *ast.CallExpr:
		switch function := value.Fun.(type) {
		case *ast.Ident:
			return strings.Contains(strings.ToLower(function.Name), "err")
		case *ast.SelectorExpr:
			if receiver, ok := function.X.(*ast.Ident); ok && receiver.Name == "errors" {
				return true
			}
			return function.Sel.Name == "New" || strings.Contains(strings.ToLower(function.Sel.Name), "err")
		}
	}
	return false
}

func numericBuiltin(name string) bool {
	switch name {
	case "int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", "uintptr", "float32", "float64", "complex64", "complex128":
		return true
	default:
		return false
	}
}

func enclosingRangeKey(node ast.Node, index *ast.Ident, parents map[ast.Node]ast.Node) bool {
	for node != nil {
		if statement, ok := node.(*ast.RangeStmt); ok {
			key, ok := statement.Key.(*ast.Ident)
			return ok && (key == index || key.Obj != nil && key.Obj == index.Obj)
		}
		node = parents[node]
	}
	return false
}

func inspectWithParents(root ast.Node, visit func(ast.Node, map[ast.Node]ast.Node)) {
	parents := map[ast.Node]ast.Node{}
	var stack []ast.Node
	ast.Inspect(root, func(node ast.Node) bool {
		if node == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		if len(stack) > 0 {
			parents[node] = stack[len(stack)-1]
		}
		stack = append(stack, node)
		visit(node, parents)
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
func ScanRoot(root string, kinds CandidateKinds) ([]Candidate, error) {
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		return nil, err
	}
	return ScanSnapshot(snapshot, nil, kinds)
}

// ScanSnapshot reuses parsed source and optionally limits findings to paths.
func ScanSnapshot(snapshot repoanalysis.SourceSnapshot, relatives []string, kinds CandidateKinds) ([]Candidate, error) {
	if !kinds.valid() {
		return nil, fmt.Errorf("closure scan: invalid candidate kinds %d", kinds)
	}
	var out []Candidate
	objects := map[*ast.Object]int{}
	err := visitProduction(snapshot, relatives, func(source repoanalysis.GoFile, parsed *ast.File) {
		if kinds.includes(CandidateConstants) {
			collectObjects(parsed, source, &out, objects)
		}
		if kinds.includes(CandidateLiterals) {
			var sites []LiteralSite
			collectLiteralSites(parsed, source, &sites)
			for _, site := range sites {
				out = append(out, site.Candidate())
			}
		}
		if kinds.includes(CandidateAssumptions) {
			collectAssumptionCandidates(parsed, source, &out)
		}
	})
	if err != nil {
		return nil, err
	}
	assignStructuralIdentities(out)
	if kinds.includes(CandidateConstants) {
		if err := bindCallsites(snapshot, out, objects); err != nil {
			return nil, err
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].DeclarationKey() < out[j].DeclarationKey()
	})
	return out, nil
}

func collectAssumptionCandidates(file *ast.File, source repoanalysis.GoFile, out *[]Candidate) {
	var hints []AssumptionHint
	collectAssumptionHints(file, source, &hints)
	for _, hint := range hints {
		*out = append(*out, hint.Candidate())
	}
}

func collectAssumptionHints(file *ast.File, source repoanalysis.GoFile, out *[]AssumptionHint) {
	seen := map[string]bool{}
	imports := importedNames(file)
	receivers := importedTypedNames(file, imports)
	inspectWithParents(file, func(node ast.Node, parents map[ast.Node]ast.Node) {
		var expression, subject, authority ast.Expr
		switch typed := node.(type) {
		case *ast.CallExpr:
			expression, subject, authority = typed, calledName(typed.Fun), typed.Fun
		case *ast.BinaryExpr:
			if typed.Op >= token.EQL && typed.Op <= token.GEQ {
				expression, subject, authority = typed, typed, typed
			}
		}
		if expression == nil {
			return
		}
		for _, kind := range assumptionKinds(subject) {
			offset := int(expression.Pos()) - 1
			key := strconv.Itoa(offset) + "\x00" + string(kind)
			if seen[key] {
				continue
			}
			seen[key] = true
			*out = append(*out, AssumptionHint{
				File: source.Path, Package: filepath.ToSlash(filepath.Dir(source.Path)),
				Scope: literalScope(node, parents), Line: source.Line(expression.Pos()), Offset: offset,
				Kind: kind, Expression: formatExpression(expression), SourceID: source.ContentID,
				Policy: !importedSelector(authority, imports, receivers),
			})
		}
	})
}

func calledName(expression ast.Expr) ast.Expr {
	if selector, ok := expression.(*ast.SelectorExpr); ok {
		return selector.Sel
	}
	return expression
}

func importedNames(file *ast.File) map[string]bool {
	names := make(map[string]bool, len(file.Imports))
	for _, spec := range file.Imports {
		name := ""
		if spec.Name != nil {
			name = spec.Name.Name
		} else if path, err := strconv.Unquote(spec.Path.Value); err == nil {
			name = filepath.Base(path)
		}
		if name != "" && name != "." && name != "_" {
			names[name] = true
		}
	}
	return names
}

func importedTypedNames(file *ast.File, imports map[string]bool) map[string]bool {
	names := map[string]bool{}
	ast.Inspect(file, func(node ast.Node) bool {
		field, ok := node.(*ast.Field)
		if !ok || !typeUsesImport(field.Type, imports) {
			return true
		}
		for _, name := range field.Names {
			names[name.Name] = true
		}
		return true
	})
	return names
}

func typeUsesImport(expression ast.Expr, imports map[string]bool) bool {
	found := false
	ast.Inspect(expression, func(node ast.Node) bool {
		selector, ok := node.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		identifier, ok := selector.X.(*ast.Ident)
		if ok && imports[identifier.Name] {
			found = true
			return false
		}
		return true
	})
	return found
}

func importedSelector(expression ast.Expr, imports, receivers map[string]bool) bool {
	selector, ok := expression.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	for expression = selector.X; ; {
		switch value := expression.(type) {
		case *ast.Ident:
			return imports[value.Name] || receivers[value.Name]
		case *ast.SelectorExpr:
			expression = value.X
		default:
			return false
		}
	}
}

func visitProduction(snapshot repoanalysis.SourceSnapshot, relatives []string, visit func(repoanalysis.GoFile, *ast.File)) error {
	return visitSources(snapshot, relatives, func(source repoanalysis.GoFile) bool { return !source.Test }, visit)
}

func visitSources(snapshot repoanalysis.SourceSnapshot, relatives []string, include func(repoanalysis.GoFile) bool, visit func(repoanalysis.GoFile, *ast.File)) error {
	wanted := map[string]bool{}
	for _, relative := range relatives {
		wanted[filepath.ToSlash(relative)] = true
	}
	for _, source := range snapshot.Files {
		if !include(source) || len(wanted) > 0 && !wanted[source.Path] {
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

// RepeatedPolicyLiterals finds repeated non-trivial numeric and string literals
// in function bodies. Indexes, slice/array extents, arithmetic factors, tests,
// and generated files are excluded to avoid recommending constants for local
// math or structure facts.
func RepeatedPolicyLiterals(snapshot repoanalysis.SourceSnapshot) ([]RawPolicyLiteral, error) {
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
	collectObjects(file, source, out, nil)
}

func collectObjects(file *ast.File, source repoanalysis.GoFile, out *[]Candidate, objects map[*ast.Object]int) {
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
		index := len(*out)
		*out = append(*out, Candidate{
			Kind: closureledger.BindingConstant, Name: declaration.name.Name, File: source.Path,
			Package: filepath.ToSlash(filepath.Dir(source.Path)),
			Scope:   declaration.scope, Line: source.Line(declaration.name.Pos()),
			Expression: expression, Value: evaluated, SourceID: source.ContentID, OwnerID: source.ContentID,
			Policy: true, Doc: declaration.doc, Score: score(declaration.name.Name, declaration.doc, evaluated),
		})
		if objects != nil && declaration.name.Obj != nil {
			objects[declaration.name.Obj] = index
		}
	}
}

func bindCallsites(snapshot repoanalysis.SourceSnapshot, candidates []Candidate, objects map[*ast.Object]int) error {
	byPackage := map[string][]int{}
	packages := map[string]bool{}
	for index, candidate := range candidates {
		if candidate.Kind == closureledger.BindingConstant && candidate.Scope == "package" {
			key := candidate.Package + "\x00" + candidate.Name
			byPackage[key] = append(byPackage[key], index)
			packages[candidate.Package] = true
		}
	}
	hashes := make([]hash.Hash, len(candidates))
	for index := range candidates {
		if candidates[index].Kind == closureledger.BindingConstant {
			hashes[index] = sha256.New()
		}
	}
	err := visitProduction(snapshot, nil, func(source repoanalysis.GoFile, file *ast.File) {
		pkg := filepath.ToSlash(filepath.Dir(source.Path))
		imports := candidateImports(file, packages)
		inspectWithParents(file, func(node ast.Node, parents map[ast.Node]ast.Node) {
			if selector, ok := node.(*ast.SelectorExpr); ok {
				qualifier, ok := selector.X.(*ast.Ident)
				if !ok {
					return
				}
				for _, index := range byPackage[imports[qualifier.Name]+"\x00"+selector.Sel.Name] {
					hashCallsite(hashes[index], source, literalScope(selector, parents), selector.Sel.Pos())
				}
				return
			}
			identifier, ok := node.(*ast.Ident)
			if !ok || identifier == file.Name || identifier.Obj != nil && identifier.Obj.Pos() == identifier.Pos() {
				return
			}
			if identifier.Obj != nil {
				if index, found := objects[identifier.Obj]; found {
					hashCallsite(hashes[index], source, literalScope(identifier, parents), identifier.Pos())
				}
				return
			}
			if _, selector := parents[node].(*ast.SelectorExpr); selector {
				return
			}
			for _, index := range byPackage[pkg+"\x00"+identifier.Name] {
				hashCallsite(hashes[index], source, literalScope(identifier, parents), identifier.Pos())
			}
		})
	})
	if err != nil {
		return err
	}
	for index := range candidates {
		if hashes[index] != nil {
			candidates[index].CallsiteID = hex.EncodeToString(hashes[index].Sum(nil))
		}
	}
	return nil
}

func candidateImports(file *ast.File, packages map[string]bool) map[string]string {
	imports := map[string]string{}
	for _, spec := range file.Imports {
		importPath, err := strconv.Unquote(spec.Path.Value)
		if err != nil {
			continue
		}
		matched := ""
		for candidate := range packages {
			if (importPath == candidate || strings.HasSuffix(importPath, "/"+candidate)) && len(candidate) > len(matched) {
				matched = candidate
			}
		}
		if matched == "" {
			continue
		}
		name := filepath.Base(importPath)
		if spec.Name != nil {
			name = spec.Name.Name
		}
		if name != "." && name != "_" {
			imports[name] = matched
		}
	}
	return imports
}

func hashCallsite(destination hash.Hash, source repoanalysis.GoFile, scope string, _ token.Pos) {
	hashField(destination, source.Path)
	hashField(destination, scope)
}

func assignStructuralIdentities(candidates []Candidate) {
	ordinals := map[string]int{}
	for index := range candidates {
		candidate := &candidates[index]
		base := string(candidate.Kind) + "\x00" + candidate.Package + "\x00" + candidate.File + "\x00" + candidate.Scope
		member := candidate.Name
		if candidate.Kind != closureledger.BindingConstant {
			member = strconv.Itoa(ordinals[base])
			ordinals[base]++
		}
		candidate.StructuralID = digestFields(base, member)
		candidate.SourceID = digestFields(candidate.StructuralID, candidate.Expression)
		if candidate.Kind != closureledger.BindingConstant {
			candidate.CallsiteID = candidate.StructuralID
		}
	}
}

func digestFields(fields ...string) string {
	destination := sha256.New()
	for _, field := range fields {
		hashField(destination, field)
	}
	return hex.EncodeToString(destination.Sum(nil))
}

func hashField(destination hash.Hash, value string) {
	io.WriteString(destination, value)
	destination.Write([]byte{0})
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
