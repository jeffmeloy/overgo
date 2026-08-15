package gostyle

import (
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/token"
	"path"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"overgo/internal/repoanalysis"
)

type scanner struct {
	report    *CensusReport
	result    map[string]*RuleCensus
	receivers map[string][]receiver
}

type receiver struct {
	file   repoanalysis.GoFile
	method *ast.FuncDecl
	name   string
	typeID string
}

// Census parses no source itself: all syntax and generated-file decisions come
// from the shared repository snapshot.
func Census(snapshot repoanalysis.SourceSnapshot) (CensusReport, error) {
	report := CensusReport{Identity: snapshot.Identity(), Rules: make([]RuleCensus, len(policy))}
	scan := scanner{report: &report, result: map[string]*RuleCensus{}, receivers: map[string][]receiver{}}
	for index, rule := range policy {
		report.Rules[index] = RuleCensus{Rule: rule, Available: rule.Mechanism == Syntax}
		if !report.Rules[index].Available {
			report.Rules[index].SkipReason = unavailableReason(rule.Mechanism)
		}
		scan.result[rule.ID] = &report.Rules[index]
	}
	for _, source := range snapshot.Files {
		generated, err := source.Generated()
		if err != nil {
			return CensusReport{}, fmt.Errorf("parse %s: %w", source.Path, err)
		}
		if generated {
			report.Exclusions = append(report.Exclusions, Exclusion{
				File: source.Path, Scope: "syntax policy", Reason: "generated source is owned by its generator; toolchain formatting remains applicable",
			})
			continue
		}
		file, _ := source.Syntax()
		if expression, err := buildExpression(file); err != nil {
			return CensusReport{}, fmt.Errorf("build constraint %s: %w", source.Path, err)
		} else if expression != "" {
			report.BuildConstraints = append(report.BuildConstraints, BuildConstraint{File: source.Path, Expression: expression})
		}
		scan.file(source, file)
	}
	scan.receiverConsistency()
	for index := range report.Rules {
		diagnostics := report.Rules[index].Diagnostics
		sortDiagnostics(diagnostics)
	}
	return report, nil
}

func unavailableReason(mechanism Mechanism) string {
	switch mechanism {
	case Toolchain:
		return "recorded by the toolchain adapter; syntax census does not imitate it"
	case TypeAware:
		return "reserved for the shared type-aware pass; syntax-only guesses are not evidence"
	case Delegated:
		return "reported by the existing structural analysis owner"
	default:
		return "unsupported analysis mechanism"
	}
}

func buildExpression(file *ast.File) (string, error) {
	for _, group := range file.Comments {
		if group.End() > file.Package {
			break
		}
		for _, comment := range group.List {
			if constraint.IsGoBuild(comment.Text) {
				expression, err := constraint.Parse(comment.Text)
				if err != nil {
					return "", err
				}
				return expression.String(), nil
			}
		}
	}
	return "", nil
}

func (s *scanner) file(source repoanalysis.GoFile, file *ast.File) {
	s.packageImports(source, file)
	for _, declaration := range file.Decls {
		s.declaration(source, file, declaration)
	}
}

func (s *scanner) packageImports(source repoanalysis.GoFile, file *ast.File) {
	packageName := file.Name.Name
	baseName := strings.TrimSuffix(packageName, "_test")
	if baseName == "" || strings.ToLower(baseName) != baseName || strings.Contains(baseName, "_") {
		s.add("package-imports", source, file.Name, packageName, "package name is not lowercase and concise")
	}
	for _, spec := range file.Imports {
		if spec.Name == nil {
			continue
		}
		if spec.Name.Name == "." {
			s.add("package-imports", source, spec, "", "dot import obscures package ownership")
		}
		path, _ := strconv.Unquote(spec.Path.Value)
		if spec.Name.Name == "_" && packageName != "main" && !source.Test && path != "embed" {
			s.add("package-imports", source, spec, "", "blank import is outside main, tests, or compiler-defined embed use")
		}
	}
}

func (s *scanner) declaration(source repoanalysis.GoFile, file *ast.File, declaration ast.Decl) {
	switch declaration := declaration.(type) {
	case *ast.GenDecl:
		if declaration.Tok == token.VAR {
			s.add("global-state", source, declaration, "", "package-level mutable state requires ownership review")
		}
		for _, spec := range declaration.Specs {
			switch spec := spec.(type) {
			case *ast.TypeSpec:
				s.mixedCaps(source, spec.Name)
				if _, ok := spec.Type.(*ast.InterfaceType); ok {
					s.add("interface-ownership", source, spec, spec.Name.Name, "interface declaration requires demonstrated consumer ownership")
				}
				if structure, ok := spec.Type.(*ast.StructType); ok {
					for _, field := range structure.Fields.List {
						if isSelector(field.Type, "context", "Context") {
							s.add("context", source, field, fieldName(field), "context.Context stored in a struct")
						}
					}
				}
			case *ast.ValueSpec:
				for _, name := range spec.Names {
					s.mixedCaps(source, name)
				}
			}
		}
		ast.Inspect(declaration, func(node ast.Node) bool {
			s.node(source, file, node, "")
			return true
		})
	case *ast.FuncDecl:
		s.function(source, file, declaration)
	}
}

func (s *scanner) function(source repoanalysis.GoFile, file *ast.File, function *ast.FuncDecl) {
	if !source.Test || (!strings.HasPrefix(function.Name.Name, "Test") && !strings.HasPrefix(function.Name.Name, "Benchmark") && !strings.HasPrefix(function.Name.Name, "Example")) {
		s.mixedCaps(source, function.Name)
	}
	if strings.HasPrefix(function.Name.Name, "Get") && function.Name.Name != "Get" {
		s.add("get-prefix", source, function.Name, function.Name.Name, "Get-prefixed API may hide cost or blocking")
	}
	if function.Recv != nil && len(function.Recv.List) > 0 {
		field := function.Recv.List[0]
		name := ""
		if len(field.Names) > 0 {
			name = field.Names[0].Name
		}
		typeID := path.Dir(source.Path) + "/" + receiverType(field.Type)
		s.receivers[typeID] = append(s.receivers[typeID], receiver{source, function, name, typeID})
		if name == "this" || name == "self" || name == "receiver" {
			s.add("receivers", source, field, function.Name.Name, "receiver name is needlessly verbose")
		}
		if name != "" && function.Body != nil && !identUsed(function.Body, name) {
			s.add("receivers", source, field, function.Name.Name, "unused receiver should be unnamed")
		}
	}
	parameters := fieldList(function.Type.Params)
	for index, field := range parameters {
		if isSelector(field.Type, "context", "Context") && index != 0 {
			s.add("context", source, field, function.Name.Name, "context.Context is not the first parameter")
		}
	}
	results := fieldList(function.Type.Results)
	for index, field := range results {
		if isIdent(field.Type, "error") && index != len(results)-1 {
			s.add("errors", source, field, function.Name.Name, "error is not the final return")
		}
		if function.Name.IsExported() && pointerError(field.Type) {
			s.add("errors", source, field, function.Name.Name, "exported API returns a concrete error pointer")
		}
	}
	if function.Body != nil && len(function.Body.List) > 5 && hasNakedReturn(function.Body) {
		s.add("returns-copy", source, function.Name, function.Name.Name, "nontrivial function uses a naked return")
	}
	if source.Test && !isTestEntry(function.Name.Name) && testingParameter(parameters) && callsFailureWithoutHelper(function.Body) {
		s.add("test-quality", source, function.Name, function.Name.Name, "test helper can fail without calling testing.TB.Helper")
	}
	if function.Body != nil {
		symbol := functionSymbol(function)
		ast.Inspect(function.Body, func(node ast.Node) bool {
			s.node(source, file, node, symbol)
			return true
		})
	}
}

func (s *scanner) node(source repoanalysis.GoFile, file *ast.File, node ast.Node, symbol string) {
	switch node := node.(type) {
	case *ast.CallExpr:
		name := callName(node.Fun)
		switch {
		case file.Name.Name != "main" && flagRegistration(name):
			s.add("library-flags", source, node, symbol, "importable package registers a command-line flag")
		case file.Name.Name != "main" && name == "context.Background":
			s.add("background-context", source, node, symbol, "library call chain starts a background context")
		case name == "panic" || strings.HasPrefix(lastName(name), "Must"):
			s.add("panic-must", source, node, symbol, "panic-style call requires initialization or invariant ownership")
		case name == "reflect.DeepEqual" && source.Test:
			s.add("test-quality", source, node, symbol, "reflect.DeepEqual can hide useful got/want evidence")
		case (name == "errors.New" || name == "fmt.Errorf") && len(node.Args) > 0:
			if literal, ok := node.Args[0].(*ast.BasicLit); ok && literal.Kind == token.STRING {
				message, err := strconv.Unquote(literal.Value)
				if err == nil && unconventionalError(message) {
					s.add("errors", source, literal, symbol, "static error string starts uppercase or ends with punctuation")
				}
			}
		}
	case *ast.GoStmt:
		s.add("goroutine-ownership", source, node, symbol, "goroutine requires visible cancellation, join, or process-lifetime ownership")
	}
}

func (s *scanner) receiverConsistency() {
	for _, receivers := range s.receivers {
		names := map[string]bool{}
		for _, receiver := range receivers {
			if receiver.name != "" {
				names[receiver.name] = true
			}
		}
		if len(names) < 2 {
			continue
		}
		for _, receiver := range receivers {
			s.add("receivers", receiver.file, receiver.method, receiver.method.Name.Name, "receiver name is inconsistent across methods on "+receiver.typeID)
		}
	}
}

func (s *scanner) mixedCaps(source repoanalysis.GoFile, name *ast.Ident) {
	if name.Name != "_" && !strings.HasPrefix(name.Name, "_C") && strings.Contains(name.Name, "_") {
		s.add("mixed-caps", source, name, name.Name, "identifier uses underscore-separated words")
	}
}

func isTestEntry(name string) bool {
	return strings.HasPrefix(name, "Test") || strings.HasPrefix(name, "Benchmark") || strings.HasPrefix(name, "Fuzz") || strings.HasPrefix(name, "Example")
}

func (s *scanner) add(ruleID string, source repoanalysis.GoFile, node ast.Node, symbol, rationale string) {
	result := s.result[ruleID]
	result.Diagnostics = append(result.Diagnostics, Diagnostic{
		RuleID: ruleID, Tier: result.Rule.Tier, SourceSection: result.Rule.SourceSection,
		File: source.Path, Line: source.Line(node.Pos()), Symbol: symbol, Rationale: rationale,
		DeterministicFix: result.Rule.DeterministicFix,
	})
}

func fieldList(list *ast.FieldList) []*ast.Field {
	if list == nil {
		return nil
	}
	var fields []*ast.Field
	for _, field := range list.List {
		count := len(field.Names)
		if count == 0 {
			count = 1
		}
		for range count {
			fields = append(fields, field)
		}
	}
	return fields
}

func isSelector(expression ast.Expr, packageName, name string) bool {
	selector, ok := expression.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	owner, ownerOK := selector.X.(*ast.Ident)
	return ownerOK && owner.Name == packageName && selector.Sel.Name == name
}

func isIdent(expression ast.Expr, name string) bool {
	identifier, ok := expression.(*ast.Ident)
	return ok && identifier.Name == name
}

func pointerError(expression ast.Expr) bool {
	pointer, ok := expression.(*ast.StarExpr)
	if !ok {
		return false
	}
	name := lastName(callName(pointer.X))
	return strings.HasSuffix(name, "Error") || strings.HasSuffix(name, "Err")
}

func fieldName(field *ast.Field) string {
	if len(field.Names) == 0 {
		return ""
	}
	return field.Names[0].Name
}

func receiverType(expression ast.Expr) string {
	if pointer, ok := expression.(*ast.StarExpr); ok {
		expression = pointer.X
	}
	return callName(expression)
}

func identUsed(node ast.Node, name string) bool {
	used := false
	ast.Inspect(node, func(node ast.Node) bool {
		if identifier, ok := node.(*ast.Ident); ok && identifier.Name == name {
			used = true
		}
		return !used
	})
	return used
}

func callName(expression ast.Expr) string {
	switch expression := expression.(type) {
	case *ast.Ident:
		return expression.Name
	case *ast.SelectorExpr:
		owner := callName(expression.X)
		if owner == "" {
			return expression.Sel.Name
		}
		return owner + "." + expression.Sel.Name
	case *ast.IndexExpr:
		return callName(expression.X)
	case *ast.IndexListExpr:
		return callName(expression.X)
	}
	return ""
}

func functionSymbol(function *ast.FuncDecl) string {
	if function.Recv == nil || len(function.Recv.List) == 0 {
		return function.Name.Name
	}
	return receiverType(function.Recv.List[0].Type) + "." + function.Name.Name
}

func flagRegistration(name string) bool {
	if !strings.HasPrefix(name, "flag.") {
		return false
	}
	switch lastName(name) {
	case "Bool", "BoolFunc", "BoolVar", "Duration", "DurationVar", "Float64", "Float64Var", "Func", "Int", "Int64", "Int64Var", "IntVar", "String", "StringVar", "TextVar", "Uint", "Uint64", "Uint64Var", "UintVar", "Var":
		return true
	}
	return false
}

func lastName(name string) string {
	if index := strings.LastIndexByte(name, '.'); index >= 0 {
		return name[index+1:]
	}
	return name
}

func unconventionalError(message string) bool {
	if message == "" {
		return false
	}
	first, _ := utf8Rune(message)
	last, _ := utf8LastRune(message)
	return unicode.IsUpper(first) || strings.ContainsRune(".:;!?", last)
}

func utf8Rune(value string) (rune, int) {
	for _, r := range value {
		return r, len(string(r))
	}
	return 0, 0
}

func utf8LastRune(value string) (rune, int) {
	var last rune
	for _, r := range value {
		last = r
	}
	return last, len(string(last))
}

func hasNakedReturn(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(node ast.Node) bool {
		if statement, ok := node.(*ast.ReturnStmt); ok && len(statement.Results) == 0 {
			found = true
		}
		return !found
	})
	return found
}

func testingParameter(fields []*ast.Field) bool {
	for _, field := range fields {
		expression := field.Type
		if pointer, ok := expression.(*ast.StarExpr); ok {
			expression = pointer.X
		}
		if isSelector(expression, "testing", "T") || isSelector(expression, "testing", "B") || isSelector(expression, "testing", "TB") {
			return true
		}
	}
	return false
}

func callsFailureWithoutHelper(body *ast.BlockStmt) bool {
	if body == nil {
		return false
	}
	fails, helper := false, false
	ast.Inspect(body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		switch lastName(callName(call.Fun)) {
		case "Fatal", "Fatalf", "Fail", "FailNow", "Error", "Errorf":
			fails = true
		case "Helper":
			helper = true
		}
		return true
	})
	return fails && !helper
}

func sortDiagnostics(diagnostics []Diagnostic) {
	sort.Slice(diagnostics, func(i, j int) bool {
		left, right := diagnostics[i], diagnostics[j]
		if left.File != right.File {
			return left.File < right.File
		}
		if left.Line != right.Line {
			return left.Line < right.Line
		}
		if left.Symbol != right.Symbol {
			return left.Symbol < right.Symbol
		}
		return left.Rationale < right.Rationale
	})
}
