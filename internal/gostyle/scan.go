package gostyle

import (
	"fmt"
	"go/ast"
	"go/token"
	"path"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"overgo/internal/repoanalysis"
)

type receiverFact struct {
	file   repoanalysis.GoFile
	method *ast.FuncDecl
	name   string
	typeID string
}

type fileFacts struct {
	diagnostics map[string][]Diagnostic
	receivers   []receiverFact
	constraint  string
}

type factCache map[string]fileFacts

type fileScanner struct {
	source      repoanalysis.GoFile
	file        *ast.File
	imports     map[string]string
	diagnostics map[string][]Diagnostic
	receivers   []receiverFact
}

// Census reports one snapshot. Compare uses the same implementation with a
// shared content cache so unchanged candidate files reuse base facts.
func Census(snapshot repoanalysis.SourceSnapshot, options ...Options) (CensusReport, error) {
	return census(snapshot, firstOption(options), factCache{})
}

func census(snapshot repoanalysis.SourceSnapshot, option Options, cache factCache) (CensusReport, error) {
	started := time.Now()
	if err := validatePolicy(); err != nil {
		return CensusReport{}, err
	}
	report := CensusReport{
		Identity: snapshot.Identity(), BuildContext: option.Build.Context,
		Rules: make([]RuleCensus, len(policy)),
	}
	results := make(map[string]*RuleCensus, len(policy))
	for index, rule := range policy {
		selected := rule.Mechanism == Syntax
		report.Rules[index] = RuleCensus{Rule: rule, Selected: selected}
		if !selected {
			report.Rules[index].SkipReason = unavailableReason(rule.Mechanism)
		}
		results[rule.ID] = &report.Rules[index]
	}
	var receivers []receiverFact
	for _, source := range snapshot.Files {
		selected, classified := option.Build.Files[source.Path]
		if classified && !selected {
			report.Exclusions = append(report.Exclusions, Exclusion{
				File: source.Path, Scope: "host build context", Reason: "excluded by go list for " + option.Build.Context,
			})
			report.BuildConstraints = append(report.BuildConstraints, BuildConstraint{
				File: source.Path, Expression: "go list selection", Selected: false,
			})
			continue
		}
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
		key := source.Path + "\x00" + source.ContentID
		facts, reused := cache[key]
		if reused {
			report.Analysis.FilesReused++
		} else {
			facts, err = analyzeFile(source)
			if err != nil {
				return CensusReport{}, err
			}
			cache[key] = facts
			report.Analysis.FilesScanned++
			report.Analysis.SyntaxPasses++
		}
		if facts.constraint != "" {
			report.BuildConstraints = append(report.BuildConstraints, BuildConstraint{
				File: source.Path, Expression: facts.constraint, Selected: true,
			})
		}
		for ruleID, diagnostics := range facts.diagnostics {
			results[ruleID].Diagnostics = append(results[ruleID].Diagnostics, diagnostics...)
		}
		receivers = append(receivers, facts.receivers...)
	}
	receiverConsistency(results["receivers"], receivers)
	for index := range report.Rules {
		sortDiagnostics(report.Rules[index].Diagnostics)
		report.Analysis.Diagnostics += len(report.Rules[index].Diagnostics)
	}
	report.Analysis.Duration = time.Since(started)
	return report, nil
}

func firstOption(options []Options) Options {
	if len(options) == 0 {
		return Options{}
	}
	return options[0]
}

func unavailableReason(mechanism Mechanism) string {
	switch mechanism {
	case Toolchain:
		return "selected by the toolchain adapter; syntax census does not imitate it"
	case TypeAware:
		return "reserved for the shared type-aware pass; syntax-only guesses are not evidence"
	case Delegated:
		return "reported by the existing structural analysis owner"
	default:
		return "unsupported analysis mechanism"
	}
}

func analyzeFile(source repoanalysis.GoFile) (fileFacts, error) {
	file, err := source.Syntax()
	if err != nil {
		return fileFacts{}, fmt.Errorf("parse %s: %w", source.Path, err)
	}
	constraint, err := source.BuildExpression()
	if err != nil {
		return fileFacts{}, fmt.Errorf("build constraint %s: %w", source.Path, err)
	}
	scanner := fileScanner{
		source: source, file: file, imports: importBindings(file), diagnostics: map[string][]Diagnostic{},
	}
	scanner.packageImports()
	scanner.declarationNames()
	for _, declaration := range file.Decls {
		scanner.declaration(declaration)
	}
	return fileFacts{diagnostics: scanner.diagnostics, receivers: scanner.receivers, constraint: constraint}, nil
}

func importBindings(file *ast.File) map[string]string {
	bindings := make(map[string]string, len(file.Imports))
	for _, spec := range file.Imports {
		importPath, err := strconv.Unquote(spec.Path.Value)
		if err != nil {
			continue
		}
		name := path.Base(importPath)
		if spec.Name != nil {
			name = spec.Name.Name
		}
		if name != "." && name != "_" {
			bindings[name] = importPath
		}
	}
	return bindings
}

func (s *fileScanner) packageImports() {
	packageName := s.file.Name.Name
	baseName := strings.TrimSuffix(packageName, "_test")
	if baseName == "" || strings.ToLower(baseName) != baseName || strings.Contains(baseName, "_") {
		s.add("package-imports", s.file.Name, packageName, "package name is not lowercase and concise")
	}
	for _, spec := range s.file.Imports {
		if spec.Name == nil {
			continue
		}
		if spec.Name.Name == "." {
			s.add("package-imports", spec, "", "dot import obscures package ownership")
		}
		importPath, _ := strconv.Unquote(spec.Path.Value)
		if spec.Name.Name == "_" && packageName != "main" && !s.source.Test && importPath != "embed" {
			s.add("package-imports", spec, "", "blank import is outside main, tests, or compiler-defined embed use")
		}
	}
}

func (s *fileScanner) declarationNames() {
	exceptions := map[token.Pos]bool{}
	for _, declaration := range s.file.Decls {
		if function, ok := declaration.(*ast.FuncDecl); ok && s.source.Test && isTestEntry(function.Name.Name) {
			exceptions[function.Name.Pos()] = true
		}
	}
	cgo := s.imports["C"] == "C"
	seen := map[token.Pos]bool{}
	ast.Inspect(s.file, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.Ident:
			if node.Obj != nil && node.Obj.Pos() == node.Pos() && !seen[node.Pos()] {
				seen[node.Pos()] = true
				s.mixedCaps(node, exceptions[node.Pos()] || cgo)
			}
		case *ast.Field:
			for _, name := range node.Names {
				if !seen[name.Pos()] {
					seen[name.Pos()] = true
					s.mixedCaps(name, false)
				}
			}
		case *ast.ImportSpec:
			if node.Name != nil && node.Name.Name != "." && node.Name.Name != "_" {
				s.mixedCaps(node.Name, false)
			}
		}
		return true
	})
}

func (s *fileScanner) declaration(declaration ast.Decl) {
	switch declaration := declaration.(type) {
	case *ast.GenDecl:
		if declaration.Tok == token.VAR {
			s.add("global-state", declaration, "", "package-level mutable state requires ownership review")
		}
		for _, spec := range declaration.Specs {
			if typeSpec, ok := spec.(*ast.TypeSpec); ok {
				if _, ok := typeSpec.Type.(*ast.InterfaceType); ok {
					s.add("interface-ownership", typeSpec, typeSpec.Name.Name, "interface declaration requires demonstrated consumer ownership")
				}
				if structure, ok := typeSpec.Type.(*ast.StructType); ok {
					for _, field := range structure.Fields.List {
						if s.qualifiedSelector(field.Type, "context", "Context") {
							s.add("context", field, fieldName(field), "context.Context stored in a struct")
						}
					}
				}
			}
		}
		ast.Inspect(declaration, func(node ast.Node) bool { s.node(node, ""); return true })
	case *ast.FuncDecl:
		s.function(declaration)
	}
}

func (s *fileScanner) function(function *ast.FuncDecl) {
	if strings.HasPrefix(function.Name.Name, "Get") && function.Name.Name != "Get" {
		s.add("get-prefix", function.Name, function.Name.Name, "Get-prefixed API may hide cost or blocking")
	}
	var receiverObject *ast.Object
	if function.Recv != nil && len(function.Recv.List) > 0 {
		field := function.Recv.List[0]
		name := ""
		if len(field.Names) > 0 {
			name, receiverObject = field.Names[0].Name, field.Names[0].Obj
		}
		typeID := path.Dir(s.source.Path) + "/" + receiverType(field.Type)
		s.receivers = append(s.receivers, receiverFact{s.source, function, name, typeID})
		if name == "this" || name == "self" || name == "receiver" {
			s.add("receivers", field, function.Name.Name, "receiver name is needlessly verbose")
		}
	}
	parameters := fieldList(function.Type.Params)
	for index, field := range parameters {
		if s.qualifiedSelector(field.Type, "context", "Context") && index != 0 {
			s.add("context", field, function.Name.Name, "context.Context is not the first parameter")
		}
	}
	results := fieldList(function.Type.Results)
	for index, field := range results {
		if builtinIdent(field.Type, "error") && index != len(results)-1 {
			s.add("errors", field, function.Name.Name, "error is not the final return")
		}
	}
	if function.Body == nil {
		return
	}
	facts := struct{ receiverUsed, nakedReturn, testFailure, testHelper bool }{}
	symbol := functionSymbol(function)
	ast.Inspect(function.Body, func(node ast.Node) bool {
		if literal, nested := node.(*ast.FuncLit); nested {
			s.scanNested(literal.Body, symbol)
			return false
		}
		if identifier, ok := node.(*ast.Ident); ok && receiverObject != nil && identifier.Obj == receiverObject {
			facts.receiverUsed = true
		}
		if statement, ok := node.(*ast.ReturnStmt); ok && len(statement.Results) == 0 {
			facts.nakedReturn = true
		}
		if call, ok := node.(*ast.CallExpr); ok {
			switch lastName(callName(call.Fun)) {
			case "Fatal", "Fatalf", "Fail", "FailNow", "Error", "Errorf":
				facts.testFailure = true
			case "Helper":
				facts.testHelper = true
			}
		}
		s.node(node, symbol)
		return true
	})
	if receiverObject != nil && !facts.receiverUsed {
		s.add("receivers", function.Recv.List[0], function.Name.Name, "unused receiver should be unnamed")
	}
	if len(function.Body.List) > 5 && facts.nakedReturn {
		s.add("returns-copy", function.Name, function.Name.Name, "nontrivial function uses a naked return")
	}
	if s.source.Test && !isTestEntry(function.Name.Name) && s.testingParameter(parameters) && facts.testFailure && !facts.testHelper {
		s.add("test-quality", function.Name, function.Name.Name, "test helper can fail without calling testing.TB.Helper")
	}
}

func (s *fileScanner) scanNested(body *ast.BlockStmt, symbol string) {
	ast.Inspect(body, func(node ast.Node) bool {
		s.node(node, symbol)
		return true
	})
}

func (s *fileScanner) node(node ast.Node, symbol string) {
	switch node := node.(type) {
	case *ast.CallExpr:
		qualified := s.qualifiedCall(node.Fun)
		name := callName(node.Fun)
		switch {
		case s.file.Name.Name != "main" && flagRegistration(qualified):
			s.add("library-flags", node, symbol, "importable package registers a command-line flag")
		case s.file.Name.Name != "main" && qualified == "context.Background":
			s.add("background-context", node, symbol, "library call chain starts a background context")
		case name == "panic" && builtinCall(node.Fun) || strings.HasPrefix(lastName(name), "Must"):
			s.add("panic-must", node, symbol, "panic-style call requires initialization or invariant ownership")
		case qualified == "reflect.DeepEqual" && s.source.Test:
			s.add("test-quality", node, symbol, "reflect.DeepEqual can hide useful got/want evidence")
		case (qualified == "errors.New" || qualified == "fmt.Errorf") && len(node.Args) > 0:
			if literal, ok := node.Args[0].(*ast.BasicLit); ok && literal.Kind == token.STRING {
				message, err := strconv.Unquote(literal.Value)
				if err == nil && unconventionalError(message) {
					s.add("errors", literal, symbol, "static error string starts uppercase or ends with punctuation")
				}
			}
		}
	case *ast.GoStmt:
		s.add("goroutine-ownership", node, symbol, "goroutine requires visible cancellation, join, or process-lifetime ownership")
	}
}

func (s *fileScanner) qualifiedSelector(expression ast.Expr, importPath, name string) bool {
	selector, ok := expression.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	owner, ownerOK := selector.X.(*ast.Ident)
	return ownerOK && owner.Obj == nil && s.imports[owner.Name] == importPath && selector.Sel.Name == name
}

func (s *fileScanner) qualifiedCall(expression ast.Expr) string {
	selector, ok := expression.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	owner, ok := selector.X.(*ast.Ident)
	if !ok || owner.Obj != nil {
		return ""
	}
	if importPath := s.imports[owner.Name]; importPath != "" {
		return importPath + "." + selector.Sel.Name
	}
	return ""
}

func (s *fileScanner) testingParameter(fields []*ast.Field) bool {
	for _, field := range fields {
		expression := field.Type
		if pointer, ok := expression.(*ast.StarExpr); ok {
			expression = pointer.X
		}
		if s.qualifiedSelector(expression, "testing", "T") || s.qualifiedSelector(expression, "testing", "B") || s.qualifiedSelector(expression, "testing", "TB") {
			return true
		}
	}
	return false
}

func (s *fileScanner) mixedCaps(name *ast.Ident, exception bool) {
	if !exception && name.Name != "_" && !strings.HasPrefix(name.Name, "_C") && strings.Contains(name.Name, "_") {
		s.add("mixed-caps", name, name.Name, "identifier uses underscore-separated words")
	}
}

func (s *fileScanner) add(ruleID string, node ast.Node, symbol, message string) {
	s.diagnostics[ruleID] = append(s.diagnostics[ruleID], Diagnostic{
		File: s.source.Path, Line: s.source.Line(node.Pos()), Symbol: symbol, Message: message,
	})
}

func receiverConsistency(result *RuleCensus, receivers []receiverFact) {
	byType := map[string][]receiverFact{}
	for _, receiver := range receivers {
		byType[receiver.typeID] = append(byType[receiver.typeID], receiver)
	}
	for typeID, methods := range byType {
		names := map[string]bool{}
		for _, method := range methods {
			if method.name != "" {
				names[method.name] = true
			}
		}
		if len(names) < 2 {
			continue
		}
		for _, method := range methods {
			result.Diagnostics = append(result.Diagnostics, Diagnostic{
				File: method.file.Path, Line: method.file.Line(method.method.Pos()), Symbol: method.method.Name.Name,
				Message: "receiver name is inconsistent across methods on " + typeID,
			})
		}
	}
}

func isTestEntry(name string) bool {
	for _, prefix := range []string{"Test", "Benchmark", "Fuzz", "Example"} {
		if strings.HasPrefix(name, prefix) && (len(name) == len(prefix) || !unicode.IsLower(rune(name[len(prefix)]))) {
			return true
		}
	}
	return false
}

func fieldList(list *ast.FieldList) []*ast.Field {
	if list == nil {
		return nil
	}
	var fields []*ast.Field
	for _, field := range list.List {
		count := max(1, len(field.Names))
		for range count {
			fields = append(fields, field)
		}
	}
	return fields
}

func builtinIdent(expression ast.Expr, name string) bool {
	identifier, ok := expression.(*ast.Ident)
	return ok && identifier.Name == name && identifier.Obj == nil
}

func builtinCall(expression ast.Expr) bool {
	identifier, ok := expression.(*ast.Ident)
	return ok && identifier.Obj == nil
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
	first, _ := utf8.DecodeRuneInString(message)
	last, _ := utf8.DecodeLastRuneInString(message)
	return unicode.IsUpper(first) || strings.ContainsRune(".:;!?", last)
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
		return left.Message < right.Message
	})
}
