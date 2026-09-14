package repoanalysis

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"path"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

type goStyleKind string

const (
	goStyleFormat              goStyleKind = "format"
	goStylePackageDoc          goStyleKind = "package-doc"
	goStyleExportDoc           goStyleKind = "export-doc"
	goStyleReceiverConsistency goStyleKind = "receiver-consistency"
	goStyleInitialism          goStyleKind = "initialism"
	goStyleGetter              goStyleKind = "getter"
	goStyleNamedResult         goStyleKind = "named-result"
	goStyleNakedReturn         goStyleKind = "naked-return"
	goStyleErrorText           goStyleKind = "error-text"
	goStyleImportAlias         goStyleKind = "import-alias"
)

type goStyleFinding struct {
	Kind   goStyleKind
	File   string
	Line   int
	Symbol string
	Detail string
}

type goStyleReport struct {
	Source   string
	Files    int
	Excluded int
	Findings []goStyleFinding
}

// ValidateGoStyleDelta rejects objective findings introduced by candidate.
func ValidateGoStyleDelta(candidate, baseline SourceSnapshot) error {
	current, err := goStyleCensus(candidate)
	if err != nil {
		return err
	}
	prior, err := goStyleCensus(baseline)
	if err != nil {
		return err
	}
	known := make(map[string]bool, len(prior.Findings))
	for _, finding := range prior.Findings {
		known[finding.key()] = true
	}
	var introduced []goStyleFinding
	for _, finding := range current.Findings {
		if !known[finding.key()] {
			introduced = append(introduced, finding)
		}
	}
	if len(introduced) == 0 {
		return nil
	}
	parts := make([]string, 0, len(introduced))
	for _, finding := range introduced {
		parts = append(parts, fmt.Sprintf("%s:%d:%s:%s", finding.File, finding.Line, finding.Kind, finding.Symbol))
	}
	return fmt.Errorf("new Go style findings: %s", strings.Join(parts, ", "))
}

func goStyleCensus(snapshot SourceSnapshot) (goStyleReport, error) {
	report := goStyleReport{Source: snapshot.Identity()}
	packages := map[string][]GoFile{}
	receivers := map[string]map[string]goStyleFinding{}
	aliases := map[string]map[string]goStyleFinding{}
	for _, source := range snapshot.Files {
		generated, err := source.Generated()
		if err != nil {
			return goStyleReport{}, err
		}
		if generated {
			report.Excluded++
			continue
		}
		report.Files++
		file, err := source.Syntax()
		if err != nil {
			return goStyleReport{}, err
		}
		packageKey := packageStyleKey(source.Path, file.Name.Name)
		packages[packageKey] = append(packages[packageKey], source)
		if formatted, err := format.Source(source.blob.data); err != nil || !bytes.Equal(formatted, source.blob.data) {
			report.add(source, goStyleFormat, file.Package, file.Name.Name, "source differs from gofmt")
		}
		inspectStyleFile(&report, source, file, receivers, aliases)
	}
	for key, sources := range packages {
		var documented *GoFile
		duplicate := false
		for _, source := range sources {
			file, _ := source.Syntax()
			if file.Doc != nil && strings.TrimSpace(file.Doc.Text()) != "" {
				if documented != nil {
					duplicate = true
				}
				current := source
				documented = &current
			}
		}
		if documented == nil || duplicate {
			source := sources[0]
			detail := "package comment is absent"
			if duplicate {
				source, detail = *documented, "package has multiple comments"
			}
			report.add(source, goStylePackageDoc, token.NoPos, key, detail)
		}
	}
	appendInconsistentNames(&report, receivers, goStyleReceiverConsistency, "receiver")
	appendInconsistentNames(&report, aliases, goStyleImportAlias, "import alias")
	sort.Slice(report.Findings, func(i, j int) bool {
		left, right := report.Findings[i], report.Findings[j]
		if left.File != right.File {
			return left.File < right.File
		}
		if left.Line != right.Line {
			return left.Line < right.Line
		}
		if left.Kind != right.Kind {
			return left.Kind < right.Kind
		}
		return left.Symbol < right.Symbol
	})
	return report, nil
}

func (f goStyleFinding) key() string {
	return strings.Join([]string{string(f.Kind), f.File, f.Symbol, f.Detail}, "\x00")
}

func inspectStyleFile(
	report *goStyleReport,
	source GoFile,
	file *ast.File,
	receivers, aliases map[string]map[string]goStyleFinding,
) {
	for _, imported := range file.Imports {
		if imported.Name == nil || imported.Name.Name == "_" || imported.Name.Name == "." {
			continue
		}
		path, err := strconv.Unquote(imported.Path.Value)
		if err != nil {
			continue
		}
		rememberStyleName(aliases, path, imported.Name.Name, report.finding(source, goStyleImportAlias, imported.Pos(), path, imported.Name.Name))
	}
	for _, declaration := range file.Decls {
		switch value := declaration.(type) {
		case *ast.FuncDecl:
			inspectStyleFunction(report, source, value, receivers)
		case *ast.GenDecl:
			if !source.Test {
				inspectStyleDeclaration(report, source, value)
			}
		}
	}
}

func inspectStyleFunction(report *goStyleReport, source GoFile, function *ast.FuncDecl, receivers map[string]map[string]goStyleFinding) {
	name := function.Name.Name
	if !source.Test && ast.IsExported(name) && !validDoc(function.Doc, name) {
		report.add(source, goStyleExportDoc, function.Pos(), name, "exported function needs a name-led sentence")
	}
	if getterMethod(function) {
		report.add(source, goStyleGetter, function.Name.Pos(), name, "getter prefix obscures the result noun")
	}
	if bad := nonCanonicalInitialism(name); bad != "" {
		report.add(source, goStyleInitialism, function.Name.Pos(), name, "noncanonical initialism "+bad)
	}
	if function.Recv != nil && len(function.Recv.List) != 0 && len(function.Recv.List[0].Names) != 0 {
		receiver := function.Recv.List[0]
		typeName := receiverTypeName(receiver.Type)
		if typeName != "" {
			name := receiver.Names[0].Name
			rememberStyleName(receivers, packageStyleKey(source.Path, typeName), name,
				report.finding(source, goStyleReceiverConsistency, receiver.Pos(), typeName, name))
		}
	}
	if function.Body == nil {
		return
	}
	used := map[string]bool{}
	for node := range ast.Preorder(function.Body) {
		if identifier, ok := node.(*ast.Ident); ok {
			used[identifier.Name] = true
		}
	}
	ast.Inspect(function.Body, func(node ast.Node) bool {
		switch value := node.(type) {
		case *ast.FuncLit:
			return false
		case *ast.ReturnStmt:
			if len(value.Results) == 0 && function.Type.Results != nil {
				report.add(source, goStyleNakedReturn, value.Pos(), name, "function uses a naked return")
			}
		case *ast.CallExpr:
			inspectErrorText(report, source, value, name)
		}
		return true
	})
	if function.Type.Results != nil {
		for _, result := range function.Type.Results.List {
			for _, resultName := range result.Names {
				if resultName.Name == "result" && !used[resultName.Name] {
					report.add(source, goStyleNamedResult, resultName.Pos(), name, "named result is unused by the function body")
				}
			}
		}
	}
}

func inspectStyleDeclaration(report *goStyleReport, source GoFile, declaration *ast.GenDecl) {
	for _, specification := range declaration.Specs {
		var names []*ast.Ident
		var doc *ast.CommentGroup
		switch value := specification.(type) {
		case *ast.TypeSpec:
			names, doc = []*ast.Ident{value.Name}, value.Doc
		case *ast.ValueSpec:
			names, doc = value.Names, value.Doc
		}
		if doc == nil {
			doc = declaration.Doc
		}
		for _, name := range names {
			if ast.IsExported(name.Name) && !validDoc(doc, name.Name) {
				report.add(source, goStyleExportDoc, name.Pos(), name.Name, "exported declaration needs a name-led sentence")
			}
			if bad := nonCanonicalInitialism(name.Name); bad != "" {
				report.add(source, goStyleInitialism, name.Pos(), name.Name, "noncanonical initialism "+bad)
			}
		}
	}
}

func inspectErrorText(report *goStyleReport, source GoFile, call *ast.CallExpr, scope string) {
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || len(call.Args) == 0 || (selector.Sel.Name != "New" && selector.Sel.Name != "Errorf") {
		return
	}
	packageName, ok := selector.X.(*ast.Ident)
	if !ok || packageName.Name != "errors" && packageName.Name != "fmt" {
		return
	}
	literal, ok := call.Args[0].(*ast.BasicLit)
	if !ok || literal.Kind != token.STRING {
		return
	}
	text, err := strconv.Unquote(literal.Value)
	if err != nil || text == "" {
		return
	}
	first, _ := utf8.DecodeRuneInString(text)
	if unicode.IsUpper(first) && !technicalErrorPrefix(text) {
		report.add(source, goStyleErrorText, literal.Pos(), scope, "error text starts with a capital letter")
	}
}

func technicalErrorPrefix(text string) bool {
	word := strings.Fields(text)[0]
	uppercase := 0
	for _, value := range word {
		if unicode.IsUpper(value) {
			uppercase++
		} else if unicode.IsDigit(value) {
			return true
		}
	}
	return uppercase > 1
}

func appendInconsistentNames(report *goStyleReport, groups map[string]map[string]goStyleFinding, kind goStyleKind, label string) {
	for _, names := range groups {
		multiple := false
		var first string
		for name := range names {
			if first != "" && name != first {
				multiple = true
				break
			}
			first = name
		}
		if !multiple {
			continue
		}
		for name, finding := range names {
			finding.Kind = kind
			finding.Detail = fmt.Sprintf("%s %q differs within one owner", label, name)
			report.Findings = append(report.Findings, finding)
		}
	}
}

func rememberStyleName(groups map[string]map[string]goStyleFinding, owner, name string, finding goStyleFinding) {
	if groups[owner] == nil {
		groups[owner] = map[string]goStyleFinding{}
	}
	groups[owner][name] = finding
}

func (r *goStyleReport) add(source GoFile, kind goStyleKind, position token.Pos, symbol, detail string) {
	r.Findings = append(r.Findings, r.finding(source, kind, position, symbol, detail))
}

func (r goStyleReport) finding(source GoFile, kind goStyleKind, position token.Pos, symbol, detail string) goStyleFinding {
	return goStyleFinding{Kind: kind, File: source.Path, Line: source.Line(position), Symbol: symbol, Detail: detail}
}

func packageStyleKey(file, name string) string {
	directory := path.Dir(file)
	if directory == "." {
		return name
	}
	return directory + "#" + name
}

func receiverTypeName(expression ast.Expr) string {
	switch value := expression.(type) {
	case *ast.Ident:
		return value.Name
	case *ast.IndexExpr:
		return receiverTypeName(value.X)
	case *ast.IndexListExpr:
		return receiverTypeName(value.X)
	case *ast.StarExpr:
		return receiverTypeName(value.X)
	default:
		return ""
	}
}

func validDoc(group *ast.CommentGroup, name string) bool {
	if group == nil {
		return false
	}
	text := strings.TrimSpace(group.Text())
	return strings.HasPrefix(text, name+" ") && strings.HasSuffix(text, ".")
}

func nonCanonicalInitialism(name string) string {
	for _, value := range []string{"Id", "Url", "Http", "Json", "Api", "Cpu", "Gpu", "Cuda", "Gguf"} {
		remaining := name
		for {
			_, after, found := strings.Cut(remaining, value)
			if !found {
				break
			}
			next, _ := utf8.DecodeRuneInString(after)
			if after == "" || !unicode.IsLower(next) {
				return value
			}
			remaining = after
		}
	}
	return ""
}

func getterMethod(function *ast.FuncDecl) bool {
	name := function.Name.Name
	if function.Recv == nil || !ast.IsExported(name) || !strings.HasPrefix(name, "Get") || len(name) == len("Get") {
		return false
	}
	suffix, _ := utf8.DecodeRuneInString(name[len("Get"):])
	if !unicode.IsUpper(suffix) {
		return false
	}
	if function.Type.Params != nil && len(function.Type.Params.List) != 0 {
		return false
	}
	return function.Type.Results != nil && len(function.Type.Results.List) != 0
}
