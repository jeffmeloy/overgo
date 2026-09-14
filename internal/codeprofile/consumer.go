package codeprofile

import (
	"errors"
	"go/ast"
	"go/token"
	"path"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/repoanalysis"
)

type ConsumerDeclaration struct {
	File                 string `json:"file"`
	Package              string `json:"package"`
	Name                 string `json:"name"`
	Receiver             string `json:"receiver,omitzero"`
	Kind                 string `json:"kind"`
	Exported             bool   `json:"exported,omitzero"`
	ProductionReferences int    `json:"production_references,omitzero"`
	TestReferences       int    `json:"test_references,omitzero"`
	ExternalReferences   int    `json:"external_references,omitzero"`
	Boundary             string `json:"boundary,omitzero"`
}

type ConsumerSummary struct {
	Production int `json:"production"`
	TestOnly   int `json:"test_only"`
	Boundary   int `json:"boundary"`
	Zero       int `json:"zero"`
}

// ConsumerReference is one resolved syntactic edge from a declaration that
// observes another declaration. The same conservative index used for impact
// closure owns these edges.
type ConsumerReference struct {
	From   ConsumerDeclaration `json:"from"`
	To     ConsumerDeclaration `json:"to"`
	Kind   string              `json:"kind"`
	Line   int                 `json:"line"`
	Offset int                 `json:"offset"`
}

// referenceSite keeps syntax provenance in the existing reverse edge index.
type referenceSite struct {
	kind         string
	line, offset int
}

// NewUnconsumedSurface returns candidate declarations that did not exist in
// the baseline and have no production caller or mechanically protected use.
func NewUnconsumedSurface(base, candidate []ConsumerDeclaration) []ConsumerDeclaration {
	known := make(map[string]bool, len(base))
	for _, declaration := range base {
		known[declarationIdentity(declaration)] = true
	}
	var added []ConsumerDeclaration
	for _, declaration := range candidate {
		if !known[declarationIdentity(declaration)] && declaration.ProductionReferences == 0 && declaration.Boundary == "" {
			added = append(added, declaration)
		}
	}
	return added
}

func declarationIdentity(declaration ConsumerDeclaration) string {
	return strings.Join([]string{declaration.Package, declaration.File, declaration.Kind, declaration.Receiver, declaration.Name}, "\x00")
}

type consumerIndex struct {
	declarations []ConsumerDeclaration
	keys         map[string][]int
	methods      map[string][]int
	objects      map[*ast.Object]int
	definitions  map[*ast.Ident]int
	reverse      map[int]map[int]map[referenceSite]bool
	// interfaceMethods holds every method name declared by any interface in
	// the snapshot; only concrete methods matching one (or exported methods,
	// which may satisfy interfaces outside the snapshot such as io.Reader)
	// retain the method-dispatch boundary. Unexported methods matching no
	// snapshot interface cannot be dynamically dispatched and are reportable.
	interfaceMethods map[string]bool
	// testOnlyImports marks packages imported exclusively from test files:
	// test-support identified through imports, never through package names.
	testOnlyImports map[string]bool
}

type referenceCaller struct {
	index int
	valid bool
}

// ProductionConsumerCensus classifies declarations in changed production
// files using the repository's parsed syntax and go-list build selection.
// A nil changed set includes every production declaration.
func ProductionConsumerCensus(snapshot repoanalysis.SourceSnapshot, selection repoanalysis.BuildSelection, changed map[string]bool) ([]ConsumerDeclaration, ConsumerSummary, error) {
	index, err := productionConsumerIndex(snapshot, selection, changed)
	if err != nil {
		return nil, ConsumerSummary{}, err
	}
	return index.declarations, summarizeConsumers(index.declarations), nil
}

// ProductionConsumerGraph returns the complete declarations and resolved
// edges from the existing production consumer index.
func ProductionConsumerGraph(snapshot repoanalysis.SourceSnapshot, selection repoanalysis.BuildSelection) ([]ConsumerDeclaration, []ConsumerReference, ConsumerSummary, error) {
	index, err := productionConsumerIndex(snapshot, selection, nil)
	if err != nil {
		return nil, nil, ConsumerSummary{}, err
	}
	var references []ConsumerReference
	for target, callers := range index.reverse {
		if target < 0 || target >= len(index.declarations) {
			return nil, nil, ConsumerSummary{}, errors.New("code profile: consumer graph target is invalid")
		}
		for caller, sites := range callers {
			if caller < 0 || caller >= len(index.declarations) {
				return nil, nil, ConsumerSummary{}, errors.New("code profile: consumer graph caller is invalid")
			}
			for site := range sites {
				references = append(references, ConsumerReference{From: index.declarations[caller], To: index.declarations[target], Kind: site.kind, Line: site.line, Offset: site.offset})
			}
		}
	}
	sort.Slice(references, func(i, j int) bool {
		left, right := references[i], references[j]
		leftKey, rightKey := declarationIdentity(left.From)+"\x00"+declarationIdentity(left.To), declarationIdentity(right.From)+"\x00"+declarationIdentity(right.To)
		if leftKey != rightKey {
			return leftKey < rightKey
		}
		if left.Offset != right.Offset {
			return left.Offset < right.Offset
		}
		return left.Kind < right.Kind
	})
	return index.declarations, references, summarizeConsumers(index.declarations), nil
}

func productionConsumerIndex(snapshot repoanalysis.SourceSnapshot, selection repoanalysis.BuildSelection, changed map[string]bool) (consumerIndex, error) {
	index := consumerIndex{
		keys: map[string][]int{}, methods: map[string][]int{}, objects: map[*ast.Object]int{},
		definitions: map[*ast.Ident]int{}, reverse: map[int]map[int]map[referenceSite]bool{},
		interfaceMethods: map[string]bool{}, testOnlyImports: map[string]bool{},
	}
	packageNames, err := repoanalysis.PackageNames(snapshot, selection)
	if err != nil {
		return consumerIndex{}, err
	}
	productionImporters := map[string]map[string]bool{}
	testImports := map[string]bool{}
	for _, source := range snapshot.Files {
		file, _ := source.Syntax()
		for node := range ast.Preorder(file) {
			if value, ok := node.(*ast.InterfaceType); ok && value.Methods != nil {
				for _, field := range value.Methods.List {
					for _, name := range field.Names {
						index.interfaceMethods[name.Name] = true
					}
				}
			}
		}
		importer := packagePath(source, file, selection)
		for _, imported := range file.Imports {
			importPath, err := strconv.Unquote(imported.Path.Value)
			if err != nil {
				continue
			}
			if source.Test {
				testImports[importPath] = true
			} else {
				if productionImporters[importPath] == nil {
					productionImporters[importPath] = map[string]bool{}
				}
				productionImporters[importPath][importer] = true
			}
		}
	}
	// Test-helper status is TRANSITIVE: a package is a test helper when every
	// production importer is itself a test helper (or it has none and tests
	// import it). One fixture package importing another must not promote the
	// imported helper to production surface.
	for importPath := range testImports {
		if len(productionImporters[importPath]) == 0 {
			index.testOnlyImports[importPath] = true
		}
	}
	for changedHelpers := true; changedHelpers; {
		changedHelpers = false
		for importPath := range testImports {
			if index.testOnlyImports[importPath] {
				continue
			}
			helperOnly := len(productionImporters[importPath]) > 0
			for importer := range productionImporters[importPath] {
				if !index.testOnlyImports[importer] {
					helperOnly = false
					break
				}
			}
			if helperOnly {
				index.testOnlyImports[importPath] = true
				changedHelpers = true
			}
		}
	}
	for _, source := range snapshot.Files {
		if source.Test || changed != nil && !changed[source.Path] {
			continue
		}
		file, _ := source.Syntax()
		generated, err := source.Generated()
		if err != nil {
			return consumerIndex{}, err
		}
		index.addFile(source, file, packagePath(source, file, selection), selected(source.Path, selection), generated)
	}
	for _, source := range snapshot.Files {
		file, _ := source.Syntax()
		index.references(source, file, packagePath(source, file, selection), packageNames, selected(source.Path, selection))
	}
	return index, nil
}

func (c *consumerIndex) addFile(source repoanalysis.GoFile, file *ast.File, packagePath string, active, generated bool) {
	boundary := ""
	if c.testOnlyImports[packagePath] {
		boundary = "test-helper"
	} else if generated {
		boundary = "generated"
	} else if !active {
		boundary = "build-variant"
	}
	for _, declaration := range file.Decls {
		switch value := declaration.(type) {
		case *ast.FuncDecl:
			kind, key, functionBoundary := "function", symbolKey(packagePath, value.Name.Name), boundary
			receiver := ""
			if value.Recv != nil {
				kind, key, receiver = "method", "", receiverName(value)
				// Only plausibly-dispatched methods stay boundary: exported
				// (may satisfy interfaces outside the snapshot) or matching a
				// declared interface method. Unexported non-interface methods
				// are direct-call only and report when unused.
				if functionBoundary == "" && (ast.IsExported(value.Name.Name) || c.interfaceMethods[value.Name.Name]) {
					functionBoundary = "method-dispatch"
				}
			}
			if value.Body == nil {
				functionBoundary = "external"
			} else if value.Name.Name == "init" || file.Name.Name == "main" && value.Name.Name == "main" {
				functionBoundary = "command"
			} else if exportedByCgo(value) {
				functionBoundary = "cgo"
			}
			c.add(source.Path, packagePath, value.Name, receiver, kind, key, functionBoundary)
		case *ast.GenDecl:
			for _, spec := range value.Specs {
				switch named := spec.(type) {
				case *ast.TypeSpec:
					c.add(source.Path, packagePath, named.Name, "", "type", symbolKey(packagePath, named.Name.Name), boundary)
					c.addBoundaryFields(source.Path, packagePath, named)
				case *ast.ValueSpec:
					valueBoundary := boundary
					if valueBoundary == "" && value.Tok == token.VAR && (len(named.Values) > 0 || hasDirective(named.Doc, "//go:embed")) {
						valueBoundary = "initialization"
					}
					for _, name := range named.Names {
						c.add(source.Path, packagePath, name, "", strings.ToLower(value.Tok.String()), symbolKey(packagePath, name.Name), valueBoundary)
					}
				}
			}
		}
	}
}

func exportedByCgo(function *ast.FuncDecl) bool {
	return hasDirective(function.Doc, "//export "+function.Name.Name)
}

func hasDirective(group *ast.CommentGroup, want string) bool {
	if group != nil {
		for _, comment := range group.List {
			if strings.TrimSpace(comment.Text) == want || strings.HasPrefix(strings.TrimSpace(comment.Text), want+" ") {
				return true
			}
		}
	}
	return false
}

func (c *consumerIndex) add(file, packagePath string, name *ast.Ident, receiver, kind, key, boundary string) {
	index := len(c.declarations)
	c.declarations = append(c.declarations, ConsumerDeclaration{
		File: file, Package: packagePath, Name: name.Name, Receiver: receiver,
		Kind: kind, Exported: ast.IsExported(name.Name), Boundary: boundary,
	})
	c.definitions[name] = index
	if name.Obj != nil {
		c.objects[name.Obj] = index
	}
	if kind == "method" {
		c.methods[name.Name] = append(c.methods[name.Name], index)
	} else {
		c.keys[key] = append(c.keys[key], index)
	}
}

func (c *consumerIndex) addBoundaryFields(file, packagePath string, spec *ast.TypeSpec) {
	var fields *ast.FieldList
	kind, boundary := "", ""
	switch value := spec.Type.(type) {
	case *ast.InterfaceType:
		fields, kind, boundary = value.Methods, "interface-method", "interface"
	case *ast.StructType:
		fields, kind, boundary = value.Fields, "field", "serialization"
	}
	if fields == nil {
		return
	}
	for _, field := range fields.List {
		if boundary == "serialization" && field.Tag == nil {
			continue
		}
		for _, name := range field.Names {
			c.add(file, packagePath, name, "", kind, "", boundary)
		}
	}
}

func (c *consumerIndex) references(source repoanalysis.GoFile, file *ast.File, packagePath string, packageNames map[string]string, active bool) {
	aliases := map[string]string{}
	for _, imported := range file.Imports {
		importPath, err := strconv.Unquote(imported.Path.Value)
		if err != nil {
			continue
		}
		alias := packageNames[importPath]
		if imported.Name != nil {
			alias = imported.Name.Name
		}
		if alias == "" {
			continue
		}
		aliases[alias] = importPath
	}
	for _, declaration := range file.Decls {
		caller := referenceCaller{}
		if function, ok := declaration.(*ast.FuncDecl); ok {
			if index, found := c.definitions[function.Name]; found {
				caller = referenceCaller{index: index, valid: true}
			}
		}
		c.referenceNode(source, declaration, packagePath, aliases, active, caller)
	}
}

func (c *consumerIndex) referenceNode(source repoanalysis.GoFile, root ast.Node, packagePath string, aliases map[string]string, active bool, caller referenceCaller) {
	selectorNames := map[*ast.Ident]bool{}
	invocations := map[ast.Node]bool{}
	ast.PreorderStack(root, nil, func(node ast.Node, ancestors []ast.Node) bool {
		if selector, ok := node.(*ast.SelectorExpr); ok {
			selectorNames[selector.Sel] = true
		}
		if call, ok := node.(*ast.CallExpr); ok {
			// Deferred bodies and asynchronous execution retain a use edge
			// until execution context is resolved; syntax is not a trace.
			for _, parent := range ancestors {
				switch parent.(type) {
				case *ast.FuncLit, *ast.GoStmt, *ast.DeferStmt:
					return true
				}
			}
			invocations[referenceCallee(call.Fun)] = true
		}
		return true
	})
	for node := range ast.Preorder(root) {
		site := referenceSite{kind: "use"}
		site.line, site.offset = source.Line(node.Pos()), int(node.Pos())-1
		if invocations[node] {
			site.kind = "call"
		}
		switch value := node.(type) {
		case *ast.CallExpr:
			c.reflectionBoundary(value)
		case *ast.SelectorExpr:
			// Locally bound names shadow import aliases.
			if qualifier, ok := value.X.(*ast.Ident); ok && qualifier.Obj == nil {
				if imported := aliases[qualifier.Name]; imported != "" {
					c.count(c.resolve(c.keys[symbolKey(imported, value.Sel.Name)], active), source.Test, true, caller, site)
					continue
				}
			}
			// A selector use is a reference, credited to every same-named
			// method (receiver types are unresolved syntactically, so the
			// over-credit errs toward "used", never toward a false dead
			// report). The old blanket method-dispatch marking hid every
			// unused concrete method from the census.
			if site.kind == "call" {
				site.kind = "interface"
			}
			c.count(c.resolve(c.methods[value.Sel.Name], active), source.Test, false, caller, site)
		case *ast.Ident:
			if _, defined := c.definitions[value]; defined {
				continue
			}
			if selectorNames[value] {
				continue
			}
			if value.Obj != nil {
				if index, ok := c.objects[value.Obj]; ok {
					c.count([]int{index}, source.Test, false, caller, site)
				}
				continue
			}
			if candidates := c.resolve(c.keys[symbolKey(packagePath, value.Name)], active); len(candidates) == 1 {
				c.count(candidates, source.Test, false, caller, site)
			}
		}
	}
}

func referenceCallee(expression ast.Expr) ast.Expr {
	for {
		switch value := expression.(type) {
		case *ast.ParenExpr:
			expression = value.X
		case *ast.IndexExpr:
			expression = value.X
		case *ast.IndexListExpr:
			expression = value.X
		default:
			return expression
		}
	}
}

func (c *consumerIndex) resolve(indices []int, active bool) []int {
	if !active {
		return indices
	}
	selected := make([]int, 0, len(indices))
	for _, index := range indices {
		if c.declarations[index].Boundary != "build-variant" {
			selected = append(selected, index)
		}
	}
	if len(selected) > 0 {
		return selected
	}
	return indices
}

func (c *consumerIndex) reflectionBoundary(call *ast.CallExpr) {
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || (selector.Sel.Name != "MethodByName" && selector.Sel.Name != "FieldByName") || len(call.Args) == 0 {
		return
	}
	literal, ok := call.Args[0].(*ast.BasicLit)
	if !ok {
		return
	}
	name, err := strconv.Unquote(literal.Value)
	if err == nil {
		c.markBoundary(c.methods[name], "reflection")
	}
}

func (c *consumerIndex) count(indices []int, test, external bool, caller referenceCaller, site referenceSite) {
	for _, index := range indices {
		if external {
			c.declarations[index].ExternalReferences++
		}
		if test {
			c.declarations[index].TestReferences++
		} else {
			c.declarations[index].ProductionReferences++
		}
		if caller.valid {
			edgeSite := site
			if edgeSite.kind == "call" && c.declarations[index].Kind != "function" && c.declarations[index].Kind != "method" {
				edgeSite.kind = "use"
			}
			if c.reverse[index] == nil {
				c.reverse[index] = map[int]map[referenceSite]bool{}
			}
			if c.reverse[index][caller.index] == nil {
				c.reverse[index][caller.index] = map[referenceSite]bool{}
			}
			c.reverse[index][caller.index][edgeSite] = true
		}
	}
}

func (c *consumerIndex) markBoundary(indices []int, boundary string) {
	for _, index := range indices {
		if c.declarations[index].Boundary == "" || c.declarations[index].Boundary == "method-dispatch" {
			c.declarations[index].Boundary = boundary
		}
	}
}

func summarizeConsumers(declarations []ConsumerDeclaration) (summary ConsumerSummary) {
	for _, declaration := range declarations {
		switch {
		case declaration.ProductionReferences > 0:
			summary.Production++
		case declaration.TestReferences > 0:
			summary.TestOnly++
		case declaration.Boundary != "":
			summary.Boundary++
		default:
			summary.Zero++
		}
	}
	return summary
}

func packagePath(source repoanalysis.GoFile, file *ast.File, selection repoanalysis.BuildSelection) string {
	if value := selection.Packages[source.Path]; value != "" {
		return value
	}
	return path.Dir(source.Path) + "#" + file.Name.Name
}

func selected(file string, selection repoanalysis.BuildSelection) bool {
	value, known := selection.Files[file]
	return !known || value
}

func symbolKey(packagePath, name string) string { return packagePath + "\x00" + name }
