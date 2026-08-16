package codeprofile

import (
	"go/ast"
	"go/token"
	"path"
	"strconv"
	"strings"

	"overgo/internal/repoanalysis"
)

type ConsumerDeclaration struct {
	File                 string `json:"file"`
	Package              string `json:"package"`
	Name                 string `json:"name"`
	Kind                 string `json:"kind"`
	Exported             bool   `json:"exported,omitempty"`
	ProductionReferences int    `json:"production_references,omitempty"`
	TestReferences       int    `json:"test_references,omitempty"`
	ExternalReferences   int    `json:"external_references,omitempty"`
	Boundary             string `json:"boundary,omitempty"`
}

type ConsumerSummary struct {
	Production int `json:"production"`
	TestOnly   int `json:"test_only"`
	Boundary   int `json:"boundary"`
	Zero       int `json:"zero"`
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
	return strings.Join([]string{declaration.Package, declaration.File, declaration.Kind, declaration.Name}, "\x00")
}

type consumerIndex struct {
	declarations []ConsumerDeclaration
	keys         map[string][]int
	methods      map[string][]int
	objects      map[*ast.Object]int
	definitions  map[*ast.Ident]bool
}

// ProductionConsumerCensus classifies declarations in changed production
// files using the repository's parsed syntax and go-list build selection.
// A nil changed set includes every production declaration.
func ProductionConsumerCensus(snapshot repoanalysis.SourceSnapshot, selection repoanalysis.BuildSelection, changed map[string]bool) ([]ConsumerDeclaration, ConsumerSummary, error) {
	index := consumerIndex{keys: map[string][]int{}, methods: map[string][]int{}, objects: map[*ast.Object]int{}, definitions: map[*ast.Ident]bool{}}
	packageNames := map[string]string{}
	for _, source := range snapshot.Files {
		file, err := source.Syntax()
		if err != nil {
			return nil, ConsumerSummary{}, err
		}
		packageNames[packagePath(source, file, selection)] = file.Name.Name
	}
	for _, source := range snapshot.Files {
		if source.Test || changed != nil && !changed[source.Path] {
			continue
		}
		file, _ := source.Syntax()
		generated, err := source.Generated()
		if err != nil {
			return nil, ConsumerSummary{}, err
		}
		index.addFile(source, file, packagePath(source, file, selection), selected(source.Path, selection), generated)
	}
	for _, source := range snapshot.Files {
		file, _ := source.Syntax()
		index.references(source, file, packagePath(source, file, selection), packageNames, selected(source.Path, selection))
	}
	return index.declarations, summarizeConsumers(index.declarations), nil
}

func (c *consumerIndex) addFile(source repoanalysis.GoFile, file *ast.File, packagePath string, active, generated bool) {
	boundary := ""
	if path.Base(packagePath) == "testutil" {
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
			if value.Recv != nil {
				kind, key = "method", ""
				if functionBoundary == "" {
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
			c.add(source.Path, packagePath, value.Name, kind, key, functionBoundary)
		case *ast.GenDecl:
			for _, spec := range value.Specs {
				switch named := spec.(type) {
				case *ast.TypeSpec:
					c.add(source.Path, packagePath, named.Name, "type", symbolKey(packagePath, named.Name.Name), boundary)
					c.addBoundaryFields(source.Path, packagePath, named)
				case *ast.ValueSpec:
					valueBoundary := boundary
					if valueBoundary == "" && value.Tok == token.VAR && (len(named.Values) > 0 || hasDirective(named.Doc, "//go:embed")) {
						valueBoundary = "initialization"
					}
					for _, name := range named.Names {
						c.add(source.Path, packagePath, name, strings.ToLower(value.Tok.String()), symbolKey(packagePath, name.Name), valueBoundary)
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

func (c *consumerIndex) add(file, packagePath string, name *ast.Ident, kind, key, boundary string) {
	index := len(c.declarations)
	c.declarations = append(c.declarations, ConsumerDeclaration{
		File: file, Package: packagePath, Name: name.Name, Kind: kind, Exported: ast.IsExported(name.Name), Boundary: boundary,
	})
	c.definitions[name] = true
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
			c.add(file, packagePath, name, kind, "", boundary)
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
		alias := path.Base(importPath)
		if imported.Name != nil {
			alias = imported.Name.Name
		} else if name := packageNames[importPath]; name != "" {
			alias = name
		}
		aliases[alias] = importPath
	}
	selectorNames := map[*ast.Ident]bool{}
	ast.Inspect(file, func(node ast.Node) bool {
		if selector, ok := node.(*ast.SelectorExpr); ok {
			selectorNames[selector.Sel] = true
		}
		return true
	})
	ast.Inspect(file, func(node ast.Node) bool {
		switch value := node.(type) {
		case *ast.CallExpr:
			c.reflectionBoundary(value)
		case *ast.SelectorExpr:
			if qualifier, ok := value.X.(*ast.Ident); ok {
				if imported := aliases[qualifier.Name]; imported != "" {
					c.count(c.resolve(c.keys[symbolKey(imported, value.Sel.Name)], active), source.Test, true)
					return true
				}
			}
			c.markBoundary(c.methods[value.Sel.Name], "method-dispatch")
		case *ast.Ident:
			if c.definitions[value] {
				return true
			}
			if selectorNames[value] {
				return true
			}
			if value.Obj != nil {
				if index, ok := c.objects[value.Obj]; ok {
					c.count([]int{index}, source.Test, false)
				}
				return true
			}
			if candidates := c.resolve(c.keys[symbolKey(packagePath, value.Name)], active); len(candidates) == 1 {
				c.count(candidates, source.Test, false)
			}
		}
		return true
	})
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

func (c *consumerIndex) count(indices []int, test, external bool) {
	for _, index := range indices {
		if external {
			c.declarations[index].ExternalReferences++
		}
		if test {
			c.declarations[index].TestReferences++
		} else {
			c.declarations[index].ProductionReferences++
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
