package repoanalysis

import (
	"go/ast"
	"go/token"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
)

// schemaNamed returns the schema a string literal names, as the gate counts one.
//   - a schema's name opens with overgo and closes with a version after a dot or a slash (the rule of landing.recordSchemas)
func schemaNamed(literal *ast.BasicLit) (string, bool) {
	if literal.Kind != token.STRING {
		return "", false
	}
	name := strings.Trim(literal.Value, "\"`")
	version, found := strings.CutPrefix(name[strings.LastIndexAny(name, "./")+1:], "v")
	_, err := strconv.Atoi(version)
	return name, found && err == nil && strings.HasPrefix(name, "overgo")
}

// binding says how far a value has bound the schema it carries.
type binding int

const (
	// boundName is the schema's own string.
	boundName binding = iota + 1
	// boundContract is an artifact.DocumentContract of the schema.
	boundContract
	// boundCodec is an artifact.DocumentCodec over such a contract.
	boundCodec
)

// carried is one schema a value carries.
//   - schema == "" stands for the argument a summary follows
type carried struct {
	schema  string
	binding binding
}

// schemaUse is what a declaration does with a schema a value carries.
type schemaUse int

const (
	// useBind puts the schema in a contract or a descriptor, which makes it a record's schema.
	useBind schemaUse = iota
	// useWrite makes a document of the schema, or sends its contract where the reading cannot follow.
	useWrite
	// schemaUses is the number of uses.
	schemaUses
)

// The artifact package's directory, its types that bind a schema and their fields that hold one.
const (
	artifactDirectory = "internal/artifact"
	contractType      = "DocumentContract"
	codecType         = "DocumentCodec"
	descriptorType    = "Descriptor"
	schemaField       = "Schema"
	contractField     = "Contract"
	contractForField  = "ContractFor"
)

// codecMakers are the methods of artifact.DocumentCodec that give a document its identity.
//   - Content, ContentBytes and Batch are none, because each carries a document a maker already made
var codecMakers = []string{"New", "NewInitial", "Normalize"}

// codecCopyChange is the method of artifact.DocumentCodec that returns the codec it is called on with another copy.
const codecCopyChange = "WithClone"

// contractWriters are the methods of artifact.DocumentContract that put its schema on bytes.
var contractWriters = []string{"Content", "ContentBytes", "ContentJSON", "Descriptor", "OwnedContentBytes"}

// include returns a set with the values it lacked, and whether it grew.
func include(set []carried, values ...carried) ([]carried, bool) {
	grew := false
	for _, value := range values {
		if !slices.Contains(set, value) {
			set, grew = append(set, value), true
		}
	}
	return set, grew
}

// rebind returns the values at one binding, without the ones bound less far than least.
func rebind(values []carried, least, bound binding) []carried {
	var out []carried
	for _, value := range values {
		if value.binding >= least {
			out, _ = include(out, carried{value.schema, bound})
		}
	}
	return out
}

// summaryKey names one parameter of one function, given an argument bound so far.
type summaryKey struct {
	function  ast.Node
	parameter int
	binding   binding
}

// summary is what a function does with one argument.
//   - uses holds each schemaUse the function puts the argument to
//   - returned holds each binding the function returns the argument at
type summary struct {
	uses     [schemaUses]bool
	returned []binding
}

// schemaFlow is the state of one reading of where schemas go.
//   - functions and methods map a name to its declarations, keyed as the walk from the commands keys them
//   - held maps a package-level name, or a function's declaration, to what it carries or returns
type schemaFlow struct {
	functions map[string][]reachNode
	methods   map[string][]reachNode
	held      map[any][]carried
	summaries map[summaryKey]*summary
	grew      bool
}

// hold adds values to what a package-level name or a function's declaration carries.
//   - the argument a summary follows is no schema and is not held
func (f *schemaFlow) hold(holder any, values []carried) {
	for _, value := range values {
		if value.schema != "" {
			grown, grew := include(f.held[holder], value)
			f.held[holder], f.grew = grown, f.grew || grew
		}
	}
}

// named returns what a package-level name carries.
//   - a variable or a constant carries what its initializer and its assignments gave it
//   - a function carries what it returns, called or named as a value
func (f *schemaFlow) named(directory, name string) []carried {
	key := directory + " " + name
	values := slices.Clone(f.held[key])
	for _, function := range f.functions[key] {
		values, _ = include(values, f.held[function.node]...)
	}
	return values
}

// parameterObject returns the object of the parameter an argument at a place goes to.
//   - an argument past the last parameter goes to a variadic last parameter
func parameterObject(function *ast.FuncDecl, index int) *ast.Object {
	var names []*ast.Ident
	variadic := false
	for _, field := range function.Type.Params.List {
		names = append(names, field.Names...)
		_, variadic = field.Type.(*ast.Ellipsis)
	}
	if index >= len(names) && variadic {
		index = len(names) - 1
	}
	if index < 0 || index >= len(names) {
		return nil
	}
	return names[index].Obj
}

// summary returns what a function does with the argument at one place, bound so far.
//   - a function met again while its summary is pending reads as doing nothing more, because the reading ends
func (f *schemaFlow) summary(target reachNode, index int, bound binding) *summary {
	key := summaryKey{target.node, index, bound}
	if known, found := f.summaries[key]; found {
		return known
	}
	result := &summary{}
	f.summaries[key] = result
	function := target.node.(*ast.FuncDecl)
	parameter := parameterObject(function, index)
	if parameter == nil || function.Body == nil {
		return result
	}
	scan := schemaScan{flow: f, file: target.file, locals: map[*ast.Object][]carried{parameter: {{"", bound}}}}
	scan.use = func(use schemaUse, value carried) {
		if value.schema == "" {
			result.uses[use] = true
		}
	}
	for _, value := range scan.run(function.Body) {
		if value.schema == "" && !slices.Contains(result.returned, value.binding) {
			result.returned = append(result.returned, value.binding)
		}
	}
	return result
}

// schemaScan reads one declaration for the schemas its values carry.
//   - locals maps a local variable or a parameter to what it carries
//   - use receives each use the declaration puts a carried schema to
type schemaScan struct {
	flow   *schemaFlow
	file   *reachFile
	locals map[*ast.Object][]carried
	use    func(schemaUse, carried)
	grew   bool
}

// run reads a declaration until its locals stop growing, and returns what it returns.
func (s *schemaScan) run(node ast.Node) []carried {
	for {
		s.grew = false
		returned := s.walk(node)
		if !s.grew {
			return returned
		}
	}
}

// walk reads one function body or initializer and returns what its return statements carry.
//   - a function literal's returns are its own
func (s *schemaScan) walk(node ast.Node) []carried {
	var returned []carried
	ast.Inspect(node, func(node ast.Node) bool {
		switch typed := node.(type) {
		case *ast.FuncLit:
			s.walk(typed.Body)
			return false
		case *ast.ReturnStmt:
			for _, result := range typed.Results {
				returned, _ = include(returned, s.values(result)...)
			}
		case *ast.AssignStmt:
			s.assign(typed.Lhs, typed.Rhs)
		case *ast.ValueSpec:
			targets := make([]ast.Expr, len(typed.Names))
			for index, name := range typed.Names {
				targets[index] = name
			}
			s.assign(targets, typed.Values)
		case *ast.CallExpr:
			s.call(typed)
		case *ast.CompositeLit:
			s.composite(typed, typed.Type)
		}
		return true
	})
	return returned
}

// local reports whether an identifier names a variable or a parameter of the declaration read.
func (s *schemaScan) local(name *ast.Ident) bool {
	return name.Obj != nil && s.file.syntax.Scope.Lookup(name.Name) != name.Obj
}

// imported returns the module directory of the package an expression names, and whether it names an import.
func (s *schemaScan) imported(expression ast.Expr) (string, bool) {
	qualifier, isIdent := expression.(*ast.Ident)
	if !isIdent || qualifier.Obj != nil {
		return "", false
	}
	directory, found := s.file.imports[qualifier.Name]
	return directory, found
}

// escape counts each contract and codec among values as written, because it goes where the reading cannot follow.
//   - a schema's string that goes there is bound to nothing, so it writes nothing
func (s *schemaScan) escape(values []carried) {
	for _, value := range values {
		if value.binding >= boundContract {
			s.use(useWrite, value)
		}
	}
}

// keep adds values to what a local variable carries.
func (s *schemaScan) keep(variable *ast.Object, values []carried) {
	grown, grew := include(s.locals[variable], values...)
	s.locals[variable], s.grew = grown, s.grew || grew
}

// assign gives each target what its source carries.
//   - a local variable and a package-level variable hold it
//   - every target of one call holds what the call returns, because the reading keeps a function's results as one set
//   - the Schema field of a contract and the Contract and ContractFor fields of a codec bind as they do in a literal
//   - only a local variable that already holds a contract or a codec takes a field, because the reading knows no types
//   - any other field or element lets a contract or a codec escape
func (s *schemaScan) assign(targets, sources []ast.Expr) {
	if len(sources) == 1 && len(targets) > 1 {
		sources = slices.Repeat(sources, len(targets))
	}
	if len(targets) != len(sources) {
		return
	}
	for index, source := range sources {
		values, target := s.values(source), targets[index]
		if name, isIdent := target.(*ast.Ident); isIdent {
			switch {
			case name.Name == "_":
			case s.local(name):
				s.keep(name.Obj, values)
			default:
				s.flow.hold(s.file.directory+" "+name.Name, values)
				s.escape(slices.DeleteFunc(slices.Clone(values), func(value carried) bool { return value.schema != "" }))
			}
			continue
		}
		bound, binds := values, false
		for field := true; field; {
			selector, isSelector := target.(*ast.SelectorExpr)
			switch {
			case isSelector && selector.Sel.Name == schemaField:
				bound, binds, target = rebind(bound, boundName, boundContract), true, selector.X
			case isSelector && (selector.Sel.Name == contractField || selector.Sel.Name == contractForField):
				bound, target = rebind(bound, boundName, boundCodec), selector.X
			default:
				field = false
			}
		}
		holder, isIdent := target.(*ast.Ident)
		if !isIdent || !s.local(holder) || len(rebind(s.locals[holder.Obj], boundContract, boundContract)) == 0 {
			s.escape(values)
			continue
		}
		if binds {
			for _, value := range values {
				s.use(useBind, value)
			}
		}
		s.keep(holder.Obj, bound)
	}
}

// values returns the schemas an expression carries.
//   - a schema's literal carries it as a name
//   - a name carries what its variable, constant, parameter or function holds
//   - the Schema field of a contract carries its schema as a name, and the Contract field of a codec as a contract
func (s *schemaScan) values(expression ast.Expr) []carried {
	switch typed := expression.(type) {
	case *ast.BasicLit:
		if name, named := schemaNamed(typed); named {
			return []carried{{name, boundName}}
		}
	case *ast.ParenExpr:
		return s.values(typed.X)
	case *ast.StarExpr:
		return s.values(typed.X)
	case *ast.UnaryExpr:
		if typed.Op == token.AND {
			return s.values(typed.X)
		}
	case *ast.Ident:
		if s.local(typed) {
			return s.locals[typed.Obj]
		}
		return s.flow.named(s.file.directory, typed.Name)
	case *ast.SelectorExpr:
		if directory, imported := s.imported(typed.X); imported {
			return s.flow.named(directory, typed.Sel.Name)
		}
		switch typed.Sel.Name {
		case contractField:
			return rebind(s.values(typed.X), boundContract, boundContract)
		case schemaField:
			return rebind(s.values(typed.X), boundName, boundName)
		}
	case *ast.CompositeLit:
		return s.composite(typed, typed.Type)
	case *ast.CallExpr:
		return s.call(typed)
	case *ast.FuncLit:
		return s.walk(typed.Body)
	}
	return nil
}

// artifactType returns the name of the artifact package's type an expression writes, or nothing.
func (s *schemaScan) artifactType(expression ast.Expr) string {
	for {
		switch typed := expression.(type) {
		case *ast.IndexExpr:
			expression = typed.X
			continue
		case *ast.IndexListExpr:
			expression = typed.X
			continue
		case *ast.Ident:
			if s.file.directory == artifactDirectory && !s.local(typed) {
				return typed.Name
			}
		case *ast.SelectorExpr:
			if directory, imported := s.imported(typed.X); imported && directory == artifactDirectory {
				return typed.Sel.Name
			}
		}
		return ""
	}
}

// composite returns what a composite literal of a declared type carries.
//   - a contract carries the schema of its Schema field, and binds it
//   - a codec carries the schemas of its Contract and ContractFor fields
//   - a descriptor binds and writes the schema of its Schema field
//   - any other literal lets the contracts and codecs among its elements escape
//   - an element that writes no type takes the element type of the literal around it
func (s *schemaScan) composite(literal *ast.CompositeLit, declared ast.Expr) []carried {
	kind := s.artifactType(declared)
	var element ast.Expr
	switch typed := declared.(type) {
	case *ast.ArrayType:
		element = typed.Elt
	case *ast.MapType:
		element = typed.Value
	}
	var made []carried
	for _, item := range literal.Elts {
		field := ""
		if pair, isPair := item.(*ast.KeyValueExpr); isPair {
			if name, bare := pair.Key.(*ast.Ident); bare {
				field = name.Name
			}
			item = pair.Value
		}
		var values []carried
		if nested, isLiteral := item.(*ast.CompositeLit); isLiteral && nested.Type == nil {
			values = s.composite(nested, element)
		} else {
			values = s.values(item)
		}
		switch {
		case kind == contractType && field == schemaField:
			for _, value := range values {
				s.use(useBind, value)
			}
			made, _ = include(made, rebind(values, boundName, boundContract)...)
		case kind == descriptorType && field == schemaField:
			for _, value := range values {
				s.use(useBind, value)
				s.use(useWrite, value)
			}
		case kind == codecType && (field == contractField || field == contractForField):
			made, _ = include(made, rebind(values, boundName, boundCodec)...)
		case kind != contractType && kind != descriptorType && kind != codecType:
			s.escape(values)
		}
	}
	return made
}

// call reads one call and returns what its result carries.
//   - a maker called on a codec, and a writer called on a contract, write its schema
//   - an argument to a function of this module is put to the uses its summary holds, and comes back as the summary returns it
//   - an argument to any other callee lets a contract or a codec escape
//   - a method called returns what the methods of its name return, as the walk from the commands finds a method by its name
func (s *schemaScan) call(call *ast.CallExpr) []carried {
	callee := call.Fun
	for generic := true; generic; {
		switch typed := callee.(type) {
		case *ast.ParenExpr:
			callee = typed.X
		case *ast.IndexExpr:
			callee = typed.X
		case *ast.IndexListExpr:
			callee = typed.X
		default:
			generic = false
		}
	}
	var functions []reachNode
	var made []carried
	switch typed := callee.(type) {
	case *ast.FuncLit:
		made = s.walk(typed.Body)
	case *ast.Ident:
		if !s.local(typed) {
			functions, made = s.flow.functions[s.file.directory+" "+typed.Name], s.flow.named(s.file.directory, typed.Name)
		}
	case *ast.SelectorExpr:
		if directory, imported := s.imported(typed.X); imported {
			functions, made = s.flow.functions[directory+" "+typed.Sel.Name], s.flow.named(directory, typed.Sel.Name)
			break
		}
		for _, value := range s.values(typed.X) {
			if value.binding == boundCodec && slices.Contains(codecMakers, typed.Sel.Name) ||
				value.binding == boundContract && slices.Contains(contractWriters, typed.Sel.Name) {
				s.use(useWrite, value)
			}
			if value.binding == boundCodec && typed.Sel.Name == codecCopyChange {
				made, _ = include(made, value)
			}
		}
		for _, method := range s.flow.methods[selectionKey(s.file.directory, typed.Sel.Name)] {
			made, _ = include(made, s.flow.held[method.node]...)
		}
	}
	for index, argument := range call.Args {
		values := s.values(argument)
		if len(functions) == 0 {
			s.escape(values)
			continue
		}
		for _, value := range values {
			for _, function := range functions {
				does := s.flow.summary(function, index, value.binding)
				for use, put := range does.uses {
					if put {
						s.use(schemaUse(use), value)
					}
				}
				for _, bound := range does.returned {
					made, _ = include(made, carried{value.schema, bound})
				}
			}
		}
	}
	return made
}

// unwrittenRecordSchemas lists the record schemas of a snapshot's production files that no command writes, and counts the schemas named and the records among them.
//   - named = the schemas the production files outside test support name
//   - record schema = a named schema that reaches the Schema field of an artifact.DocumentContract or an artifact.Descriptor, because the store keeps a document under its descriptor's schema
//   - a schema that reaches neither is a wire document's, an identity's or a table's, and is no record's
//   - writer = a declaration a command reaches that calls a maker on a codec of the schema, calls a writer on a contract of it, writes it in a descriptor, or lets a contract or a codec of it escape
//   - reached = entered by reachFromCommands, which enters no declaration of a test-support package
//   - a schema passes from a constant to a contract, a codec, a function's result, an argument and a local variable by their names
//   - the reading follows names, never types, so a name two declarations share carries what both hold, and the list is a lower bound
//   - each line = the first file that names the schema + the schema
func unwrittenRecordSchemas(snapshot gosyntax.Snapshot, outside map[string]bool) (list []string, named, records int, err error) {
	reach, err := reachFromCommands(snapshot, outside)
	if err != nil {
		return nil, 0, 0, err
	}
	flow := &schemaFlow{functions: map[string][]reachNode{}, methods: map[string][]reachNode{}, held: map[any][]carried{}, summaries: map[summaryKey]*summary{}}
	namedIn, bound, written := map[string]string{}, map[string]bool{}, map[string]bool{}
	for _, file := range reach.files {
		for _, declaration := range file.syntax.Decls {
			if function, isFunction := declaration.(*ast.FuncDecl); isFunction {
				key, into := file.directory+" "+function.Name.Name, flow.functions
				if function.Recv != nil {
					key, into = selectionKey(file.directory, function.Name.Name), flow.methods
				}
				into[key] = append(into[key], reachNode{file, function})
			}
		}
		if reach.support(file.directory) {
			continue
		}
		ast.Inspect(file.syntax, func(node ast.Node) bool {
			if literal, isLiteral := node.(*ast.BasicLit); isLiteral {
				if name, isSchema := schemaNamed(literal); isSchema && (namedIn[name] == "" || file.source.Path < namedIn[name]) {
					namedIn[name] = file.source.Path
				}
			}
			return true
		})
	}
	read := func(file *reachFile, declaration, body ast.Node) {
		scan := schemaScan{flow: flow, file: file, locals: map[*ast.Object][]carried{}}
		scan.use = func(use schemaUse, value carried) {
			switch {
			case use == useBind:
				bound[value.schema] = true
			case reach.walked[declaration]:
				written[value.schema] = true
			}
		}
		flow.hold(declaration, scan.run(body))
	}
	for flow.grew = true; flow.grew; {
		flow.grew = false
		for _, file := range reach.files {
			for _, declaration := range file.syntax.Decls {
				switch typed := declaration.(type) {
				case *ast.FuncDecl:
					if typed.Body != nil {
						read(file, typed, typed.Body)
					}
				case *ast.GenDecl:
					for _, spec := range typed.Specs {
						if _, isValue := spec.(*ast.ValueSpec); isValue {
							read(file, spec, spec)
						}
					}
				}
			}
		}
	}
	for name, file := range namedIn {
		if !bound[name] {
			continue
		}
		records++
		if !written[name] {
			list = append(list, file+" "+name)
		}
	}
	slices.Sort(list)
	return list, len(namedIn), records, nil
}

// schemaList is the file that keeps the record schemas no command writes.
const schemaList = "unwritten_record_schemas.txt"

// artifactFixture stands for the artifact package in the fixture rows, with the shapes the reading follows.
const artifactFixture = `package artifact

type DocumentContract struct {
	Kind              int
	MediaType, Schema string
}

type Descriptor struct{ MediaType, Schema string }

type DocumentCodec[T any] struct {
	Contract    DocumentContract
	ContractFor func(T) DocumentContract
}

func JSONContract(kind int, schema string) DocumentContract {
	return DocumentContract{Kind: kind, MediaType: "application/json", Schema: schema}
}

func JSONDocumentCodec[T any](name string, kind int, schema string) DocumentCodec[T] {
	return DocumentCodec[T]{Contract: DocumentContract{Kind: kind, Schema: schema}}
}

func JSONContent(contract DocumentContract, value any) []byte { return contract.ContentBytes(nil) }
`

// TestEveryProcessRecordSchemaHasAProductionWriter holds every record schema production code names to a writer a command reaches.
//   - the checkout's unwritten record schemas = the entries of schemaList, both ways
//   - an unwritten schema the list lacks is refused, because the list only shrinks
//   - a listed schema that a command writes, or that production code no longer binds, is refused
//   - every entry carries one of sortClasses, and belongs to one group of sortRecord of that class
//   - each fixture row pins one rule of the reading on invented files
func TestEveryProcessRecordSchemaHasAProductionWriter(t *testing.T) {
	t.Parallel()
	const command = "package main\n\nimport \"overgo/internal/lib\"\n\nfunc main() { lib.Use() }\n"
	const lib = "package lib\n\nimport \"overgo/internal/artifact\"\n\n"
	for _, row := range []struct {
		rule  string
		files map[string]string
		want  []string
	}{
		{"a maker called on a codec writes its schema where a command reaches the call, and a maker only a test calls writes nothing", map[string]string{
			"internal/lib/lib.go": lib + "const madeSchema, idleSchema = \"overgo/made/v1\", \"overgo/idle/v1\"\n\nvar madeCodec = artifact.JSONDocumentCodec[int](\"made\", 1, madeSchema)\n\n" +
				"var idleCodec = artifact.JSONDocumentCodec[int](\"idle\", 1, idleSchema)\n\nfunc Use() { madeCodec.NewInitial(1) }\n\nfunc Idle() { idleCodec.New(1) }\n",
			"internal/lib/lib_test.go": "package lib\n\nfunc TestIdle(t *testing.T) { Idle() }\n",
		}, []string{"internal/lib/lib.go overgo/idle/v1"}},
		{"Content and Batch of a codec carry a document a maker made, so neither writes its schema, and no reader does", map[string]string{
			"internal/lib/lib.go": lib + "var heldCodec = artifact.JSONDocumentCodec[int](\"held\", 1, \"overgo/held/v1\")\n\n" +
				"func Use(schema string) bool {\n\theldCodec.Content(1)\n\theldCodec.Batch(\"key\", 1)\n\theldCodec.Require(1)\n\treturn heldCodec.Contract.Schema == schema\n}\n",
		}, []string{"internal/lib/lib.go overgo/held/v1"}},
		{"a writer called on a contract, or on the contract of a codec, writes its schema, and a reader writes nothing", map[string]string{
			"internal/lib/lib.go": lib + "var paged = artifact.DocumentContract{Kind: 1, Schema: \"overgo/paged/v1\"}\n\n" +
				"var kept = artifact.DocumentCodec[int]{Contract: artifact.JSONContract(1, \"overgo/kept/v1\")}\n\nvar read = artifact.JSONContract(1, \"overgo/read/v1\")\n\n" +
				"func Use() {\n\tpaged.ContentJSON(1, 2)\n\tkept.Contract.OwnedContentBytes(nil)\n\tread.ValidateContent(nil, 1)\n\tread.Identify(nil)\n}\n",
		}, []string{"internal/lib/lib.go overgo/read/v1"}},
		{"a schema passes by name from a constant of another package into a contract written in place", map[string]string{
			"cmd/tool/main.go": "package main\n\nimport (\n\t\"overgo/internal/artifact\"\n\t\"overgo/internal/lib\"\n)\n\n" +
				"func main() { artifact.JSONContent(artifact.JSONContract(1, lib.SentSchema), lib.Use()) }\n",
			"internal/lib/lib.go": lib + "const SentSchema, HeldSchema = \"overgo/sent/v1\", \"overgo/held/v1\"\n\nfunc Use() artifact.DocumentContract { return artifact.JSONContract(1, HeldSchema) }\n",
		}, []string{"internal/lib/lib.go overgo/held/v1"}},
		{"an argument is followed into the function it is passed to, one call at a time", map[string]string{
			"internal/lib/lib.go": lib + "var sent = artifact.JSONContract(1, \"overgo/sent/v1\")\n\nvar checked = artifact.JSONContract(1, \"overgo/checked/v1\")\n\n" +
				"func publish(contract artifact.DocumentContract, value any) { artifact.JSONContent(contract, value) }\n\n" +
				"func check(contract artifact.DocumentContract) { contract.ValidateContent(nil, 1) }\n\n" +
				"func Use() {\n\tpublish(sent, 1)\n\tcheck(checked)\n}\n\nfunc Spare() { publish(checked, 1) }\n",
		}, []string{"internal/lib/lib.go overgo/checked/v1"}},
		{"a function's result carries what it returns, to every target of the call", map[string]string{
			"internal/lib/lib.go": lib + "var oldCodec = artifact.JSONDocumentCodec[int](\"old\", 1, \"overgo/doc/v1\")\n\nvar newCodec = artifact.JSONDocumentCodec[int](\"new\", 1, \"overgo/doc/v2\")\n\n" +
				"var restCodec = artifact.JSONDocumentCodec[int](\"rest\", 1, \"overgo/rest/v1\")\n\n" +
				"func codecFor(version int) (artifact.DocumentCodec[int], error) {\n\tif version == 1 {\n\t\treturn oldCodec, nil\n\t}\n\treturn newCodec, nil\n}\n\n" +
				"func Use() {\n\tcodec, err := codecFor(2)\n\t_ = err\n\tcodec.New(1)\n\trestCodec.Require(1)\n}\n",
		}, []string{"internal/lib/lib.go overgo/rest/v1"}},
		{"a method called carries what the methods of its name return", map[string]string{
			"internal/lib/lib.go": lib + "var pageContract = artifact.JSONContract(1, \"overgo/page/v1\")\n\nvar leafContract = artifact.JSONContract(1, \"overgo/leaf/v1\")\n\n" +
				"type page struct{}\n\nfunc (page) contract() artifact.DocumentContract { return pageContract }\n\nfunc (page) other() artifact.DocumentContract { return leafContract }\n\n" +
				"func Use(p page) { p.contract().ContentBytes(nil) }\n",
		}, []string{"internal/lib/lib.go overgo/leaf/v1"}},
		{"a schema that reaches no contract and no descriptor is no record's", map[string]string{
			"internal/lib/lib.go": lib + "type reply struct{ Schema string }\n\nvar lost = artifact.JSONContract(1, \"overgo/lost/v1\")\n\n" +
				"func Use(schema string) (reply, []byte, bool) {\n\treturn reply{Schema: \"overgo/reply/v1\"}, []byte(\"overgo/seed/v1\"), schema == \"overgo/seen/v1\" && lost.Schema != schema\n}\n",
		}, []string{"internal/lib/lib.go overgo/lost/v1"}},
		{"a descriptor written with a schema binds it and writes it", map[string]string{
			"internal/lib/lib.go": lib + "func Use() artifact.Descriptor {\n\treturn artifact.Descriptor{MediaType: \"text/plain\", Schema: \"overgo/stamped/v1\"}\n}\n\n" +
				"func Spare() artifact.Descriptor { return artifact.Descriptor{Schema: \"overgo/unstamped/v1\"} }\n",
		}, []string{"internal/lib/lib.go overgo/unstamped/v1"}},
		{"a local variable carries what it is assigned, and a field assigned on a codec binds as it does in a literal", map[string]string{
			"internal/lib/lib.go": lib + "var nowCodec = artifact.JSONDocumentCodec[int](\"now\", 1, \"overgo/now/v2\")\n\n" +
				"var earlierCodec = func() artifact.DocumentCodec[int] {\n\tearlier := nowCodec\n\tearlier.Contract.Schema = \"overgo/now/v1\"\n\treturn earlier\n}()\n\n" +
				"func Use() {\n\tcodec := nowCodec\n\tcodec.New(1)\n\tearlier := earlierCodec\n\tearlier.Require(1)\n}\n",
		}, []string{"internal/lib/lib.go overgo/now/v1"}},
		{"a contract that goes where the reading cannot follow counts as written, and its schema's string that goes there does not", map[string]string{
			"internal/lib/lib.go": lib + "type job struct{ contract artifact.DocumentContract }\n\nvar queued = artifact.JSONContract(1, \"overgo/queued/v1\")\n\n" +
				"var handed = artifact.JSONContract(1, \"overgo/handed/v1\")\n\nvar parked = artifact.JSONContract(1, \"overgo/parked/v1\")\n\nvar named = artifact.JSONContract(1, \"overgo/named/v1\")\n\n" +
				"func Use(run func(artifact.DocumentContract), report func(string)) job {\n\trun(handed)\n\treport(named.Schema)\n\treturn job{contract: queued}\n}\n\n" +
				"func Spare() job { return job{contract: parked} }\n",
		}, []string{"internal/lib/lib.go overgo/named/v1", "internal/lib/lib.go overgo/parked/v1"}},
		{"a writer in a test-support package is no production writer, and a schema only test support names is not production's", map[string]string{
			"internal/lib/lib.go": lib + "var Probe = artifact.JSONContract(1, \"overgo/probe/v1\")\n\nfunc Use() {}\n",
			"internal/libtest/libtest.go": "package libtest\n\nimport (\n\t\"overgo/internal/artifact\"\n\t\"overgo/internal/lib\"\n)\n\n" +
				"var fake = artifact.JSONContract(1, \"overgo/fake/v1\")\n\nfunc Fixture() {\n\tlib.Probe.ContentBytes(nil)\n\tfake.ContentBytes(nil)\n}\n",
			"internal/other/other_test.go": "package other\n\nimport \"overgo/internal/libtest\"\n\nfunc TestOther(t *testing.T) { libtest.Fixture() }\n",
		}, []string{"internal/lib/lib.go overgo/probe/v1"}},
		{"a maker in a function a package variable holds counts as reached, called or not, because the walk from the commands enters every variable", map[string]string{
			"internal/lib/lib.go": lib + "var builtCodec = artifact.JSONDocumentCodec[int](\"built\", 1, \"overgo/built/v1\")\n\nvar idleCodec = artifact.JSONDocumentCodec[int](\"idle\", 1, \"overgo/idle/v1\")\n\n" +
				"var build = func() { builtCodec.New(1) }\n\nfunc Use() {}\n\nfunc Idle() { idleCodec.New(1) }\n",
		}, []string{"internal/lib/lib.go overgo/idle/v1"}},
	} {
		contents := map[string][]byte{"cmd/tool/main.go": []byte(command), "internal/artifact/artifact.go": []byte(artifactFixture)}
		for name, content := range row.files {
			contents[name] = []byte(content)
		}
		snapshot, err := gosyntax.Snapshot{}.Overlay(contents)
		if err != nil {
			t.Fatal(err)
		}
		if found, _, _, err := unwrittenRecordSchemas(snapshot, nil); err != nil || !slices.Equal(found, row.want) {
			t.Errorf("%s: unwritten = %q, %v, want %q", row.rule, found, err, row.want)
		}
	}

	snapshot, outside := checkoutSource(t)
	found, named, records, err := unwrittenRecordSchemas(snapshot, outside)
	if err != nil {
		t.Fatal(err)
	}
	if records == 0 || records == len(found) {
		t.Fatalf("the reading found %d record schemas and a writer for %d, want some of each", records, records-len(found))
	}
	kept, classOf, err := sortedEntries(keptList(t, schemaList))
	if err != nil {
		t.Fatalf("%s: %v", schemaList, err)
	}
	missing, stale := listDrift(found, kept)
	for _, key := range missing {
		t.Errorf("no command reaches a writer of %s, so reach its writer from a command or delete its type, validator and codec, because %s only shrinks", key, schemaList)
	}
	for _, key := range stale {
		t.Errorf("%s lists %s, which a command writes or production code no longer binds, so delete its line and take it out of its group in %s", schemaList, key, sortRecord)
	}
	_, differs := heldEntries(recordGroups(t), func(group sortGroup) []string { return group.schemas }, kept, classOf)
	for _, differ := range differs {
		t.Errorf("%s: %s", sortRecord, differ)
	}
	t.Logf("schemas=%d records=%d unwritten=%d", named, records, len(found))
}
