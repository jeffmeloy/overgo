package repoanalysis

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/inferencesurface"
	"overgo/internal/processcontrol"
)

// reachFile is one production Go file the walk reads.
//   - imports maps each name the file imports a package as to that package's directory in this module
//   - a package outside the module maps to ""
type reachFile struct {
	source    gosyntax.File
	syntax    *ast.File
	directory string
	imports   map[string]string
}

// reachNode is one declaration the walk enters, which is a function, a method, a type, a variable or a constant.
type reachNode struct {
	file *reachFile
	node ast.Node
}

// unreached is one function or method no command reaches.
//   - key = file + receiver + name, because build tags give one name a declaration per file
//   - lines = its source lines, its comment among them
type unreached struct {
	key   string
	lines int
}

// ignoreTag is the build tag of a file no build selects.
const ignoreTag = "ignore"

// neverBuilt reports whether a file's build constraint needs ignoreTag.
//   - true <=> the file builds with every tag set and does not build once ignoreTag alone is cleared
func neverBuilt(syntax *ast.File) bool {
	for _, group := range syntax.Comments {
		for _, comment := range group.List {
			if comment.Pos() > syntax.Package || !constraint.IsGoBuild(comment.Text) {
				continue
			}
			expression, err := constraint.Parse(comment.Text)
			return err == nil && expression.Eval(func(string) bool { return true }) &&
				!expression.Eval(func(tag string) bool { return tag != ignoreTag })
		}
	}
	return false
}

// importName returns the name a file imports a package as when the import gives none.
//   - a package of this module is imported as it declares itself
//   - any other package is imported as the last element of its path that is no major version
func importName(imported string, declared map[string]string) string {
	if directory, local := strings.CutPrefix(imported, modulePrefix); local && declared[directory] != "" {
		return declared[directory]
	}
	name := path.Base(imported)
	if digits, versioned := strings.CutPrefix(name, "v"); versioned {
		if _, err := strconv.Atoi(digits); err == nil {
			return path.Base(path.Dir(imported))
		}
	}
	return name
}

// selectionKey returns the key a selector of name marks from a file of directory.
//   - an exported name is selected from any package
//   - an unexported name is selected from its own package only
func selectionKey(directory, name string) string {
	if ast.IsExported(name) {
		return name
	}
	return directory + " " + name
}

// receiverType returns the name of the type a method is declared on.
func receiverType(method *ast.FuncDecl) string {
	expression := method.Recv.List[0].Type
	for {
		switch typed := expression.(type) {
		case *ast.StarExpr:
			expression = typed.X
		case *ast.ParenExpr:
			expression = typed.X
		case *ast.IndexExpr:
			expression = typed.X
		case *ast.IndexListExpr:
			expression = typed.X
		case *ast.Ident:
			return typed.Name
		default:
			return ""
		}
	}
}

// declarationWalk is the state of one walk of names from the commands.
//   - named maps a package directory and a name to the functions and the type so named
//   - reached holds the keys of named the walk met
//   - selected holds the selection keys the walk met
//   - walked holds every declaration the walk entered
type declarationWalk struct {
	named    map[string][]reachNode
	reached  map[string]bool
	selected map[string]bool
	walked   map[ast.Node]bool
	queue    []reachNode
}

// reach marks the functions and the type of a name in a package directory as reached.
func (w *declarationWalk) reach(directory, name string) {
	if key := directory + " " + name; !w.reached[key] {
		w.reached[key] = true
		w.queue = append(w.queue, w.named[key]...)
	}
}

// drain enters every queued declaration and follows the names it writes.
//   - an identifier that no local declaration shadows reaches its name in the file's package
//   - a selector on an import of this module reaches its name in that package
//   - any other selector marks its name as selected
//   - an interface marks each method it declares as selected, because it needs the method of every type it holds, called or not
//   - a bare key of a composite literal is a field or a constant, never a function or a type
//   - the name a function declaration declares is no reference, and a method's is in no scope
func (w *declarationWalk) drain() {
	for len(w.queue) > 0 {
		next := w.queue[len(w.queue)-1]
		w.queue = w.queue[:len(w.queue)-1]
		if w.walked[next.node] {
			continue
		}
		w.walked[next.node] = true
		file := next.file
		var declared *ast.Ident
		if function, isFunction := next.node.(*ast.FuncDecl); isFunction {
			declared = function.Name
		}
		var visit func(ast.Node) bool
		visit = func(node ast.Node) bool {
			switch typed := node.(type) {
			case *ast.InterfaceType:
				for _, method := range typed.Methods.List {
					for _, name := range method.Names {
						w.selected[selectionKey(file.directory, name.Name)] = true
					}
				}
			case *ast.KeyValueExpr:
				if _, bare := typed.Key.(*ast.Ident); bare {
					ast.Inspect(typed.Value, visit)
					return false
				}
			case *ast.SelectorExpr:
				if qualifier, isIdent := typed.X.(*ast.Ident); isIdent && qualifier.Obj == nil {
					if directory, imported := file.imports[qualifier.Name]; imported {
						if directory != "" {
							w.reach(directory, typed.Sel.Name)
						}
						return false
					}
				}
				w.selected[selectionKey(file.directory, typed.Sel.Name)] = true
				ast.Inspect(typed.X, visit)
				return false
			case *ast.Ident:
				if typed != declared && (typed.Obj == nil || file.syntax.Scope.Lookup(typed.Name) == typed.Obj) {
					w.reach(file.directory, typed.Name)
				}
			}
			return true
		}
		ast.Inspect(next.node, visit)
	}
}

// commandReach is what one walk from the commands read and entered.
//   - files = the production files some build selects
//   - walked holds every declaration the walk entered
//   - support reports whether a package directory is a test-support package
type commandReach struct {
	files   []*reachFile
	walked  map[ast.Node]bool
	support func(directory string) bool
}

// reachFromCommands walks a snapshot's production files from its commands.
//   - a file whose build constraint needs ignoreTag is no part of the package beside it and is no root
//   - a generated file is walked for what it reaches
//   - test-support package = no main package, whose every importing production file outside it is in a test-support package and none in a main package (the rule of landing.newUnconsumedExports)
//   - roots = main of every main package + init, variables and constants of every package that is not test support, because a command links each such package
//   - a method is reached <=> its receiver type is reached and (its name is selected or outside holds its name)
//   - outside = the exported method names of interfaces declared outside the module, whose callers the walk cannot read
//   - the walk reads names, never types, so a name two declarations share reaches both
func reachFromCommands(snapshot gosyntax.Snapshot, outside map[string]bool) (commandReach, error) {
	var files []*reachFile
	declared, commands := map[string]string{}, map[string]bool{}
	importers := map[string][]*reachFile{}
	for _, source := range snapshot.Files {
		if source.Test {
			continue
		}
		syntax, err := source.Syntax()
		if err != nil {
			return commandReach{}, err
		}
		file := &reachFile{source: source, syntax: syntax, directory: path.Dir(source.Path), imports: map[string]string{}}
		for _, spec := range syntax.Imports {
			imported, _ := strconv.Unquote(spec.Path.Value)
			if directory, local := strings.CutPrefix(imported, modulePrefix); local && directory != file.directory {
				importers[directory] = append(importers[directory], file)
			}
		}
		if !neverBuilt(syntax) {
			files = append(files, file)
			declared[file.directory] = syntax.Name.Name
			commands[file.directory] = syntax.Name.Name == "main"
		}
	}
	support := map[string]bool{}
	var testSupport func(directory string) bool
	testSupport = func(directory string) bool {
		if known, found := support[directory]; found {
			return known
		}
		// A directory met again while its answer is pending reads as production, because Go's imports are acyclic.
		support[directory] = false
		support[directory] = !commands[directory] && !slices.ContainsFunc(importers[directory], func(importer *reachFile) bool {
			return importer.syntax.Name.Name == "main" || !testSupport(importer.directory)
		})
		return support[directory]
	}

	walk := declarationWalk{named: map[string][]reachNode{}, reached: map[string]bool{}, selected: map[string]bool{}, walked: map[ast.Node]bool{}}
	var methods []reachNode
	for _, file := range files {
		for _, spec := range file.syntax.Imports {
			imported, _ := strconv.Unquote(spec.Path.Value)
			directory, local := strings.CutPrefix(imported, modulePrefix)
			if !local {
				directory = ""
			}
			name := importName(imported, declared)
			if spec.Name != nil {
				name = spec.Name.Name
			}
			if name == "." && local {
				return commandReach{}, fmt.Errorf("%s imports %s as \".\", and the walk follows a name only through the name its package is imported as", file.source.Path, imported)
			}
			file.imports[name] = directory
		}
		linked := !testSupport(file.directory)
		for _, declaration := range file.syntax.Decls {
			switch typed := declaration.(type) {
			case *ast.FuncDecl:
				if typed.Recv != nil {
					methods = append(methods, reachNode{file, typed})
					continue
				}
				key := file.directory + " " + typed.Name.Name
				walk.named[key] = append(walk.named[key], reachNode{file, typed})
			case *ast.GenDecl:
				for _, spec := range typed.Specs {
					switch spec := spec.(type) {
					case *ast.TypeSpec:
						key := file.directory + " " + spec.Name.Name
						walk.named[key] = append(walk.named[key], reachNode{file, spec})
					case *ast.ValueSpec:
						if linked {
							walk.queue = append(walk.queue, reachNode{file, spec})
						}
					}
				}
			}
		}
	}
	for directory, command := range commands {
		if testSupport(directory) {
			continue
		}
		if command {
			walk.reach(directory, "main")
		}
		walk.reach(directory, "init")
	}
	for grew := true; grew; {
		walk.drain()
		grew = false
		for _, method := range methods {
			declaration, directory := method.node.(*ast.FuncDecl), method.file.directory
			name := declaration.Name.Name
			if !walk.walked[declaration] && walk.reached[directory+" "+receiverType(declaration)] && (walk.selected[selectionKey(directory, name)] || outside[name]) {
				walk.queue, grew = append(walk.queue, method), true
			}
		}
	}
	return commandReach{files, walk.walked, testSupport}, nil
}

// unreachedDeclarations lists the functions and methods of a snapshot's production files that no command reaches, and counts the ones it read.
//   - universe = top-level functions and methods of the files reachFromCommands walks, outside generated files and test-support packages
//   - the walk reads names, never types, so the list is a lower bound
func unreachedDeclarations(snapshot gosyntax.Snapshot, outside map[string]bool) ([]unreached, int, error) {
	reach, err := reachFromCommands(snapshot, outside)
	if err != nil {
		return nil, 0, err
	}
	var list []unreached
	universe := 0
	for _, file := range reach.files {
		if generated, _ := file.source.Generated(); generated || reach.support(file.directory) {
			continue
		}
		for _, declaration := range file.syntax.Decls {
			function, isFunction := declaration.(*ast.FuncDecl)
			if !isFunction {
				continue
			}
			universe++
			if reach.walked[function] {
				continue
			}
			name, from := function.Name.Name, function.Pos()
			if function.Recv != nil {
				name = receiverType(function) + "." + name
			}
			if function.Doc != nil {
				from = function.Doc.Pos()
			}
			list = append(list, unreached{file.source.Path + " " + name, file.source.Line(function.End()) - file.source.Line(from) + 1})
		}
	}
	slices.SortFunc(list, func(a, b unreached) int { return strings.Compare(a.key, b.key) })
	return list, universe, nil
}

// outsidePackages is the go list query for the directory of every package outside the module that a package of the module links.
var outsidePackages = []string{"list", "-deps", "-f", "{{if .Standard}}{{.Dir}}{{else if not .Module.Main}}{{.Dir}}{{end}}", "./cmd/...", "./internal/..."}

// outsideMethods returns the exported method names of every interface declared outside the module at root.
//   - a package outside the module calls a method of this module only through an interface it declares
//   - the walk cannot read that call, so a method so named counts as selected
func outsideMethods(ctx context.Context, root string) (map[string]bool, error) {
	var listed, failure bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{Path: "go", Args: outsidePackages, Dir: root, Stdout: &listed, Stderr: &failure})
	if err == nil && receipt.ExitCode != 0 {
		err = fmt.Errorf("exit %d: %s", receipt.ExitCode, strings.TrimSpace(failure.String()))
	}
	if err != nil {
		return nil, fmt.Errorf("go %s: %w", strings.Join(outsidePackages, " "), err)
	}
	var group sync.WaitGroup
	var guard sync.Mutex
	names := map[string]bool{}
	var failed error
	for directory := range strings.FieldsFuncSeq(listed.String(), func(r rune) bool { return r == '\n' || r == '\r' }) {
		group.Go(func() {
			found, err := interfaceMethods(directory)
			guard.Lock()
			defer guard.Unlock()
			for _, name := range found {
				names[name] = true
			}
			if failed == nil {
				failed = err
			}
		})
	}
	group.Wait()
	return names, failed
}

// interfaceMethods returns the exported method names of the interfaces that the non-test Go files of one directory declare.
func interfaceMethods(directory string) ([]string, error) {
	entries, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") || strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(directory, entry.Name()))
		if err != nil {
			return nil, err
		}
		if !bytes.Contains(data, []byte("interface")) {
			continue
		}
		syntax, err := parser.ParseFile(token.NewFileSet(), entry.Name(), data, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		ast.Inspect(syntax, func(node ast.Node) bool {
			if declared, isInterface := node.(*ast.InterfaceType); isInterface {
				for _, method := range declared.Methods.List {
					for _, name := range method.Names {
						if name.IsExported() {
							names = append(names, name.Name)
						}
					}
				}
			}
			return true
		})
	}
	return names, nil
}

// listDrift compares what a walk found with its kept list, both sorted.
//   - missing = found and not kept
//   - stale = kept and not found
func listDrift(found, kept []string) (missing, stale []string) {
	for _, key := range found {
		if _, listed := slices.BinarySearch(kept, key); !listed {
			missing = append(missing, key)
		}
	}
	for _, key := range kept {
		if _, unreached := slices.BinarySearch(found, key); !unreached {
			stale = append(stale, key)
		}
	}
	return missing, stale
}

// checkoutSource returns the Go files of the checkout's commands and packages, and the interface methods declared outside the module.
func checkoutSource(t *testing.T) (gosyntax.Snapshot, map[string]bool) {
	t.Helper()
	const root = "../.."
	snapshot, err := gosyntax.DiscoverGo(root, "cmd", "internal")
	if err != nil {
		t.Fatal(err)
	}
	outside, err := outsideMethods(t.Context(), root)
	if err != nil {
		t.Fatal(err)
	}
	return snapshot, outside
}

// keptList reads a kept list beside the tests, one entry a line.
//   - a list out of byte order, or one that repeats a line, is refused
func keptList(t *testing.T, name string) []string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	var kept []string
	for line := range strings.Lines(string(data)) {
		if line = strings.TrimSpace(line); line != "" {
			kept = append(kept, line)
		}
	}
	if !slices.IsSorted(kept) || len(slices.Compact(slices.Clone(kept))) != len(kept) {
		t.Fatalf("%s is not sorted or repeats a line, want one entry a line in byte order", name)
	}
	return kept
}

// sortClasses are the classes a kept entry is sorted into, in byte order.
//   - dead = nothing should reach it, and a later commit deletes it
//   - prototype = part of a model family's own path, which stays
//   - shadowed = the walk lists it wrongly, and a rule of the walk takes it off the list
//   - test-held = only tests of live behaviour use it, and it moves beside them or they take the live path
//   - wire = a capability no command offers, which a later commit gives a route
var sortClasses = []string{"dead", "prototype", "shadowed", "test-held", "wire"}

// sortedEntries splits the lines of a kept list into its entries and the class each is sorted into.
//   - line = class + space + entry
//   - a class outside sortClasses is refused
//   - an entry under two classes is refused, because a later commit acts on one
//   - the entries come back in byte order
func sortedEntries(lines []string) ([]string, map[string]string, error) {
	classOf := map[string]string{}
	var entries []string
	for _, line := range lines {
		class, entry, found := strings.Cut(line, " ")
		if !found || !slices.Contains(sortClasses, class) {
			return nil, nil, fmt.Errorf("the line %q opens with no class, want one of %q", line, sortClasses)
		}
		if earlier, twice := classOf[entry]; twice {
			return nil, nil, fmt.Errorf("%s is sorted as %s and as %s, want one class", entry, earlier, class)
		}
		classOf[entry] = class
		entries = append(entries, entry)
	}
	slices.Sort(entries)
	return entries, classOf, nil
}

// sortRecord is the document that gives every group of kept entries its class and its reason.
const sortRecord = "docs/verification/unreached-declarations.txt"

// groupsPart opens the part of sortRecord the tests read, and the text before it is for a person.
const groupsPart = "== groups =="

// The words of a group in sortRecord.
const (
	groupWord         = "group "
	classField        = "class"
	surfaceField      = "surface"
	declarationsField = "declarations"
	linesField        = "lines"
	membersField      = "members"
	schemasField      = "schemas"
	reasonField       = "reason"
	insideSurface     = "inside"
	outsideSurface    = "outside"
)

// reasonLimit is the most statements a group gives as its reason.
//   - source = the plan's step sort-unreached, which asks for a reason in one or two statements
//   - scope = sortRecord
//   - reconsider when a group needs a third statement to name what replaced it
const reasonLimit = 2

// sortGroup is one group of sortRecord, kept entries that share a class and a reason.
//   - members select kept declarations and schemas select kept record schemas
//   - selector = a file, for every kept entry of it, or a file and one entry of it
//   - declarations and lines count the kept declarations, never the schemas
type sortGroup struct {
	name, class, surface      string
	declarations, lines       int
	members, schemas, reasons []string
}

// sortGroups reads the groups of a record's text.
//   - a group opens with groupWord and its name, and ends at an empty line
//   - every other line of a group is a field, a colon, a space and a value
//   - a group states its class, surface, declarations and lines once, and a reason up to reasonLimit times
//   - a members or schemas line names a file, and after it the entries of the file the group holds, or none for all of them
//   - a name stated twice, a field stated twice, an unknown field, a count that is no number and a group that selects nothing are refused
func sortGroups(text string) ([]sortGroup, error) {
	_, part, found := strings.Cut(strings.ReplaceAll(text, "\r\n", "\n"), groupsPart+"\n")
	if !found {
		return nil, fmt.Errorf("the record has no part %q", groupsPart)
	}
	var groups []sortGroup
	seen := map[string]bool{}
	for block := range strings.SplitSeq(part, "\n\n") {
		lines := strings.Split(strings.TrimSpace(block), "\n")
		if lines[0] == "" {
			continue
		}
		name, opened := strings.CutPrefix(lines[0], groupWord)
		if !opened || name == "" || strings.Contains(name, " ") || seen[name] {
			return nil, fmt.Errorf("a group opens with %q, want %q and a name of one word the record gives once", lines[0], groupWord)
		}
		seen[name] = true
		group := sortGroup{name: name}
		stated := map[string]bool{}
		for _, line := range lines[1:] {
			field, value, found := strings.Cut(line, ": ")
			if !found || value == "" {
				return nil, fmt.Errorf("group %s holds the line %q, want a field, a colon, a space and a value", name, line)
			}
			once := true
			var err error
			switch field {
			case classField:
				group.class = value
			case surfaceField:
				group.surface = value
			case declarationsField:
				group.declarations, err = strconv.Atoi(value)
			case linesField:
				group.lines, err = strconv.Atoi(value)
			case membersField, schemasField:
				once = false
				file, entries, _ := strings.Cut(value, " ")
				selectors := []string{file}
				if entries != "" {
					selectors = nil
					for entry := range strings.FieldsSeq(entries) {
						selectors = append(selectors, file+" "+entry)
					}
				}
				if field == membersField {
					group.members = append(group.members, selectors...)
				} else {
					group.schemas = append(group.schemas, selectors...)
				}
			case reasonField:
				once = false
				group.reasons = append(group.reasons, value)
			default:
				return nil, fmt.Errorf("group %s states the field %q, which the record does not have", name, field)
			}
			if err != nil || once && stated[field] {
				return nil, fmt.Errorf("group %s states %s as %q, want it once and its counts as numbers", name, field, value)
			}
			stated[field] = true
		}
		whole := stated[classField] && stated[surfaceField] && stated[declarationsField] && stated[linesField]
		if !whole || len(group.reasons) == 0 || len(group.reasons) > reasonLimit || len(group.members)+len(group.schemas) == 0 {
			return nil, fmt.Errorf("group %s lacks a field, a reason or a selector, or gives more than %d reasons", name, reasonLimit)
		}
		if !slices.Contains(sortClasses, group.class) || group.surface != insideSurface && group.surface != outsideSurface {
			return nil, fmt.Errorf("group %s is of class %q and surface %q, want one of %q and %s or %s", name, group.class, group.surface, sortClasses, insideSurface, outsideSurface)
		}
		groups = append(groups, group)
	}
	return groups, nil
}

// heldEntries gives each group the kept entries its selectors name, and says where the groups and the list differ.
//   - every kept entry is held by one group and by no second
//   - a selector that names no kept entry is refused
//   - a group's class is the class of each entry it holds
func heldEntries(groups []sortGroup, selectors func(sortGroup) []string, entries []string, classOf map[string]string) (map[string][]string, []string) {
	held, holder := map[string][]string{}, map[string]string{}
	var differs []string
	for _, group := range groups {
		for _, selector := range selectors(group) {
			named := []string{selector}
			if !strings.Contains(selector, " ") {
				named = nil
				for _, entry := range entries {
					if strings.HasPrefix(entry, selector+" ") {
						named = append(named, entry)
					}
				}
			} else if classOf[selector] == "" {
				named = nil
			}
			if len(named) == 0 {
				differs = append(differs, fmt.Sprintf("group %s selects %s, which names no kept entry", group.name, selector))
			}
			for _, entry := range named {
				switch {
				case holder[entry] != "":
					differs = append(differs, fmt.Sprintf("%s is in group %s and in group %s, want one group", entry, holder[entry], group.name))
				case classOf[entry] != group.class:
					differs = append(differs, fmt.Sprintf("%s is sorted as %s and its group %s as %s", entry, classOf[entry], group.name, group.class))
				}
				holder[entry] = group.name
				held[group.name] = append(held[group.name], entry)
			}
		}
	}
	for _, entry := range entries {
		if holder[entry] == "" {
			differs = append(differs, fmt.Sprintf("%s is in no group of %s", entry, sortRecord))
		}
	}
	return held, differs
}

// groupFigures says where a group's stated figures differ from what the walk and the surface give.
//   - declarations = the kept declarations the group holds
//   - lines = the source lines of those declarations
//   - surface = inside when every file the group selects lies in a package of the inference surface, outside when none does, and no group lies on both sides
func groupFigures(groups []sortGroup, held map[string][]string, lines map[string]int, inside func(directory string) bool) []string {
	var differs []string
	for _, group := range groups {
		total := 0
		for _, entry := range held[group.name] {
			total += lines[entry]
		}
		if group.declarations != len(held[group.name]) || group.lines != total {
			differs = append(differs, fmt.Sprintf("group %s states %d declarations of %d lines, and the walk gives %d of %d", group.name, group.declarations, group.lines, len(held[group.name]), total))
		}
		for _, selector := range slices.Concat(group.members, group.schemas) {
			file, _, _ := strings.Cut(selector, " ")
			if inside(path.Dir(file)) != (group.surface == insideSurface) {
				differs = append(differs, fmt.Sprintf("group %s states its surface as %s, and %s lies on the other side", group.name, group.surface, file))
			}
		}
	}
	return differs
}

// recordGroups reads the groups of sortRecord from the checkout.
func recordGroups(t *testing.T) []sortGroup {
	t.Helper()
	text, err := os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(sortRecord)))
	if err != nil {
		t.Fatal(err)
	}
	groups, err := sortGroups(string(text))
	if err != nil {
		t.Fatalf("%s: %v", sortRecord, err)
	}
	return groups
}

// unreachedList is the file that keeps the census of declarations no command reaches.
const unreachedList = "unreached_declarations.txt"

// TestEveryProcessDeclarationIsReachedByACommand holds every production function and method outside test support to a command that reaches it.
//   - the checkout's unreached declarations = the entries of unreachedList, both ways
//   - an unreached declaration the list lacks is refused, because the list only shrinks
//   - a listed declaration that a command reaches, or that the tree no longer declares, is refused
//   - every entry carries one of sortClasses, and belongs to one group of sortRecord of that class
//   - every group states the declarations it holds, their lines and its side of the inference surface as the walk and inferencesurface.Packages give them
//   - each fixture row pins one rule of the walk, of the list or of the record on invented files
func TestEveryProcessDeclarationIsReachedByACommand(t *testing.T) {
	t.Parallel()
	const command = "package main\n\nimport \"overgo/internal/lib\"\n\nfunc main() { lib.Use() }\n"
	for _, row := range []struct {
		rule  string
		files map[string]string
		want  []string
	}{
		{"one name reaches the declaration of each build-tag twin, and each twin is listed by its file", map[string]string{
			"internal/lib/lib.go":          "package lib\n\nfunc Use() int { return free() }\n",
			"internal/lib/free_windows.go": "//go:build windows\n\npackage lib\n\nfunc free() int { return 1 }\n\nfunc spare() {}\n",
			"internal/lib/free_unix.go":    "//go:build !windows\n\npackage lib\n\nfunc free() int { return 2 }\n\nfunc spare() {}\n",
		}, []string{"internal/lib/free_unix.go spare", "internal/lib/free_windows.go spare"}},
		{"a package variable reaches the function it holds, called or not", map[string]string{
			"internal/lib/lib.go": "package lib\n\nvar held = map[string]func(){\"start\": start}\n\nfunc Use() {}\n\nfunc start() {}\n\nfunc idle() {}\n",
		}, []string{"internal/lib/lib.go idle"}},
		{"a method is reached through its receiver type and its name", map[string]string{
			"cmd/tool/main.go": "package main\n\nimport \"overgo/internal/lib\"\n\nfunc main() {\n\tvar local struct{ grow int }\n\t_, _ = local.grow, lib.Box{}.Area()\n}\n",
			"internal/lib/lib.go": "package lib\n\ntype Box struct{}\n\nfunc (Box) Area() int { return 1 }\n\nfunc (Box) Spare() int { return 2 }\n\nfunc (*Box) String() string { return \"\" }\n\nfunc (Box) grow() {}\n\n" +
				"type Lid[T any] struct{}\n\nfunc (*Lid[T]) Area() int { return 3 }\n",
		}, []string{"internal/lib/lib.go Box.Spare", "internal/lib/lib.go Box.grow", "internal/lib/lib.go Lid.Area"}},
		{"an interface reaches the method it declares, called or not", map[string]string{
			"internal/lib/lib.go": "package lib\n\ntype node interface{ isNode() }\n\ntype leaf struct{}\n\nfunc (leaf) isNode() {}\n\nfunc (leaf) spare() {}\n\nfunc Use() node { return leaf{} }\n",
		}, []string{"internal/lib/lib.go leaf.spare"}},
		{"a function only a test calls is listed", map[string]string{
			"internal/lib/lib.go":      "package lib\n\nfunc Use() {}\n\nfunc OnlyTested() {}\n",
			"internal/lib/lib_test.go": "package lib\n\nfunc TestOnly(t *testing.T) { OnlyTested() }\n",
		}, []string{"internal/lib/lib.go OnlyTested"}},
		{"a test-support package is skipped, and what it alone calls is listed", map[string]string{
			"internal/lib/lib.go":          "package lib\n\nfunc Use() {}\n\nfunc OnlySupport() {}\n",
			"internal/libtest/libtest.go":  "package libtest\n\nimport (\n\t\"overgo/internal/deep\"\n\t\"overgo/internal/lib\"\n)\n\nfunc Fixture() { deep.Helper(); lib.OnlySupport() }\n\nfunc spare() {}\n",
			"internal/deep/deep.go":        "package deep\n\nfunc Helper() {}\n\nfunc spare() {}\n",
			"internal/other/other_test.go": "package other\n\nimport \"overgo/internal/libtest\"\n\nfunc TestOther(t *testing.T) { libtest.Fixture() }\n",
		}, []string{"internal/lib/lib.go OnlySupport"}},
		{"an init of a package a command links reaches what it calls", map[string]string{
			"cmd/tool/main.go":    "package main\n\nimport \"overgo/internal/lib\"\n\nfunc main() { println(lib.Name) }\n",
			"internal/lib/lib.go": "package lib\n\nconst Name = \"lib\"\n\nfunc init() { register() }\n\nfunc register() {}\n\nfunc spare() {}\n",
		}, []string{"internal/lib/lib.go spare"}},
		{"a main file no build selects is no root, no part of the package beside it, and not listed", map[string]string{
			"internal/lib/lib.go": "package lib\n\nfunc Use() {}\n\nfunc emit() {}\n",
			"internal/lib/gen.go": "//go:build ignore\n\npackage main\n\nfunc main() { emit() }\n\nfunc emit() {}\n",
		}, []string{"internal/lib/lib.go emit"}},
		{"a generated file reaches what it calls and is not listed", map[string]string{
			"internal/lib/lib.go":          "package lib\n\nfunc Use() {}\n\nfunc fromGenerated() {}\n\nfunc spare() {}\n",
			"internal/lib/zz_generated.go": "// Code generated by fixture. DO NOT EDIT.\n\npackage lib\n\nfunc init() { fromGenerated() }\n\nfunc unused() {}\n",
		}, []string{"internal/lib/lib.go spare"}},
		{"a local that shadows a function, and a field key of its name, reach nothing", map[string]string{
			"internal/lib/lib.go": "package lib\n\ntype options struct{ limit int }\n\nfunc limit() int { return 1 }\n\nfunc scale() int { return 2 }\n\nfunc Use() options { scale := 3; return options{limit: scale} }\n",
		}, []string{"internal/lib/lib.go limit", "internal/lib/lib.go scale"}},
		{"a package is followed by the name it is imported as, and a local of that name is no package", map[string]string{
			"cmd/tool/main.go":       "package main\n\nimport (\n\t\"overgo/internal/go-lib\"\n\tpeer \"overgo/internal/peer\"\n)\n\nfunc main() {\n\tlib.Use()\n\tpeer.Use()\n\tpeer := struct{ Spare func() }{}\n\tpeer.Spare()\n}\n",
			"internal/go-lib/lib.go": "package lib\n\nfunc Use() {}\n\nfunc Spare() {}\n",
			"internal/peer/peer.go":  "package peer\n\nfunc Use() {}\n\nfunc Spare() {}\n",
		}, []string{"internal/go-lib/lib.go Spare", "internal/peer/peer.go Spare"}},
	} {
		contents := map[string][]byte{"cmd/tool/main.go": []byte(command)}
		for name, content := range row.files {
			contents[name] = []byte(content)
		}
		snapshot, err := gosyntax.Snapshot{}.Overlay(contents)
		if err != nil {
			t.Fatal(err)
		}
		found, _, err := unreachedDeclarations(snapshot, map[string]bool{"String": true})
		var keys []string
		for _, declaration := range found {
			keys = append(keys, declaration.key)
		}
		if err != nil || !slices.Equal(keys, row.want) {
			t.Errorf("%s: unreached = %q, %v, want %q", row.rule, keys, err, row.want)
		}
	}
	if missing, stale := listDrift([]string{"a", "b"}, []string{"b", "c"}); !slices.Equal(missing, []string{"a"}) || !slices.Equal(stale, []string{"c"}) {
		t.Errorf("the comparison names %q as unreached and unlisted and %q as listed and reached, want a and c", missing, stale)
	}

	// fixtureLines is the source lines the record rows give every kept declaration, and fixtureInside the one directory they put inside the surface.
	const fixtureLines, fixtureInside = 3, "internal/in"
	block := func(name, class, surface string, declarations, lines int, rest ...string) string {
		return fmt.Sprintf("%s%s\n%s: %s\n%s: %s\n%s: %d\n%s: %d\n%s\n\n", groupWord, name, classField, class, surfaceField, surface,
			declarationsField, declarations, linesField, lines, strings.Join(rest, "\n"))
	}
	const reason = reasonField + ": Nothing calls it."
	for _, row := range []struct {
		rule   string
		list   []string
		record string
		want   []string
	}{
		{"a group holds every kept declaration of a file, or the ones it names, and states their count, their lines and its side of the surface",
			[]string{"dead internal/in/a.go Old", "dead internal/in/a.go Older", "wire internal/out/b.go Box.Open"},
			block("old", "dead", insideSurface, 2, 2*fixtureLines, membersField+": internal/in/a.go", reason) +
				block("open", "wire", outsideSurface, 1, fixtureLines, membersField+": internal/out/b.go Box.Open", reason, reason), nil},
		{"a kept declaration in two groups, and one in no group, are refused",
			[]string{"dead internal/in/a.go Old", "dead internal/in/a.go Older", "dead internal/in/c.go Lost"},
			block("one", "dead", insideSurface, 1, fixtureLines, membersField+": internal/in/a.go Old", reason) +
				block("two", "dead", insideSurface, 2, 2*fixtureLines, membersField+": internal/in/a.go", reason),
			[]string{"internal/in/a.go Old is in group one and in group two", "internal/in/c.go Lost is in no group"}},
		{"a group of another class than its declaration, and a selector that names no kept declaration, are refused",
			[]string{"dead internal/in/a.go Old"},
			block("old", "wire", insideSurface, 1, fixtureLines, membersField+": internal/in/a.go Old Gone", reason),
			[]string{"sorted as dead and its group old as wire", "selects internal/in/a.go Gone, which names no kept entry"}},
		{"a count the walk does not give, and a side of the surface a file is not on, are refused",
			[]string{"dead internal/in/a.go Old", "dead internal/out/b.go Odd"},
			block("old", "dead", insideSurface, 1, 2*fixtureLines, membersField+": internal/in/a.go", membersField+": internal/out/b.go", reason),
			[]string{"states 1 declarations of 6 lines, and the walk gives 2 of 6", "internal/out/b.go lies on the other side"}},
		{"lines the walk does not give are refused",
			[]string{"dead internal/in/a.go Old"},
			block("old", "dead", insideSurface, 1, 2*fixtureLines, membersField+": internal/in/a.go", reason),
			[]string{"states 1 declarations of 6 lines, and the walk gives 1 of 3"}},
		{"a group that holds record schemas alone states no declaration, and its side is its schemas' side",
			nil, block("kept", "dead", outsideSurface, 0, 0, schemasField+": internal/in/a.go overgo/kept/v1", reason),
			[]string{"internal/in/a.go lies on the other side"}},
		{"a class outside the classes is refused", []string{"gone internal/in/a.go Old"}, "", []string{"opens with no class"}},
		{"an entry under two classes is refused", []string{"dead internal/in/a.go Old", "wire internal/in/a.go Old"}, "", []string{"sorted as dead and as wire"}},
		{"a group the record gives twice is refused", nil,
			strings.Repeat(block("old", "dead", insideSurface, 0, 0, membersField+": internal/in/a.go", reason), 2), []string{"a name of one word the record gives once"}},
		{"a field the record does not have is refused", nil,
			block("old", "dead", insideSurface, 0, 0, membersField+": internal/in/a.go", reason, "owner: nobody"), []string{"states the field \"owner\""}},
		{"a group with more reasons than reasonLimit is refused", nil,
			block("old", "dead", insideSurface, 0, 0, membersField+": internal/in/a.go", reason, reason, reason), []string{"lacks a field, a reason or a selector"}},
		{"a class or a side the record does not have is refused", nil,
			block("old", "dead", "beside", 0, 0, membersField+": internal/in/a.go", reason), []string{"surface \"beside\""}},
	} {
		var got []string
		entries, classOf, err := sortedEntries(row.list)
		groups, recordErr := sortGroups(groupsPart + "\n" + row.record)
		for _, failure := range []error{err, recordErr} {
			if failure != nil {
				got = append(got, failure.Error())
			}
		}
		if len(got) == 0 {
			sizes := map[string]int{}
			for _, entry := range entries {
				sizes[entry] = fixtureLines
			}
			held, differs := heldEntries(groups, func(group sortGroup) []string { return group.members }, entries, classOf)
			got = append(differs, groupFigures(groups, held, sizes, func(directory string) bool { return directory == fixtureInside })...)
		}
		matched := len(got) == len(row.want)
		for index := 0; matched && index < len(got); index++ {
			matched = strings.Contains(got[index], row.want[index])
		}
		if !matched {
			t.Errorf("%s: refusals = %q, want one for each of %q", row.rule, got, row.want)
		}
	}

	snapshot, outside := checkoutSource(t)
	found, universe, err := unreachedDeclarations(snapshot, outside)
	if err != nil {
		t.Fatal(err)
	}
	if universe == 0 || len(outside) == 0 {
		t.Fatalf("the walk read %d declarations and %d interface methods outside the module, want some of each", universe, len(outside))
	}
	kept, classOf, err := sortedEntries(keptList(t, unreachedList))
	if err != nil {
		t.Fatalf("%s: %v", unreachedList, err)
	}
	var keys []string
	lines, sizes := 0, map[string]int{}
	for _, declaration := range found {
		keys, lines = append(keys, declaration.key), lines+declaration.lines
		sizes[declaration.key] += declaration.lines
	}
	missing, stale := listDrift(slices.Compact(keys), kept)
	for _, key := range missing {
		t.Errorf("no command reaches %s, so wire it to a command, move it beside its tests or delete it, because %s only shrinks", key, unreachedList)
	}
	for _, key := range stale {
		t.Errorf("%s lists %s, which a command reaches or the tree no longer declares, so delete its line and take it out of its group in %s", unreachedList, key, sortRecord)
	}
	inside := surfaceDirectories(t)
	groups := recordGroups(t)
	held, differs := heldEntries(groups, func(group sortGroup) []string { return group.members }, kept, classOf)
	for _, differ := range append(differs, groupFigures(groups, held, sizes, func(directory string) bool { return inside[directory] })...) {
		t.Errorf("%s: %s", sortRecord, differ)
	}
	for _, class := range sortClasses {
		var within, beyond struct{ declarations, lines int }
		for _, entry := range kept {
			if classOf[entry] != class {
				continue
			}
			file, _, _ := strings.Cut(entry, " ")
			side := &beyond
			if inside[path.Dir(file)] {
				side = &within
			}
			side.declarations++
			side.lines += sizes[entry]
		}
		t.Logf("class=%s inside=%d of %d lines outside=%d of %d lines", class, within.declarations, within.lines, beyond.declarations, beyond.lines)
	}
	t.Logf("declarations=%d unreached=%d lines=%d groups=%d", universe, len(found), lines, len(groups))
}

// surfaceDirectories returns the package directories of the inference surface, as inferencesurface.Packages lists them.
func surfaceDirectories(t *testing.T) map[string]bool {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	packages, err := inferencesurface.Packages(t.Context(), root)
	if err != nil {
		t.Fatal(err)
	}
	inside := map[string]bool{}
	for _, pkg := range packages {
		directory, err := filepath.Rel(root, pkg.Dir)
		if err != nil {
			t.Fatal(err)
		}
		inside[filepath.ToSlash(directory)] = true
	}
	return inside
}
