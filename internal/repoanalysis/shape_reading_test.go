package repoanalysis

import (
	"cmp"
	"flag"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"path"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
)

// The thresholds of the shape reading.
const (
	// runLength is how many consecutive statements a repeated run spans.
	//   - source = the plan's step shape-reading/analyzer, which reads runs of five statements
	//   - scope = repeated runs
	//   - reconsider when runs of this length repeat by shape alone in code that shares no logic
	runLength = 5

	// runFunctions is the fewest functions a run's shape repeats in.
	//   - source = the plan's step shape-reading/analyzer
	//   - a shape repeated inside one function is that function's own structure, and a common component needs a second function to serve
	//   - scope = repeated runs
	//   - reconsider when a repeat inside one function is to be consolidated
	runFunctions = 2

	// chainPairs is the fewest consecutive call-and-test pairs on one receiver that a chain holds.
	//   - source = the plan's step shape-reading/analyzer
	//   - scope = call-and-test chains
	//   - reconsider when a helper that folds two pairs is read as worth its lines
	chainPairs = 3

	// kindConstants is the fewest named constants a switch's cases name for the switch to decide by kind.
	//   - source = the plan's step shape-reading/analyzer
	//   - a switch over two constants is a two-way choice, which an if writes as well
	//   - scope = switches by kind
	//   - reconsider when two-way switches are to be read as kinds
	kindConstants = 3

	// flagArguments is the fewest arguments a method of flag.FlagSet takes when it defines a flag.
	//   - source = package flag, whose defining methods take a name, a usage and a value, a pointer or a function, and whose Set and Init take two
	//   - scope = the class flag-definitions
	//   - reconsider when package flag gains a method of three or more arguments that defines no flag
	flagArguments = 3

	// largestShown is how many of the largest groups, chains and families the repository's listing logs.
	//   - source = the plan's step shape-reading/real-repeats, which names seven groups, and its step kinds-reading, which reads ten switches
	//   - scope = the log of the repository's listing, which asserts nothing
	//   - reconsider when a reader of the log needs more than it shows
	largestShown = 10
)

// The classes of a statement and of a repeated run.
//   - pairClass = a statement of a call-and-test pair
//   - flagClass = a statement that defines a flag through package flag or a flag.FlagSet
//   - otherClass = any other statement
const (
	pairClass  = "call-and-test"
	flagClass  = "flag-definitions"
	otherClass = "other"
)

// runClasses orders the classes, and a window of runLength statements is of the first class one of its statements holds.
//   - pairClass first because a window that tests an error is how Go handles the error, and a common component would not remove the test
//   - otherClass last, so a window of otherClass holds repeated logic alone
var runClasses = []string{pairClass, flagClass, otherClass}

// flagDefinitions maps the name of each method of flag.FlagSet that defines a flag to the number of arguments it takes.
//   - package flag's functions of the same names take the same arguments, so one map serves a call on the package and on a flag set
var flagDefinitions = func() map[string]int {
	definitions := map[string]int{}
	set := reflect.TypeFor[*flag.FlagSet]()
	for method := range set.Methods() {
		if arguments := method.Type.NumIn() - 1; arguments >= flagArguments {
			definitions[method.Name] = arguments
		}
	}
	return definitions
}()

// shapePlace is the lines of one file that a finding takes, and the function that holds them.
type shapePlace struct {
	file, function string
	from, to       int
}

// lines returns how many source lines a place spans.
func (p shapePlace) lines() int {
	return p.to - p.from + 1
}

// String writes a place as its file, its lines and its function.
func (p shapePlace) String() string {
	return fmt.Sprintf("%s:%d-%d %s", p.file, p.from, p.to, p.function)
}

// comparePlaces orders places by file, then by first line, then by last line.
func comparePlaces(a, b shapePlace) int {
	return cmp.Or(strings.Compare(a.file, b.file), cmp.Compare(a.from, b.from), cmp.Compare(a.to, b.to))
}

// joinPlaces writes places in order, separated by commas.
func joinPlaces(places []shapePlace) string {
	written := make([]string, len(places))
	for index, place := range places {
		written[index] = place.String()
	}
	return strings.Join(written, ", ")
}

// coveredLines counts the lines a set of places covers.
//   - a line two places share counts once
func coveredLines(places []shapePlace) int {
	places = slices.SortedFunc(slices.Values(places), comparePlaces)
	covered, file, last := 0, "", 0
	for _, place := range places {
		if place.file != file {
			file, last = place.file, 0
		}
		if place.to > last {
			covered += place.to - max(place.from-1, last)
			last = place.to
		}
	}
	return covered
}

// unionFind joins indexes into sets, each set named by its root.
type unionFind []int

// newUnionFind returns size indexes, each in a set of its own.
func newUnionFind(size int) unionFind {
	sets := make(unionFind, size)
	for index := range sets {
		sets[index] = index
	}
	return sets
}

// root returns the index that names the set of index.
func (u unionFind) root(index int) int {
	for u[index] != index {
		u[index] = u[u[index]]
		index = u[index]
	}
	return index
}

// join puts the sets of a and b into one.
func (u unionFind) join(a, b int) {
	u[u.root(a)] = u.root(b)
}

// callPair is one call-and-test pair of a statement list.
//   - from and end index the statements it takes, end excluded
//   - receiver = the expression the call selects its method from, "" for a function or a function of a package
type callPair struct {
	receiver  string
	from, end int
}

// statementList is one list of consecutive statements inside a function, which is a block or the body of a case.
//   - function = the declaration that holds the list, a function literal's list among them
//   - shapes holds the interned shape of each statement, and classes the class of each
//   - pairs holds the call-and-test pairs of the list in order
type statementList struct {
	file       *reachFile
	function   string
	statements []ast.Stmt
	shapes     []int
	classes    []string
	pairs      []callPair
}

// newStatementList reads the shapes, the classes and the call-and-test pairs of one list.
//   - shapes maps each shape met so far to its number, and gains the shapes the list adds
func newStatementList(file *reachFile, function string, statements []ast.Stmt, shapes map[string]int) *statementList {
	list := &statementList{file: file, function: function, statements: statements, classes: make([]string, len(statements))}
	for _, statement := range statements {
		shape := statementShape(statement)
		number, known := shapes[shape]
		if !known {
			number = len(shapes)
			shapes[shape] = number
		}
		list.shapes = append(list.shapes, number)
	}
	for index := 0; index < len(statements); {
		if pair, found := callTest(file, statements, index); found {
			list.pairs = append(list.pairs, pair)
			for ; index < pair.end; index++ {
				list.classes[index] = pairClass
			}
			continue
		}
		list.classes[index] = otherClass
		if definesFlag(statements[index]) {
			list.classes[index] = flagClass
		}
		index++
	}
	return list
}

// place returns the place of the statements from to end, end excluded.
func (l *statementList) place(from, end int) shapePlace {
	source := l.file.source
	return shapePlace{source.Path, l.function, source.Line(l.statements[from].Pos()), source.Line(l.statements[end-1].End())}
}

// statementShape writes a statement's syntax tree with every identifier and every literal's value left out.
//   - kept = the kind of each node, its nesting, the token of each operator, assignment, branch and declaration, and the kind of each literal
//   - comments are left out, because they change no behaviour
func statementShape(statement ast.Stmt) string {
	var shape strings.Builder
	ast.Inspect(statement, func(node ast.Node) bool {
		operator := token.ILLEGAL
		switch typed := node.(type) {
		case nil:
			shape.WriteByte(')')
			return true
		case *ast.CommentGroup:
			return false
		case *ast.BasicLit:
			operator = typed.Kind
		case *ast.BinaryExpr:
			operator = typed.Op
		case *ast.UnaryExpr:
			operator = typed.Op
		case *ast.AssignStmt:
			operator = typed.Tok
		case *ast.IncDecStmt:
			operator = typed.Tok
		case *ast.BranchStmt:
			operator = typed.Tok
		case *ast.GenDecl:
			operator = typed.Tok
		case *ast.RangeStmt:
			operator = typed.Tok
		}
		shape.WriteByte('(')
		shape.WriteString(reflect.TypeOf(node).String())
		if operator != token.ILLEGAL {
			shape.WriteString(operator.String())
		}
		return true
	})
	return shape.String()
}

// definesFlag reports whether a statement is, or assigns from, a call that defines a flag.
//   - call that defines a flag = a selector call of a name in flagDefinitions, with the arguments that name takes
func definesFlag(statement ast.Stmt) bool {
	var value ast.Expr
	switch typed := statement.(type) {
	case *ast.ExprStmt:
		value = typed.X
	case *ast.AssignStmt:
		if len(typed.Rhs) == 1 {
			value = typed.Rhs[0]
		}
	}
	call, isCall := value.(*ast.CallExpr)
	if !isCall {
		return false
	}
	selector, isSelector := call.Fun.(*ast.SelectorExpr)
	if !isSelector {
		return false
	}
	arguments, defines := flagDefinitions[selector.Sel.Name]
	return defines && len(call.Args) == arguments
}

// assignedCall returns the call an assignment takes its one value from, and the name of the last value it assigns.
//   - an assignment of any other form returns nil
func assignedCall(statement ast.Stmt) (*ast.CallExpr, string) {
	assignment, isAssignment := statement.(*ast.AssignStmt)
	if !isAssignment || len(assignment.Rhs) != 1 {
		return nil, ""
	}
	call, isCall := assignment.Rhs[0].(*ast.CallExpr)
	last, isIdent := assignment.Lhs[len(assignment.Lhs)-1].(*ast.Ident)
	if !isCall || !isIdent {
		return nil, ""
	}
	return call, last.Name
}

// testsAndReturns reports whether an if tests name against nil and its body ends in a return.
func testsAndReturns(test *ast.IfStmt, name string) bool {
	condition, isBinary := test.Cond.(*ast.BinaryExpr)
	if !isBinary || condition.Op != token.NEQ || len(test.Body.List) == 0 {
		return false
	}
	tested, isTested := condition.X.(*ast.Ident)
	none, isNone := condition.Y.(*ast.Ident)
	_, returns := test.Body.List[len(test.Body.List)-1].(*ast.ReturnStmt)
	return isTested && isNone && tested.Name == name && none.Name == "nil" && returns
}

// callReceiver returns the expression a call selects its method from.
//   - a function, and a function of a package the file imports, have no receiver and return ""
func callReceiver(file *reachFile, call *ast.CallExpr) string {
	selector, isSelector := call.Fun.(*ast.SelectorExpr)
	if !isSelector {
		return ""
	}
	if qualifier, isIdent := selector.X.(*ast.Ident); isIdent && qualifier.Obj == nil {
		if _, imported := file.imports[qualifier.Name]; imported {
			return ""
		}
	}
	return types.ExprString(selector.X)
}

// callTest returns the call-and-test pair that opens at statements[index], when one does.
//   - pair = an assignment from a call, followed by an if that tests the last value it assigns against nil and ends in a return
//   - an if whose init is that assignment is a pair of one statement
func callTest(file *reachFile, statements []ast.Stmt, index int) (callPair, bool) {
	var call *ast.CallExpr
	var name string
	var test *ast.IfStmt
	end := index + 1
	if opening, isIf := statements[index].(*ast.IfStmt); isIf && opening.Init != nil {
		call, name = assignedCall(opening.Init)
		test = opening
	} else if end < len(statements) {
		call, name = assignedCall(statements[index])
		if following, isIf := statements[end].(*ast.IfStmt); isIf && following.Init == nil {
			test = following
		}
		end++
	}
	if call == nil || test == nil || !testsAndReturns(test, name) {
		return callPair{}, false
	}
	return callPair{callReceiver(file, call), index, end}, true
}

// repeatGroup is one shape of consecutive statements that repeats in runFunctions or more functions.
//   - places = each place the shape takes, the overlapping windows of one list joined into one place
//   - class = the class of each of its windows
//   - lines = the lines of its places
type repeatGroup struct {
	class  string
	places []shapePlace
	lines  int
}

// String writes a group as its class, its lines and its places.
func (g repeatGroup) String() string {
	return fmt.Sprintf("run %s lines=%d places=%d %s", g.class, g.lines, len(g.places), joinPlaces(g.places))
}

// windowAt names the window of runLength statements that opens at offset in a list.
type windowAt struct{ list, offset int }

// repeatedRuns groups the windows of runLength statements whose shape repeats in runFunctions or more functions, and counts the lines the groups cover.
//   - a shape's class = the first of runClasses that a statement of any of its windows holds
//   - a shape's group joins the group of the shape after it <=> both are of one class and every window of the one is followed by a window of the other, so one longer repeat of one class is one group
//   - a line two groups share counts once in the lines covered
func repeatedRuns(lists []*statementList) ([]repeatGroup, int) {
	numbers := map[[runLength]int]int{}
	var windows [][]windowAt
	keys := make([][]int, len(lists))
	for index, list := range lists {
		for offset := 0; offset+runLength <= len(list.shapes); offset++ {
			key := [runLength]int(list.shapes[offset : offset+runLength])
			number, known := numbers[key]
			if !known {
				number = len(windows)
				numbers[key] = number
				windows = append(windows, nil)
			}
			windows[number] = append(windows[number], windowAt{index, offset})
			keys[index] = append(keys[index], number)
		}
	}
	repeats := make([]bool, len(windows))
	classes := make([]string, len(windows))
	for number, at := range windows {
		functions, held := map[string]bool{}, map[string]bool{}
		for _, window := range at {
			list := lists[window.list]
			functions[list.file.source.Path+" "+list.function] = true
			for _, class := range list.classes[window.offset : window.offset+runLength] {
				held[class] = true
			}
		}
		repeats[number] = len(functions) >= runFunctions
		classes[number] = runClasses[slices.IndexFunc(runClasses, func(class string) bool { return held[class] })]
	}
	followedBy := func(window windowAt) int {
		if next := window.offset + 1; next < len(keys[window.list]) {
			return keys[window.list][next]
		}
		return -1
	}
	sets := newUnionFind(len(windows))
	for number, at := range windows {
		next := followedBy(at[0])
		if !repeats[number] || next < 0 || next == number || !repeats[next] || classes[next] != classes[number] || len(windows[next]) != len(at) {
			continue
		}
		if !slices.ContainsFunc(at, func(window windowAt) bool { return followedBy(window) != next }) {
			sets.join(number, next)
		}
	}
	members := map[int][]windowAt{}
	for number, at := range windows {
		if repeats[number] {
			members[sets.root(number)] = append(members[sets.root(number)], at...)
		}
	}
	var groups []repeatGroup
	var every []shapePlace
	for root, at := range members {
		slices.SortFunc(at, func(a, b windowAt) int { return cmp.Or(cmp.Compare(a.list, b.list), cmp.Compare(a.offset, b.offset)) })
		group := repeatGroup{class: classes[root]}
		for index := 0; index < len(at); {
			list, from, end := lists[at[index].list], at[index].offset, at[index].offset+runLength
			for index++; index < len(at) && lists[at[index].list] == list && at[index].offset <= end; index++ {
				end = max(end, at[index].offset+runLength)
			}
			place := list.place(from, end)
			group.places = append(group.places, place)
			group.lines += place.lines()
		}
		slices.SortFunc(group.places, comparePlaces)
		groups = append(groups, group)
		every = append(every, group.places...)
	}
	slices.SortFunc(groups, func(a, b repeatGroup) int {
		return cmp.Or(cmp.Compare(b.lines, a.lines), comparePlaces(a.places[0], b.places[0]))
	})
	return groups, coveredLines(every)
}

// callChain is chainPairs or more consecutive call-and-test pairs whose calls share one receiver.
type callChain struct {
	place    shapePlace
	receiver string
	pairs    int
}

// String writes a chain as its receiver, its pairs, its lines and its place.
func (c callChain) String() string {
	return fmt.Sprintf("chain receiver=%s pairs=%d lines=%d %s", c.receiver, c.pairs, c.place.lines(), c.place)
}

// callChains lists the chains of every statement list, largest first.
//   - a pair whose call has no receiver ends a chain and opens none
func callChains(lists []*statementList) []callChain {
	var chains []callChain
	for _, list := range lists {
		for index := 0; index < len(list.pairs); {
			opening := list.pairs[index]
			next := index + 1
			for next < len(list.pairs) && list.pairs[next].from == list.pairs[next-1].end && list.pairs[next].receiver == opening.receiver {
				next++
			}
			if opening.receiver != "" && next-index >= chainPairs {
				chains = append(chains, callChain{list.place(opening.from, list.pairs[next-1].end), opening.receiver, next - index})
			}
			index = next
		}
	}
	slices.SortFunc(chains, func(a, b callChain) int {
		return cmp.Or(cmp.Compare(b.place.lines(), a.place.lines()), comparePlaces(a.place, b.place))
	})
	return chains
}

// kindSwitch is one switch whose cases name kindConstants or more named constants.
//   - constants = the keys of namedConstants its cases name, in byte order
type kindSwitch struct {
	place     shapePlace
	constants []string
}

// switchFamily is the switches that share a named constant, transitively.
//   - label = the type most of its constants are declared with, a tie going to the type first in byte order
//   - constants = every constant its switches name, in byte order
//   - lines = the lines of its switches
type switchFamily struct {
	label     string
	switches  []kindSwitch
	constants []string
	lines     int
}

// String writes a family as its label, its counts and the places of its switches.
func (f switchFamily) String() string {
	functions, packages := map[string]bool{}, map[string]bool{}
	places := make([]shapePlace, len(f.switches))
	for index, kind := range f.switches {
		functions[kind.place.file+" "+kind.place.function] = true
		packages[path.Dir(kind.place.file)] = true
		places[index] = kind.place
	}
	return fmt.Sprintf("family %s switches=%d lines=%d functions=%d packages=%d constants=%d %s",
		f.label, len(f.switches), f.lines, len(functions), len(packages), len(f.constants), joinPlaces(places))
}

// namedConstants maps each constant the files declare at their top level to the type it is declared with.
//   - key = package directory + space + name
//   - type = package directory + "." + the type its spec writes or the spec it repeats writes, or the package directory alone for an untyped constant
func namedConstants(files []*reachFile) map[string]string {
	constants := map[string]string{}
	for _, file := range files {
		for _, declaration := range file.syntax.Decls {
			general, isGeneral := declaration.(*ast.GenDecl)
			if !isGeneral || general.Tok != token.CONST {
				continue
			}
			declared := file.directory
			for _, spec := range general.Specs {
				value := spec.(*ast.ValueSpec)
				switch {
				case value.Type != nil:
					declared = file.directory + "." + types.ExprString(value.Type)
				case len(value.Values) > 0:
					declared = file.directory
				}
				for _, name := range value.Names {
					if name.Name != "_" {
						constants[file.directory+" "+name.Name] = declared
					}
				}
			}
		}
	}
	return constants
}

// switchConstants returns the keys of namedConstants that a switch's cases name, in byte order.
//   - a bare name resolves in the package of the file, and a name a function declares shadows it
//   - a selected name resolves through the name the file imports a package of the tree as
func switchConstants(file *reachFile, statement *ast.SwitchStmt, constants map[string]string) []string {
	var named []string
	for _, clause := range statement.Body.List {
		for _, expression := range clause.(*ast.CaseClause).List {
			key := ""
			switch typed := expression.(type) {
			case *ast.Ident:
				if typed.Obj == nil || file.syntax.Scope.Lookup(typed.Name) == typed.Obj {
					key = file.directory + " " + typed.Name
				}
			case *ast.SelectorExpr:
				if qualifier, isIdent := typed.X.(*ast.Ident); isIdent && qualifier.Obj == nil && file.imports[qualifier.Name] != "" {
					key = file.imports[qualifier.Name] + " " + typed.Sel.Name
				}
			}
			if _, constant := constants[key]; constant {
				named = append(named, key)
			}
		}
	}
	slices.Sort(named)
	return slices.Compact(named)
}

// switchFamilies joins the switches that share a constant into families, largest first.
func switchFamilies(switches []kindSwitch, constants map[string]string) []switchFamily {
	sets := newUnionFind(len(switches))
	first := map[string]int{}
	for index, kind := range switches {
		for _, constant := range kind.constants {
			if earlier, seen := first[constant]; seen {
				sets.join(index, earlier)
			} else {
				first[constant] = index
			}
		}
	}
	byRoot := map[int]*switchFamily{}
	for index, kind := range switches {
		family := byRoot[sets.root(index)]
		if family == nil {
			family = &switchFamily{}
			byRoot[sets.root(index)] = family
		}
		family.switches = append(family.switches, kind)
		family.constants = append(family.constants, kind.constants...)
		family.lines += kind.place.lines()
	}
	var families []switchFamily
	for _, family := range byRoot {
		slices.Sort(family.constants)
		family.constants = slices.Compact(family.constants)
		slices.SortFunc(family.switches, func(a, b kindSwitch) int { return comparePlaces(a.place, b.place) })
		counts := map[string]int{}
		for _, constant := range family.constants {
			counts[constants[constant]]++
		}
		for label, count := range counts {
			if count > counts[family.label] || count == counts[family.label] && label < family.label {
				family.label = label
			}
		}
		families = append(families, *family)
	}
	slices.SortFunc(families, func(a, b switchFamily) int {
		return cmp.Or(cmp.Compare(b.lines, a.lines), strings.Compare(a.label, b.label))
	})
	return families
}

// shapeFiles returns the production files of a snapshot that some build selects, outside generated files, with the names each imports packages as.
//   - a generated file's repeats are its generator's, so a person consolidates none of them
func shapeFiles(snapshot gosyntax.Snapshot) ([]*reachFile, error) {
	var files []*reachFile
	declared := map[string]string{}
	for _, source := range snapshot.Files {
		if source.Test {
			continue
		}
		syntax, err := source.Syntax()
		if err != nil {
			return nil, err
		}
		if neverBuilt(syntax) {
			continue
		}
		declared[path.Dir(source.Path)] = syntax.Name.Name
		if generated, _ := source.Generated(); !generated {
			files = append(files, &reachFile{source: source, syntax: syntax, directory: path.Dir(source.Path), imports: map[string]string{}})
		}
	}
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
			file.imports[name] = directory
		}
	}
	return files, nil
}

// shapeReading is what the shape reading lists of one tree.
//   - files = the production files it read
//   - runLines = the lines the repeated runs cover, a line two groups share counted once
type shapeReading struct {
	files    int
	runs     []repeatGroup
	runLines int
	chains   []callChain
	families []switchFamily
}

// readShapes reads the production files of a snapshot and lists three readings of its repeated structure.
//   - read = the files shapeFiles returns
//   - statement shape = its syntax with every identifier and every literal's value left out
//   - repeated run = runLength consecutive statements of one list whose shape repeats in runFunctions or more functions, grouped by repeatedRuns
//   - chain = chainPairs or more consecutive call-and-test pairs whose calls share one receiver
//   - switch by kind = a switch whose cases name kindConstants or more constants a package of the tree declares
//   - family = the switches that share a constant, transitively
//   - a switch's or a select's body is a list of clauses and never a run, and each clause's body is a list
func readShapes(snapshot gosyntax.Snapshot) (shapeReading, error) {
	files, err := shapeFiles(snapshot)
	if err != nil {
		return shapeReading{}, err
	}
	constants := namedConstants(files)
	shapes := map[string]int{}
	var lists []*statementList
	var switches []kindSwitch
	for _, file := range files {
		for _, declaration := range file.syntax.Decls {
			function, isFunction := declaration.(*ast.FuncDecl)
			if !isFunction || function.Body == nil {
				continue
			}
			name := function.Name.Name
			if function.Recv != nil {
				name = receiverType(function) + "." + name
			}
			read := func(statements []ast.Stmt) {
				lists = append(lists, newStatementList(file, name, statements, shapes))
			}
			ast.Inspect(function.Body, func(node ast.Node) bool {
				switch typed := node.(type) {
				case *ast.BlockStmt:
					if len(typed.List) > 0 {
						switch typed.List[0].(type) {
						case *ast.CaseClause, *ast.CommClause:
						default:
							read(typed.List)
						}
					}
				case *ast.CaseClause:
					read(typed.Body)
				case *ast.CommClause:
					read(typed.Body)
				case *ast.SwitchStmt:
					if named := switchConstants(file, typed, constants); len(named) >= kindConstants {
						place := shapePlace{file.source.Path, name, file.source.Line(typed.Pos()), file.source.Line(typed.End())}
						switches = append(switches, kindSwitch{place, named})
					}
				}
				return true
			})
		}
	}
	runs, covered := repeatedRuns(lists)
	return shapeReading{len(files), runs, covered, callChains(lists), switchFamilies(switches, constants)}, nil
}

// listing writes a reading one finding a line, the runs, then the chains, then the families, each largest first.
func (r shapeReading) listing() []string {
	var written []string
	for _, group := range r.runs {
		written = append(written, group.String())
	}
	for _, chain := range r.chains {
		written = append(written, chain.String())
	}
	for _, family := range r.families {
		written = append(written, family.String())
	}
	return written
}

// shapeFixture plants one finding of each reading and each class, and a near miss of each.
//   - runs = flag definitions in two commands, call-and-test pairs in two packages, and other statements in two functions of one package
//   - the commands' windows that reach their error test are call-and-test, and the windows of flags alone are flag definitions
//   - near misses of runs = four statements repeated in two functions, and five repeated inside one function
//   - near miss of the class call-and-test = a call whose test does not return
//   - near miss of the class flag-definitions = a call of Set, a method of flag.FlagSet that defines no flag
//   - chain = three pairs on one receiver, and its near miss three pairs on two receivers
//   - families = three switches joined through the constants they share in pairs, and one switch over constants of another type
//   - near misses of switches = two named constants and a literal, and three literals
var shapeFixture = map[string]string{
	"cmd/alpha/main.go": `package main

import "flag"

func main() {
	fs := flag.NewFlagSet("alpha", flag.ExitOnError)
	model := fs.String("model", "", "model path")
	prompt := fs.String("prompt", "", "prompt text")
	seed := fs.Int("seed", 0, "seed")
	steps := fs.Int("steps", 0, "steps")
	scale := fs.Float64("scale", 0, "scale")
	err := fs.Parse(nil)
	if err != nil {
		return
	}
	println(*model, *prompt, *seed, *steps, *scale)
}
`,
	"cmd/beta/main.go": `package main

import "flag"

func main() {
	set := flag.NewFlagSet("beta", flag.ContinueOnError)
	input := set.String("input", "", "input path")
	output := set.String("output", "", "output path")
	width := set.Int("width", 0, "width")
	height := set.Int("height", 0, "height")
	ratio := set.Float64("ratio", 0, "ratio")
	err := set.Parse(nil)
	if err != nil {
		return
	}
	println(*input, *output, *width, *height, *ratio)
}
`,
	"internal/store/store.go": `package store

func load(path string) ([]byte, error) { return nil, nil }

func decode(data []byte) (map[string]string, error) { return nil, nil }

func keep(values map[string]string) {}

func Import(path string) error {
	data, err := load(path)
	if err != nil {
		return err
	}
	parsed, err := decode(data)
	if err != nil {
		return err
	}
	keep(parsed)
	return nil
}
`,
	"internal/archive/archive.go": `package archive

func fetch(name string) ([]byte, error) { return nil, nil }

func unpack(raw []byte) ([]string, error) { return nil, nil }

func hold(entries []string) {}

func Restore(name string) error {
	raw, err := fetch(name)
	if err != nil {
		return err
	}
	entries, err := unpack(raw)
	if err != nil {
		return err
	}
	hold(entries)
	return nil
}
`,
	"internal/geometry/geometry.go": `package geometry

var registry settings

func write(value float64) (int, error) { return 0, nil }

func report(err error) {}

func Area(left, right, factor float64) float64 {
	total := left + right
	scaled := total * factor
	_, err := write(scaled)
	if err != nil {
		report(err)
	}
	registry.Set("area", "done")
	return scaled
}

func Volume(base, height, depth float64) float64 {
	sum := base + height
	product := sum * depth
	_, err := write(product)
	if err != nil {
		report(err)
	}
	registry.Set("volume", "done")
	return product
}
`,
	"internal/stats/stats.go": `package stats

func Mean(values []float64) float64 {
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	n := float64(len(values))
	return sum / n
}

func Average(samples []float64) float64 {
	total := 0.0
	for _, sample := range samples {
		total += sample
	}
	size := float64(len(samples))
	return total / size
}

func Twice(a, b int) (int, int) {
	first := 0
	{
		x := a * b
		y := x - a
		z := y * b
		w := z - x
		first = w * y
	}
	second := 0
	{
		x := a * b
		y := x - a
		z := y * b
		w := z - x
		second = w * y
	}
	return first, second
}
`,
	"internal/device/device.go": `package device

type scope struct{}

func (s *scope) upload(values []float32) (int, error) { return 0, nil }

func (s *scope) launch(a, b, c int) error { return nil }

type runner struct{}

func (r *runner) Upload(device *scope, a, b, c []float32) error {
	pa, err := device.upload(a)
	if err != nil {
		return err
	}
	pb, err := device.upload(b)
	if err != nil {
		return err
	}
	pc, err := device.upload(c)
	if err != nil {
		return err
	}
	return device.launch(pa, pb, pc)
}

func (r *runner) Mixed(device, spare *scope, a, b, c []float32) error {
	pa, err := device.upload(a)
	if err != nil {
		return err
	}
	pb, err := device.upload(b[:len(a)])
	if err != nil {
		return err
	}
	pc, err := spare.upload(c)
	if err != nil {
		return err
	}
	return device.launch(pa, pb, pc)
}
`,
	"internal/kinds/kinds.go": `package kinds

type Kind int

const (
	KindA Kind = iota
	KindB
	KindC
	KindD
	KindE
	KindF
	KindG
)

func Name(kind Kind) string {
	switch kind {
	case KindA:
		return "a"
	case KindB, KindC:
		return "bc"
	}
	return ""
}

func Pair(kind Kind) bool {
	switch kind {
	case KindA, KindB:
		return true
	case 9:
		return false
	}
	return false
}
`,
	"internal/use/use.go": `package use

import "overgo/internal/kinds"

func Size(kind kinds.Kind) int {
	switch kind {
	case kinds.KindC:
		return 1
	case kinds.KindD, kinds.KindE:
		return 2
	}
	return 0
}
`,
	"internal/third/third.go": `package third

import k "overgo/internal/kinds"

func Late(kind k.Kind) bool {
	switch kind {
	case k.KindE, k.KindF, k.KindG:
		return true
	}
	return false
}

func Word(name string) int {
	switch name {
	case "e", "f", "g":
		return 1
	}
	return 0
}
`,
	"internal/color/color.go": `package color

type Color string

const (
	Red   Color = "red"
	Green Color = "green"
	Blue  Color = "blue"
)

func Warm(color Color) bool {
	switch color {
	case Red:
		return true
	case Green, Blue:
		return false
	}
	return false
}
`,
}

// TestShapeReadingFindsPlantedFamilies holds the shape reading to the findings shapeFixture plants, and logs the repository's reading.
//   - fixture listing = each planted run with its class, the chain and the two families, and no near miss
//   - the repository's listing asserts no figure, because it informs the choice of the next consolidation and pins nothing
//   - a repository read of no production file is refused, because a reading of nothing earns no pass
func TestShapeReadingFindsPlantedFamilies(t *testing.T) {
	t.Parallel()
	fixture := fixtureSnapshot(t, shapeFixture)
	reading, err := readShapes(fixture)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"run call-and-test lines=20 places=2 internal/archive/archive.go:10-19 Restore, internal/store/store.go:10-19 Import",
		"run call-and-test lines=18 places=2 cmd/alpha/main.go:8-16 main, cmd/beta/main.go:8-16 main",
		"run other lines=16 places=2 internal/geometry/geometry.go:10-17 Area, internal/geometry/geometry.go:21-28 Volume",
		"run flag-definitions lines=12 places=2 cmd/alpha/main.go:6-11 main, cmd/beta/main.go:6-11 main",
		"chain receiver=device pairs=3 lines=12 internal/device/device.go:12-23 runner.Upload",
		"family internal/kinds.Kind switches=3 lines=16 functions=3 packages=3 constants=7 internal/kinds/kinds.go:16-21 Name, internal/third/third.go:6-9 Late, internal/use/use.go:6-11 Size",
		"family internal/color.Color switches=1 lines=6 functions=1 packages=1 constants=3 internal/color/color.go:12-17 Warm",
	}
	if got := reading.listing(); !slices.Equal(got, want) {
		t.Errorf("the fixture's listing =\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}

	{
		planted := []consolidatedGroup{
			{name: "planted run", functions: []string{"internal/store/store.go Import", "internal/archive/archive.go Restore"}, calls: []string{"load", "fetch"}},
			{name: "planted chain", functions: []string{"internal/device/device.go runner.Upload"}, calls: []string{"upload"}},
			{name: "near miss", functions: []string{"internal/store/store.go Import", "internal/archive/archive.go Restore"}, calls: []string{"launch"}},
		}
		holdConsolidated(t, fixture, reading, planted, []string{
			`group "planted run" repeats in run call-and-test lines=20 places=2 internal/archive/archive.go:10-19 Restore, internal/store/store.go:10-19 Import`,
			`group "planted chain" repeats in chain receiver=device pairs=3 lines=12 internal/device/device.go:12-23 runner.Upload`,
		})
	}
	{
		fixture := fixtureSnapshot(t, shapeFixture, caseFixture)
		reading, err := readShapes(fixture)
		if err != nil {
			t.Fatal(err)
		}
		compile := []string{"internal/suites/suites.go Compile"}
		planted := []consolidatedGroup{
			{name: "planted cases", functions: compile, cases: []string{`"a"`, `"b"`}, calls: []string{"decode"}},
			{name: "near miss", functions: compile, cases: []string{`"a"`, `"c"`}, calls: []string{"decode"}},
			{name: "absent case", functions: compile, cases: []string{`"d"`}, calls: []string{"decode"}},
			{name: "planted files", functions: []string{"internal/store/store.go", "internal/archive/archive.go"}, calls: []string{"load", "fetch"}},
		}
		holdConsolidated(t, fixture, reading, planted, []string{
			`group "planted cases" repeats in run call-and-test lines=18 places=2 internal/suites/suites.go:12-20 Compile case "a", internal/suites/suites.go:22-30 Compile case "b"`,
			`group "absent case" names case "d", which no clause of its functions holds`,
			`group "planted files" repeats in run call-and-test lines=20 places=2 internal/archive/archive.go:10-19 Restore, internal/store/store.go:10-19 Import`,
		})
	}
	snapshot, err := gosyntax.DiscoverGo(filepath.Join("..", ".."), "cmd", "internal")
	if err != nil {
		t.Fatal(err)
	}
	reading, err = readShapes(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if reading.files == 0 {
		t.Fatal("the shape reading read no production file of the repository, want the files of cmd and internal")
	}
	t.Logf("files=%d", reading.files)
	t.Logf("repeated runs groups=%d covering lines=%d", len(reading.runs), reading.runLines)
	for _, class := range runClasses {
		var groups []string
		var places []shapePlace
		for _, group := range reading.runs {
			if group.class == class {
				groups = append(groups, group.String())
				places = append(places, group.places...)
			}
		}
		t.Logf("class %s groups=%d covering lines=%d", class, len(groups), coveredLines(places))
		for _, line := range groups[:min(largestShown, len(groups))] {
			t.Log(line)
		}
	}
	chainLines, switches, switchLines := 0, 0, 0
	for _, chain := range reading.chains {
		chainLines += chain.place.lines()
	}
	for _, family := range reading.families {
		switches, switchLines = switches+len(family.switches), switchLines+family.lines
	}
	t.Logf("call-and-test chains=%d lines=%d", len(reading.chains), chainLines)
	for _, chain := range reading.chains[:min(largestShown, len(reading.chains))] {
		t.Log(chain)
	}
	t.Logf("switches over named constants=%d lines=%d families=%d", switches, switchLines, len(reading.families))
	for _, family := range reading.families[:min(largestShown, len(reading.families))] {
		t.Log(family)
	}
	repeats, err := consolidatedRepeats(snapshot, reading, slices.Concat(consolidatedGroups, consolidatedSurfaceGroups))
	if err != nil {
		t.Fatal(err)
	}
	if len(repeats) > 0 {
		t.Errorf("a consolidated group repeats again, want its places served by the common component:\n%s", strings.Join(repeats, "\n"))
	}
	checkKindsReadingRecord(t, snapshot, reading)

	found := handWrittenClones(t, snapshot)
	for file, reason := range keptClones {
		if !found[file] {
			t.Errorf("%s keeps no hand-written clone, so its entry (%s) is stale", file, reason)
		}
	}
	for file := range found {
		if keptClones[file] == "" {
			t.Errorf("%s holds a hand-written clone that keptClones does not name; DeepCopy is the codec's copy", file)
		}
	}
}
