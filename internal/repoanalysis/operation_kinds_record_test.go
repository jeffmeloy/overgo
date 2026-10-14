package repoanalysis

import (
	"fmt"
	"go/ast"
	"go/token"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
)

// kindsRecord is the record of the reading of the switches over the tensor operations.
const kindsRecord = "docs/verification/operation-kinds.txt"

// The parts of kindsRecord the test reads, and the text before them is for a person.
const (
	kindsSwitchesPart = "== switches =="
	kindsEstimatePart = "== estimate =="
)

// kindsFamily is the label of the family of switches over the tensor operations, which the record reads.
const kindsFamily = "internal/tensor.Op"

// kindsReadKeys are the keys of a switch entry that hold a reading and not a count, in the order an entry writes them.
//   - decides = what the switch decides
//   - row = what of it one row of a table of operations could hold
//   - table = yes when the table replaces the switch, no when it stays
var kindsReadKeys = []string{"decides", "row", "table"}

// kindsEstimateKeys are the keys of the record's estimate, each stated once.
//   - the table's figures = the dispatch lines of the switches it replaces, the estimated lines it deletes beside them and adds, and the net with its range
//   - makes-possible and not-made-possible = what the table makes possible that is not possible without it, and what it leaves as it is
//   - dense-block-program = how the table stands to the dense block program docs/verification/dense-block-program.txt reads
//   - table-row = whether a row is worth filing for the table, with the reason
var kindsEstimateKeys = []string{
	"table-dispatch-lines", "estimated-scaffolding-deleted", "table-deleted",
	"estimated-rows-added", "estimated-frames-added", "estimated-machinery-added", "table-added",
	"table-net", "table-net-range",
	"makes-possible", "not-made-possible", "dense-block-program", "table-row",
}

// The classes of a check that narrows a node's shape to the 32-bit words a kernel takes.
//   - dimensionCheck = one dimension or every dimension narrowed alone
//   - countCheck = a count of elements narrowed, which a rule on dimensions alone does not bound
const (
	dimensionCheck = "dimension"
	countCheck     = "count"
)

// narrowingCalls maps each function of the CUDA executor that narrows a value to 32-bit words to the class of its check.
//   - a call counts only when its first argument is a shape or a dimension of one, so uint32Checked on an attribute or a product is no check of a shape
//   - rowDimensions32 is a count check because it narrows a row width and divides the count of elements by it
var narrowingCalls = map[string]string{
	"uint32Checked":     dimensionCheck,
	"shapeDimensions32": dimensionCheck,
	"elementCount32":    countCheck,
	"rowDimensions32":   countCheck,
}

// switchCounts is what the syntax of one switch of the family holds.
//   - clauses = its case clauses, its default among them
//   - dispatch = the lines a table deletes from it outright, which are its opening line, each clause's header, a default clause of one statement whole, and its closing brace
//   - recheck = the lines of each if that refuses a failed type assertion with one return, the assertion on the statement before it
//   - dimension and count = the lines of each if that returns the error of a narrowing call on the statement before it, by the call's class
type switchCounts struct {
	clauses, dispatch, recheck, dimension, count int
}

// countSwitch finds the switch that opens on a place's first line and counts what its syntax holds.
func countSwitch(snapshot gosyntax.Snapshot, place shapePlace) (switchCounts, error) {
	var counted switchCounts
	index := slices.IndexFunc(snapshot.Files, func(file gosyntax.File) bool { return file.Path == place.file })
	if index < 0 {
		return counted, fmt.Errorf("the snapshot holds no file %s", place.file)
	}
	source := snapshot.Files[index]
	syntax, err := source.Syntax()
	if err != nil {
		return counted, err
	}
	var found *ast.SwitchStmt
	ast.Inspect(syntax, func(node ast.Node) bool {
		if statement, isSwitch := node.(*ast.SwitchStmt); isSwitch && found == nil && source.Line(statement.Pos()) == place.from {
			found = statement
		}
		return found == nil
	})
	if found == nil {
		return counted, fmt.Errorf("%s opens no switch on its first line", place)
	}
	span := func(node ast.Node) int { return source.Line(node.End()) - source.Line(node.Pos()) + 1 }
	// The lines from the switch keyword to its opening brace, and the one closing brace.
	counted.dispatch = source.Line(found.Body.Lbrace) - source.Line(found.Pos()) + 1 + 1
	for _, statement := range found.Body.List {
		clause := statement.(*ast.CaseClause)
		counted.clauses++
		if clause.List == nil && len(clause.Body) == 1 {
			counted.dispatch += span(clause)
		} else {
			counted.dispatch += source.Line(clause.Colon) - source.Line(clause.Pos()) + 1
		}
	}
	ast.Inspect(found, func(node ast.Node) bool {
		var list []ast.Stmt
		switch typed := node.(type) {
		case *ast.BlockStmt:
			list = typed.List
		case *ast.CaseClause:
			list = typed.Body
		}
		for at := 1; at < len(list); at++ {
			tested, value := guardedValue(list[at-1])
			test, returns := returningIf(list[at])
			if tested == "" || !returns {
				continue
			}
			switch typed := value.(type) {
			case *ast.TypeAssertExpr:
				if negation, isUnary := test.Cond.(*ast.UnaryExpr); isUnary && negation.Op == token.NOT && isName(negation.X, tested) {
					counted.recheck += span(test)
				}
			case *ast.CallExpr:
				function, isIdent := typed.Fun.(*ast.Ident)
				condition, isBinary := test.Cond.(*ast.BinaryExpr)
				if !isIdent || !isBinary || condition.Op != token.NEQ || !isName(condition.X, tested) || !isName(condition.Y, "nil") ||
					len(typed.Args) == 0 || !shapeOperand(typed.Args[0]) {
					continue
				}
				switch narrowingCalls[function.Name] {
				case dimensionCheck:
					counted.dimension += span(test)
				case countCheck:
					counted.count += span(test)
				}
			}
		}
		return true
	})
	return counted, nil
}

// guardedValue returns the name an assignment gives its last value, and the one expression it assigns from.
//   - an assignment of any other form returns ""
func guardedValue(statement ast.Stmt) (string, ast.Expr) {
	assignment, isAssignment := statement.(*ast.AssignStmt)
	if !isAssignment || len(assignment.Rhs) != 1 {
		return "", nil
	}
	last, isIdent := assignment.Lhs[len(assignment.Lhs)-1].(*ast.Ident)
	if !isIdent {
		return "", nil
	}
	return last.Name, assignment.Rhs[0]
}

// returningIf returns an if with no init and no else whose body is one return.
func returningIf(statement ast.Stmt) (*ast.IfStmt, bool) {
	test, isIf := statement.(*ast.IfStmt)
	if !isIf || test.Init != nil || test.Else != nil || len(test.Body.List) != 1 {
		return nil, false
	}
	_, returns := test.Body.List[0].(*ast.ReturnStmt)
	return test, returns
}

// isName reports whether an expression is the identifier name.
func isName(expression ast.Expr, name string) bool {
	ident, isIdent := expression.(*ast.Ident)
	return isIdent && ident.Name == name
}

// shapeOperand reports whether an expression is a shape, written X.Shape, or a dimension of one.
func shapeOperand(expression ast.Expr) bool {
	shape, isSelector := expression.(*ast.SelectorExpr)
	return isSelector && shape.Sel.Name == "Shape" || shapeDimension(expression)
}

// shapeDimension reports whether an expression indexes the dimensions of a shape, written X.Shape.Dims[i].
func shapeDimension(expression ast.Expr) bool {
	index, isIndex := expression.(*ast.IndexExpr)
	if !isIndex {
		return false
	}
	dimensions, isSelector := index.X.(*ast.SelectorExpr)
	if !isSelector || dimensions.Sel.Name != "Dims" {
		return false
	}
	shape, isShape := dimensions.X.(*ast.SelectorExpr)
	return isShape && shape.Sel.Name == "Shape"
}

// kindsPart returns the lines of a part of the record, from its opening line to the next part or the end, with no trailing empty line.
func kindsPart(text, opening string) ([]string, bool) {
	var lines []string
	inside, held := false, false
	for line := range strings.Lines(strings.ReplaceAll(text, "\r\n", "\n")) {
		line = strings.TrimSuffix(line, "\n")
		if strings.HasPrefix(line, "== ") {
			inside = line == opening
			held = held || inside
			continue
		}
		if inside {
			lines = append(lines, line)
		}
	}
	for len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines, held
}

// kindsField splits a line into its key and its value, and reports false for a line of prose.
func kindsField(line string) (string, string, bool) {
	key, value, keyed := strings.Cut(line, ": ")
	if !keyed || key == "" || strings.Contains(key, " ") {
		return "", "", false
	}
	return key, value, true
}

// kindsFixture plants one switch with each count of switchCounts and a near miss of each.
//   - near miss of a recheck = a failed assertion refused together with a second condition
//   - near miss of a dimension check = uint32Checked on a value that is no dimension of a shape
//   - near miss of a count check = a narrowing whose error a second condition joins
//   - near miss of a dispatch line = a default of two statements, whose header alone counts
const kindsFixture = `package fixture

func launch(node *Node, attributes any) error {
	switch node.Op {
	case OpA, OpB:
		value, ok := attributes.(Scale)
		if !ok {
			return errBad
		}
		width, err := uint32Checked(node.Shape.Dims[0], "width")
		if err != nil {
			return err
		}
		count, err := elementCount32(node.Shape)
		if err != nil {
			return err
		}
		return run(value, width, count, 0)
	case OpC:
		value, ok := attributes.(Clamp)
		if !ok || value.Low > value.High {
			return errBad
		}
		offset, err := uint32Checked(value.Offset, "offset")
		if err != nil {
			return err
		}
		rows, err := elementCount32(node.Shape)
		if err != nil || rows == 0 {
			return errBad
		}
		return run(value, offset, rows, 0)
	default:
		report(node)
		return errBad
	}
}
`

// The counts kindsFixture plants, read from its text.
//   - fixtureSwitchLine = the line its switch opens on
//   - fixtureClauses = its two cases and its default
//   - fixtureDispatch = the switch's opening line, two case headers, the default's header and the closing brace
//   - fixtureCheckLines = the lines of one planted recheck, dimension check and count check each
const (
	fixtureSwitchLine = 4
	fixtureClauses    = 3
	fixtureDispatch   = 5
	fixtureCheckLines = 3
)

// kindsFamilyOfTree requires the counters to find what kindsFixture plants, then reads the tree and returns the family kindsFamily labels.
//   - the planted counts come first, so a zero the counters give the tree means none found and not a reading that matches nothing
//   - a tree with no such family, or a family of no switch, fails the test
func kindsFamilyOfTree(t *testing.T) (gosyntax.Snapshot, switchFamily) {
	t.Helper()
	const fixtureFile = "internal/fixture/launch.go"
	planted, err := gosyntax.Snapshot{}.Overlay(map[string][]byte{fixtureFile: []byte(kindsFixture)})
	if err != nil {
		t.Fatal(err)
	}
	counts, err := countSwitch(planted, shapePlace{file: fixtureFile, function: "launch", from: fixtureSwitchLine})
	plantedCounts := switchCounts{fixtureClauses, fixtureDispatch, fixtureCheckLines, fixtureCheckLines, fixtureCheckLines}
	if err != nil || counts != plantedCounts {
		t.Fatalf("the planted switch counts %+v (%v), want %+v", counts, err, plantedCounts)
	}
	snapshot, err := gosyntax.DiscoverGo(filepath.Join("..", ".."), "cmd", "internal")
	if err != nil {
		t.Fatal(err)
	}
	reading, err := readShapes(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	index := slices.IndexFunc(reading.families, func(family switchFamily) bool { return family.label == kindsFamily })
	if index < 0 || len(reading.families[index].switches) == 0 {
		t.Fatalf("the shape reading finds no switch of the family %s, want the switches over the tensor operations", kindsFamily)
	}
	return snapshot, reading.families[index]
}

// TestNoLaunchRepeatsACompileCheck holds each switch over the tensor operations free of a check that a compiled graph already holds.
//   - recheck = the refusal of a failed type assertion of an operation's attributes, which tensor.ValidateOperationAttributes holds when a graph is built, when it compiles and when one host operation runs
//   - dimension and count check = the refusal of a failed narrowing of a shape to 32-bit words, which internal/cuda/executor/fallback.go#checkLaunchWords holds when a graph compiles for the device
//   - the counters are the record's, first required to find what kindsFixture plants
func TestNoLaunchRepeatsACompileCheck(t *testing.T) {
	t.Parallel()
	snapshot, family := kindsFamilyOfTree(t)
	for _, kind := range family.switches {
		counts, err := countSwitch(snapshot, kind.place)
		if err != nil {
			t.Fatal(err)
		}
		if repeated := counts.recheck + counts.dimension + counts.count; repeated != 0 {
			t.Errorf("%s holds %d lines that check again what a compiled graph holds (%d recheck, %d dimension check, %d count check), want 0",
				kind.place, repeated, counts.recheck, counts.dimension, counts.count)
		}
	}
}

// TestKindsReadingRecordIsWhole holds the record of the reading of the switches over the tensor operations to the tree.
//   - family = the family kindsFamily labels, with its counts as the reading's listing gives them
//   - operations-by-switches = for each number of the family's switches, how many of its operations that many switches name
//   - switch entry = one switch of the family, cited as file#function, with its lines and its constants as the listing gives them and its clauses and dispatch lines as switchCounts counts them
//   - a switch's reading keys are its own, and the test requires each, refusing a table other than yes or no
//   - a part that differs from the tree is printed as the tree gives it, the readings of the switches it still holds carried, so the fix is to replace the part and read again each switch whose figures moved
//   - estimate = the sum the switches give, the estimates the reading states, and the arithmetic that joins them, the net inside its range
//   - the closing row = yes or no and the reason
func TestKindsReadingRecordIsWhole(t *testing.T) {
	t.Parallel()
	snapshot, family := kindsFamilyOfTree(t)
	raw, err := os.ReadFile(filepath.Join("..", "..", filepath.FromSlash(kindsRecord)))
	if err != nil {
		t.Fatal(err)
	}

	// The record's switch entries keep their readings, keyed by the place they cite.
	part, held := kindsPart(string(raw), kindsSwitchesPart)
	if !held {
		t.Fatalf("%s holds no part %q", kindsRecord, kindsSwitchesPart)
	}
	readings := map[string]map[string]string{}
	var cited string
	for _, line := range part {
		key, value, keyed := kindsField(line)
		switch {
		case !keyed:
		case key == "switch":
			cited = value
			readings[cited] = map[string]string{}
		case cited != "" && slices.Contains(kindsReadKeys, key):
			readings[cited][key] = value
		}
	}

	// The part as the tree gives it, each switch's reading carried from the record.
	functions, packages, named := map[string]bool{}, map[string]bool{}, map[string]int{}
	var entries [][]string
	var dispatchReplaced int
	tableAnswers := []string{"yes", "no"}
	for _, kind := range family.switches {
		functions[kind.place.file+" "+kind.place.function] = true
		packages[path.Dir(kind.place.file)] = true
		for _, constant := range kind.constants {
			named[constant]++
		}
		counts, err := countSwitch(snapshot, kind.place)
		if err != nil {
			t.Fatal(err)
		}
		place := kind.place.file + "#" + kind.place.function
		entry := []string{
			"switch: " + place,
			"lines: " + strconv.Itoa(kind.place.lines()),
			"constants: " + strconv.Itoa(len(kind.constants)),
			"clauses: " + strconv.Itoa(counts.clauses),
			"dispatch-lines: " + strconv.Itoa(counts.dispatch),
		}
		for _, key := range kindsReadKeys {
			entry = append(entry, key+": "+readings[place][key])
			if strings.TrimSpace(readings[place][key]) == "" {
				t.Errorf("%s gives the switch %s no %s, want the reading of that switch", kindsRecord, place, key)
			}
		}
		if answer := readings[place]["table"]; !slices.Contains(tableAnswers, answer) {
			t.Errorf("%s gives the switch %s the table %q, want one of %s", kindsRecord, place, answer, strings.Join(tableAnswers, ", "))
		}
		if readings[place]["table"] == "yes" {
			dispatchReplaced += counts.dispatch
		}
		entries = append(entries, entry)
	}
	byCount := map[int]int{}
	for _, count := range named {
		byCount[count]++
	}
	var distribution []string
	for _, count := range slices.Sorted(maps.Keys(byCount)) {
		distribution = append(distribution, strconv.Itoa(count)+"="+strconv.Itoa(byCount[count]))
	}
	want := []string{
		"family: " + family.label,
		"switches: " + strconv.Itoa(len(family.switches)),
		"lines: " + strconv.Itoa(family.lines),
		"functions: " + strconv.Itoa(len(functions)),
		"packages: " + strconv.Itoa(len(packages)),
		"constants: " + strconv.Itoa(len(family.constants)),
		"operations-by-switches: " + strings.Join(distribution, " "),
	}
	for _, entry := range entries {
		want = append(append(want, ""), entry...)
	}
	if !slices.Equal(part, want) {
		t.Errorf("%s's part %s differs from the tree, so replace the part with the lines below and read again each switch whose figures moved\n%s\n%s",
			kindsRecord, kindsSwitchesPart, kindsSwitchesPart, strings.Join(want, "\n"))
	}

	// The estimate's fields, each stated once.
	estimate, held := kindsPart(string(raw), kindsEstimatePart)
	if !held {
		t.Fatalf("%s holds no part %q", kindsRecord, kindsEstimatePart)
	}
	fields := map[string]string{}
	for _, line := range estimate {
		key, value, keyed := kindsField(line)
		if !keyed {
			continue
		}
		if _, twice := fields[key]; twice || !slices.Contains(kindsEstimateKeys, key) {
			t.Fatalf("%s states %s twice or out of its keys in its part %s, want each of %s once", kindsRecord, key, kindsEstimatePart, strings.Join(kindsEstimateKeys, ", "))
		}
		fields[key] = value
	}
	// number reads a field as a whole number, or a signed one where signed is set.
	number := func(key string, signed bool) int {
		t.Helper()
		value, err := strconv.Atoi(fields[key])
		if err != nil || !signed && value < 0 {
			t.Fatalf("%s states %s as %q (%v), want a whole number", kindsRecord, key, fields[key], err)
		}
		return value
	}
	// holds refuses a field the record states as other than the figure its sources give.
	holds := func(key string, figure int, signed bool, source string) {
		t.Helper()
		if given := number(key, signed); given != figure {
			t.Errorf("%s states %s as %d, and %s give %d", kindsRecord, key, given, source, figure)
		}
	}
	// within refuses a net outside the range the record states for it.
	within := func(key string, net int) {
		t.Helper()
		low, high, ranged := strings.Cut(fields[key], " ")
		lowest, lowErr := strconv.Atoi(low)
		highest, highErr := strconv.Atoi(high)
		if !ranged || lowErr != nil || highErr != nil || lowest > net || net > highest {
			t.Errorf("%s states %s as %q, want a low and a high that hold the net %d", kindsRecord, key, fields[key], net)
		}
	}
	holds("table-dispatch-lines", dispatchReplaced, false, "the dispatch lines of the switches the table replaces")
	deleted := number("table-dispatch-lines", false) + number("estimated-scaffolding-deleted", false)
	holds("table-deleted", deleted, false, "the dispatch lines and the estimated scaffolding")
	added := number("estimated-rows-added", false) + number("estimated-frames-added", true) + number("estimated-machinery-added", false)
	holds("table-added", added, false, "the estimated rows, frames and machinery")
	holds("table-net", deleted-added, true, "deleted less added")
	within("table-net-range", number("table-net", true))
	for _, key := range []string{"makes-possible", "not-made-possible", "dense-block-program"} {
		if strings.TrimSpace(fields[key]) == "" {
			t.Errorf("%s states no %s", kindsRecord, key)
		}
	}
	if answer, reason, reasoned := strings.Cut(fields["table-row"], ", because "); !reasoned || !slices.Contains(tableAnswers, answer) || strings.TrimSpace(reason) == "" {
		t.Errorf("%s closes with table-row %q, want yes or no and the reason", kindsRecord, fields["table-row"])
	}
}
