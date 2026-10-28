package repoanalysis

import (
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gosyntax"
	"overgo/internal/plan"
)

// tuningPackages are the packages whose named values trade the serving's measured costs.
var tuningPackages = []string{"internal/inference", "internal/modeldevice", "internal/cuda/executor", "internal/hostexpert"}

// tuningEvidence marks a value's comment as a tuning value's: it cites a measurement or the reference engine, as a
// value chosen against measured costs does and a format's size or alignment does not.
var tuningEvidence = regexp.MustCompile(`Strata|measured|sweep|percent|per second|\d ?(ms|us|tok/s|%|GB|GiB|MiB|MB|KiB)\b`)

// tuningProvisional is a comment naming the plan step that derives its value.
var tuningProvisional = regexp.MustCompile(`Provisional: the ([a-z0-9-]+) step`)

// tuningValue is a row of the tuning table (tuningTable): a named value of a tuning package whose comment cites a
// measurement or the reference engine.
//   - a value that trades costs names the cost (trades), what set it (basis: measured here and on what, Strata's and
//     its source, or computed at load and by what) and, where its value waits to be derived, the open step that
//     derives it (step)
//   - a value that trades nothing names what fixes it (fixed): a unit, a format, a kernel's indexing
type tuningValue struct {
	file, name          string
	trades, basis, step string
	fixed               string
}

// tuningConstant is a named numeric constant of a tuning package as the table reads it: whether its comments cite a
// measurement or the reference engine, and the provisional step its comments name, all of them and its own.
type tuningConstant struct {
	cited             bool
	step, ownStep     string
	file, name, where string
}

// tuningConstants reads the snapshot's named numeric constants outside tests, by file and name.
func tuningConstants(snapshot gosyntax.Snapshot) (map[string]tuningConstant, error) {
	found := map[string]tuningConstant{}
	for _, file := range snapshot.Files {
		if strings.HasSuffix(file.Path, "_test.go") {
			continue
		}
		parsed, err := file.Syntax()
		if err != nil {
			return nil, fmt.Errorf("tuning values: parse %s: %w", file.Path, err)
		}
		for _, declaration := range parsed.Decls {
			group, ok := declaration.(*ast.GenDecl)
			if !ok || group.Tok != token.CONST {
				continue
			}
			for _, spec := range group.Specs {
				value := spec.(*ast.ValueSpec)
				if !numeric(value) {
					continue
				}
				own := commentText(value.Doc) + commentText(value.Comment)
				all := commentText(group.Doc) + own
				// A declaration of one value without parentheses carries the value's own comment.
				if !group.Lparen.IsValid() {
					own = all
				}
				for _, name := range value.Names {
					found[file.Path+"#"+name.Name] = tuningConstant{
						cited: tuningEvidence.MatchString(all), step: provisionalStep(all), ownStep: provisionalStep(own),
						file: file.Path, name: name.Name,
					}
				}
			}
		}
	}
	return found, nil
}

func provisionalStep(comment string) string {
	if step := tuningProvisional.FindStringSubmatch(comment); step != nil {
		return step[1]
	}
	return ""
}

// tuningTableFindings lists where the snapshot's values and the table part under the plan's open steps:
//   - a value whose comments cite a measurement or the reference engine has a row
//   - a row names a numeric value its file holds, and once
//   - a trading row names its cost and basis and no fixing, a fixed row its fixing alone
//   - a row's step is one the plan holds open, and a trading row's comment names no other provisional step; a fixed
//     row's own comment names none
func tuningTableFindings(snapshot gosyntax.Snapshot, table []tuningValue, open map[string]bool) ([]string, error) {
	constants, err := tuningConstants(snapshot)
	if err != nil {
		return nil, err
	}
	var findings []string
	rows := map[string]tuningValue{}
	for _, row := range table {
		key := row.file + "#" + row.name
		constant, held := constants[key]
		switch {
		case rows[key] != (tuningValue{}):
			findings = append(findings, key+": the table holds it twice")
		case !held:
			findings = append(findings, key+": the tree holds no such numeric value")
		case row.fixed == "" && (row.trades == "" || row.basis == ""):
			findings = append(findings, key+": a trading row names its cost and its basis")
		case row.fixed != "" && (row.trades != "" || row.basis != "" || row.step != ""):
			findings = append(findings, key+": a fixed row names what fixes it and nothing it trades")
		case row.step != "" && !open[row.step]:
			findings = append(findings, fmt.Sprintf("%s: its step %s is not one the plan holds open", key, row.step))
		case row.fixed == "" && constant.step != "" && constant.step != row.step:
			findings = append(findings, fmt.Sprintf("%s: its comment names the %s step and its row %q", key, constant.step, row.step))
		case row.fixed != "" && constant.ownStep != "":
			findings = append(findings, fmt.Sprintf("%s: its comment names the %s step, and it trades nothing", key, constant.ownStep))
		}
		rows[key] = row
	}
	for key, constant := range constants {
		if constant.cited && rows[key] == (tuningValue{}) {
			findings = append(findings, key+": its comment cites a measurement or Strata and the table holds no row of it")
		}
	}
	slices.Sort(findings)
	return findings, nil
}

// numeric reports a value spec whose values are numeric literals or expressions of them.
func numeric(value *ast.ValueSpec) bool {
	if len(value.Values) == 0 {
		return false
	}
	for _, expression := range value.Values {
		literal := false
		ast.Inspect(expression, func(node ast.Node) bool {
			if basic, ok := node.(*ast.BasicLit); ok && (basic.Kind == token.INT || basic.Kind == token.FLOAT) {
				literal = true
			}
			return true
		})
		if !literal {
			return false
		}
	}
	return true
}

func commentText(group *ast.CommentGroup) string {
	if group == nil {
		return ""
	}
	return group.Text() + "\n"
}

// TestTuningValuesNameTheirCost holds the tuning packages' named values to the tuning table (tuningTable): every value
// whose comment cites a measurement or the reference engine has a row naming the cost it trades and its basis, or what
// fixes it, a row's step is open in the plan, and a comment's provisional step is its row's; a fixture refuses a value
// with no row, a row the tree lacks, a row naming no basis, a row naming a closed step and a comment parting from its
// row.
func TestTuningValuesNameTheirCost(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "values.go"), []byte(`package probe

// copied is Strata's 64.
const copied = 64

// derived trades a copy's 160 us against a miss's 50 us.
const derived = 3

// pending is a sweep's 8,192. Provisional: the chunk step derives it.
const pending = 8192

// parted is a sweep's 4,096. Provisional: the chunk step derives it.
const parted = 4096

// unit is a GB.
const unit = 1e9

// unrowed is a sweep's 2,048. Provisional: the chunk step derives it.
const unrowed = 2048

// widthGB is a GB. Provisional: the chunk step derives it.
const widthGB = 1e9

// format is a block's 32 values.
const format = 32
`), 0o644); err != nil {
		t.Fatal(err)
	}
	fixture, err := gosyntax.DiscoverGo(root, ".")
	if err != nil {
		t.Fatal(err)
	}
	table := []tuningValue{
		{file: "values.go", name: "derived", trades: "a copy against a miss"},
		{file: "values.go", name: "pending", trades: "a chunk's reads", basis: "measured here", step: "chunk"},
		{file: "values.go", name: "parted", trades: "a chunk's reads", basis: "measured here", step: "gone"},
		{file: "values.go", name: "unit", fixed: "a unit"},
		{file: "values.go", name: "absent", fixed: "a unit"},
		{file: "values.go", name: "unrowed", trades: "a chunk's reads", basis: "measured here"},
		{file: "values.go", name: "widthGB", fixed: "a unit"},
	}
	found, err := tuningTableFindings(fixture, table, map[string]bool{"chunk": true})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"values.go#absent: the tree holds no such numeric value",
		"values.go#copied: its comment cites a measurement or Strata and the table holds no row of it",
		"values.go#derived: a trading row names its cost and its basis",
		"values.go#parted: its step gone is not one the plan holds open",
		`values.go#unrowed: its comment names the chunk step and its row ""`,
		"values.go#widthGB: its comment names the chunk step, and it trades nothing",
	}
	if !slices.Equal(found, want) {
		t.Fatalf("the fixture's findings = %q, want %q", found, want)
	}

	repository, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	document, err := plan.Load(filepath.Join(repository, filepath.FromSlash(plan.Path)))
	if err != nil {
		t.Fatal(err)
	}
	open := map[string]bool{}
	for _, item := range document.Items {
		for _, step := range item.Steps {
			if step.Status == plan.StatusOpen {
				open[step.ID] = true
			}
		}
	}
	live, err := gosyntax.DiscoverGo(repository, tuningPackages...)
	if err != nil {
		t.Fatal(err)
	}
	if found, err = tuningTableFindings(live, tuningTable, open); err != nil {
		t.Fatal(err)
	}
	for _, finding := range found {
		t.Error(finding)
	}
}
