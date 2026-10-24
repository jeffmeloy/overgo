package repoanalysis

import (
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"path"
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

// tuningDerivation is a tuning value's comment naming the cost it trades, or the plan step that derives it.
var (
	tuningCost        = regexp.MustCompile(`minimiz|trades|outweigh|pays where|costs? (more|less) than`)
	tuningProvisional = regexp.MustCompile(`Provisional: the ([a-z0-9-]+) step`)
)

// UnderivedTuningValue is a named value of a tuning package whose comment cites a measurement or the reference
// engine but names neither the cost it trades nor an open plan step that derives it.
type UnderivedTuningValue struct {
	File  string `json:"file"`
	Name  string `json:"name"`
	Cause string `json:"cause"`
}

// UnderivedTuningValues lists the named numeric constants of the snapshot's files whose comments cite a measurement
// or the reference engine and name neither the cost they trade nor a step open names (doctrine: a tuning value names
// the cost it minimizes, and a reference's or a sweep's value is provisional with its derivation step filed).
func UnderivedTuningValues(snapshot gosyntax.Snapshot, open map[string]bool) ([]UnderivedTuningValue, error) {
	var found []UnderivedTuningValue
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
				comment := commentText(group.Doc) + commentText(value.Doc) + commentText(value.Comment)
				if !tuningEvidence.MatchString(comment) || tuningCost.MatchString(comment) {
					continue
				}
				cause := "its comment cites a measurement or Strata and names neither the cost it trades nor its derivation step"
				if step := tuningProvisional.FindStringSubmatch(comment); step != nil {
					if open[step[1]] {
						continue
					}
					cause = fmt.Sprintf("its comment names the %s step, which the plan holds no open step of", step[1])
				}
				for _, name := range value.Names {
					found = append(found, UnderivedTuningValue{File: file.Path, Name: name.Name, Cause: cause})
				}
			}
		}
	}
	slices.SortFunc(found, func(left, right UnderivedTuningValue) int {
		return strings.Compare(left.File+"\x00"+left.Name, right.File+"\x00"+right.Name)
	})
	return found, nil
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

// TestTuningValuesNameTheirCost holds the tuning packages' named values to the doctrine: a value whose comment cites a
// measurement or the reference engine names the cost it trades, or says it is provisional and names the open plan
// step that derives it; a fixture's value with neither, and one naming a step the plan does not hold, are refused.
func TestTuningValuesNameTheirCost(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "values.go"), []byte(`package probe

// copied is Strata's 64.
const copied = 64

// derived trades a copy's 160 us against a miss's 50 us, which it minimizes.
const derived = 3

// pending is a sweep's 8,192. Provisional: the chunk step derives it.
const pending = 8192

// gone is a sweep's 4,096. Provisional: the retired step derives it.
const gone = 4096

// format is a block's 32 values.
const format = 32
`), 0o644); err != nil {
		t.Fatal(err)
	}
	fixture, err := gosyntax.DiscoverGo(root, ".")
	if err != nil {
		t.Fatal(err)
	}
	found, err := UnderivedTuningValues(fixture, map[string]bool{"chunk": true})
	if err != nil {
		t.Fatal(err)
	}
	if names := underivedNames(found); !slices.Equal(names, []string{"copied", "gone"}) {
		t.Fatalf("the fixture's underived values = %v, want copied and gone", found)
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
	if found, err = UnderivedTuningValues(live, open); err != nil {
		t.Fatal(err)
	}
	for _, value := range found {
		t.Errorf("%s %s: %s", path.Clean(value.File), value.Name, value.Cause)
	}
}

func underivedNames(values []UnderivedTuningValue) []string {
	var out []string
	for _, value := range values {
		out = append(out, value.Name)
	}
	return out
}
