package landing

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/testevidence"
)

// ceilingFall is one line a ceiling test said (testevidence.CeilingFell): the constant, its file, the ceiling and the
// count it fell to.
type ceilingFall struct {
	name, file     string
	ceiling, count int
}

// lowerCeilings writes each ceiling a count fell under, as the invariant tests said it, into the checkout, and takes
// it into the commit and the candidate the next stage reads: a ceiling only falls, and a refusal whose fix the tool
// knows is not a refusal (refusal-advice/ceilings-fall). Every fall of a file is applied to one reading of it, and every
// file is checked before any is written, so a refusal leaves the checkout as it was: a file the checkout holds a change
// of its own to, which the commit leaves out, is not written over (SQA-399).
func (l *landing) lowerCeilings(ctx context.Context, output string) error {
	opening, _, _ := strings.Cut(testevidence.CeilingFell, "%")
	falls := map[string][]ceilingFall{}
	for line := range strings.Lines(output) {
		var event struct{ Output string }
		_ = json.Unmarshal([]byte(line), &event)
		_, said, found := strings.Cut(event.Output, opening)
		if !found {
			continue
		}
		var fall ceilingFall
		if _, err := fmt.Sscanf(opening+said, testevidence.CeilingFell, &fall.name, &fall.file, &fall.ceiling, &fall.count); err != nil {
			return err
		}
		if !slices.Contains(falls[fall.file], fall) {
			falls[fall.file] = append(falls[fall.file], fall)
		}
	}
	lowered := map[string][]byte{}
	for _, file := range slices.Sorted(maps.Keys(falls)) {
		read, err := os.ReadFile(filepath.Join(l.stageRoot(), filepath.FromSlash(file)))
		if err != nil {
			return err
		}
		if held, _ := os.ReadFile(filepath.Join(l.Repo, filepath.FromSlash(file))); !bytes.Equal(held, read) {
			said := make([]string, 0, len(falls[file]))
			for _, fall := range falls[file] {
				said = append(said, fmt.Sprintf("%s fell to %d under its ceiling of %d", fall.name, fall.count, fall.ceiling))
			}
			return fmt.Errorf("%s, and %s holds a change this commit leaves out; edit: lower them there", strings.Join(said, "; "), file)
		}
		for _, fall := range falls[file] {
			if read, err = lowerCeiling(read, fall); err != nil {
				return fmt.Errorf("%s: %w; edit: lower %s to %d there", file, err, fall.name, fall.count)
			}
		}
		lowered[file] = read
	}
	if len(lowered) == 0 {
		return nil
	}
	for _, file := range slices.Sorted(maps.Keys(lowered)) {
		if err := os.WriteFile(filepath.Join(l.Repo, filepath.FromSlash(file)), lowered[file], clioptions.OutputFileMode); err != nil {
			return err
		}
		if l.scope != nil {
			l.scope = append(l.scope, file)
		}
		for _, fall := range falls[file] {
			fmt.Fprintf(l.Log, "land: %s fell to %d under its ceiling of %d, so %s holds %d in this commit\n", fall.name, fall.count, fall.ceiling, file, fall.count)
		}
	}
	return l.shipSet(ctx)
}

// lowerCeiling rewrites the one declaration of the constant a fall names, which must hold the ceiling it fell under,
// to the count, wherever it stands: alone or in a grouped block, its value aligned or not.
func lowerCeiling(source []byte, fall ceilingFall) ([]byte, error) {
	files := token.NewFileSet()
	parsed, err := parser.ParseFile(files, "", source, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return nil, err
	}
	var held []*ast.BasicLit
	ast.Inspect(parsed, func(node ast.Node) bool {
		if spec, isSpec := node.(*ast.ValueSpec); isSpec && len(spec.Values) > 0 {
			for index, ident := range spec.Names {
				if literal, isLiteral := spec.Values[min(index, len(spec.Values)-1)].(*ast.BasicLit); ident.Name == fall.name && isLiteral && literal.Value == strconv.Itoa(fall.ceiling) {
					held = append(held, literal)
				}
			}
		}
		return true
	})
	if len(held) != 1 {
		return nil, fmt.Errorf("%d declarations of %s = %d to lower to %d, want one", len(held), fall.name, fall.ceiling, fall.count)
	}
	start, end := files.Position(held[0].Pos()).Offset, files.Position(held[0].End()).Offset
	return format.Source(slices.Concat(source[:start], []byte(strconv.Itoa(fall.count)), source[end:]))
}
