package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/planverify"
	"overgo/internal/processlock"
)

const (
	// mutantsPerStep is the most bad alternatives a landed step is put to: its first comparisons on the lines its
	// landing wrote, in the order its files and lines come.
	mutantsPerStep = 3
	// mutantRecords is the file in the process state each mutation run's records are kept in, one line of JSON each.
	mutantRecords = "mutants.jsonl"
)

// inverted is each comparison and the one a mutant puts in its place: the same operands, the opposite answer.
var inverted = map[token.Token]token.Token{
	token.EQL: token.NEQ, token.NEQ: token.EQL, token.LSS: token.GEQ, token.GEQ: token.LSS, token.GTR: token.LEQ, token.LEQ: token.GTR,
}

// lineSpan is the lines of a file a landing wrote, from its first to before its last.
type lineSpan struct{ from, to int }

// mutant is a bad alternative of a landed step: its change with one comparison of its production Go inverted.
type mutant struct {
	file     string
	line     int
	column   int
	from, to token.Token
}

// String names the mutant as a record prints it.
func (m mutant) String() string { return fmt.Sprintf("%s:%d %s to %s", m.file, m.line, m.from, m.to) }

// mutantRecord is what the gate made of one landed step's bad alternatives: how many its own check was put to, how
// many it refused, and those it passed.
type mutantRecord struct {
	Step     string   `json:"step"`
	Landed   string   `json:"landed"`
	Mutants  int      `json:"mutants"`
	Refused  int      `json:"refused"`
	Survived []string `json:"survived,omitempty"`
	// Skipped says why no mutant of the step was put to its check: none could be made, its check needs the card, or
	// its check fails at its landing as it stands.
	Skipped string `json:"skipped,omitzero"`
}

// mutantsOf are the bad alternatives of a landed step: the comparisons on the lines its landing added or changed in
// its production Go, read at the landed commit, each inverted, mutantsPerStep at most.
func mutantsOf(ctx context.Context, root string, step replayed) ([]mutant, error) {
	diff, err := gitauthority.Query(ctx, root, "diff", "--unified=0", "--no-color", step.parent, step.landed, "--", "*.go")
	if err != nil {
		return nil, err
	}
	changed := map[string][]lineSpan{}
	var order []string
	file := ""
	for line := range strings.Lines(string(diff)) {
		line = strings.TrimRight(line, "\r\n")
		switch {
		case strings.HasPrefix(line, "+++ "):
			file = strings.TrimPrefix(strings.TrimPrefix(line, "+++ "), "b/")
			if strings.HasSuffix(file, "_test.go") || strings.Contains("/"+file, "/testdata/") || file == "/dev/null" {
				file = ""
			} else if _, seen := changed[file]; !seen {
				order = append(order, file)
				changed[file] = nil
			}
		case strings.HasPrefix(line, "@@ ") && file != "":
			// The hunk's new side, +start,count, the count one where it is left out.
			_, added, _ := strings.Cut(line, " +")
			added, _, _ = strings.Cut(added, " ")
			start, count, counted := strings.Cut(added, ",")
			first, err := strconv.Atoi(start)
			if err != nil {
				return nil, fmt.Errorf("loop: a hunk of %s reads %q", file, line)
			}
			lines := 1
			if counted {
				if lines, err = strconv.Atoi(count); err != nil {
					return nil, fmt.Errorf("loop: a hunk of %s reads %q", file, line)
				}
			}
			if lines > 0 {
				changed[file] = append(changed[file], lineSpan{from: first, to: first + lines})
			}
		}
	}
	var mutants []mutant
	for _, path := range order {
		source, err := gitauthority.Query(ctx, root, "show", step.landed+":"+path)
		if err != nil {
			return nil, err
		}
		files := token.NewFileSet()
		parsed, err := parser.ParseFile(files, path, source, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			compared, is := node.(*ast.BinaryExpr)
			if !is || len(mutants) == mutantsPerStep {
				return len(mutants) < mutantsPerStep
			}
			opposite, comparison := inverted[compared.Op]
			at := files.Position(compared.OpPos)
			for _, span := range changed[path] {
				if comparison && at.Line >= span.from && at.Line < span.to {
					mutants = append(mutants, mutant{file: path, line: at.Line, column: at.Column, from: compared.Op, to: opposite})
					break
				}
			}
			return true
		})
	}
	return mutants, nil
}

// onCard says a check needs the graphics card, which a mutation run on the host does not take.
func onCard(verify string) bool {
	return strings.Contains(verify, "OVERGO_CUDA_TEST=1") || strings.Contains(verify, "-tags modeltest")
}

// mutate puts each bad alternative of step to its own check in checkout, at the commit that landed it: the check is to
// pass there as it stands, and to fail with each mutant in place.
func mutate(ctx context.Context, root, checkout string, step replayed) (mutantRecord, error) {
	record := mutantRecord{Step: step.reference, Landed: step.landed}
	if onCard(step.step.Verify) {
		record.Skipped = "its check needs the card"
		return record, nil
	}
	mutants, err := mutantsOf(ctx, root, step)
	if err != nil {
		return record, err
	}
	if len(mutants) == 0 {
		record.Skipped = "its landing wrote no comparison in production Go"
		return record, nil
	}
	if err := moveReplay(root, checkout, step.landed); err != nil {
		return record, err
	}
	if _, err := planverify.Execute(ctx, checkout, step.step.Verify, nil); err != nil {
		record.Skipped = "its check fails at its landing: " + firstLine(err)
		return record, nil
	}
	for _, bad := range mutants {
		path := filepath.Join(checkout, filepath.FromSlash(bad.file))
		source, err := os.ReadFile(path)
		if err != nil {
			return record, err
		}
		// The operator is found by its line and column, which a checkout's line endings leave as the commit holds them.
		offset := 0
		for range bad.line - 1 {
			next := bytes.IndexByte(source[offset:], '\n')
			if next < 0 {
				return record, fmt.Errorf("loop: %s holds fewer than %d lines", bad.file, bad.line)
			}
			offset += next + 1
		}
		offset += bad.column - 1
		from := bad.from.String()
		if offset+len(from) > len(source) || !bytes.Equal(source[offset:offset+len(from)], []byte(from)) {
			return record, fmt.Errorf("loop: %s holds no %s where its landing wrote one", bad.file, from)
		}
		changed := append(append(bytes.Clone(source[:offset]), bad.to.String()...), source[offset+len(from):]...)
		if err := os.WriteFile(path, changed, clioptions.PrivateFileMode); err != nil {
			return record, err
		}
		_, refused := planverify.Execute(ctx, checkout, step.step.Verify, nil)
		if err := os.WriteFile(path, source, clioptions.PrivateFileMode); err != nil {
			return record, err
		}
		if ctx.Err() != nil {
			return record, ctx.Err()
		}
		record.Mutants++
		if refused != nil {
			record.Refused++
		} else {
			record.Survived = append(record.Survived, bad.String())
		}
	}
	return record, nil
}

// firstLine is an error's first line.
func firstLine(err error) string {
	line, _, _ := strings.Cut(err.Error(), "\n")
	return line
}

// mutateSteps puts the bad alternatives of the steps landed since the date, newest first, up to limit of them, to
// their own checks in the replay checkout, writing each record as a line of JSON to out and printing the share the
// gate refused.
func mutateSteps(ctx context.Context, root, since string, limit int, out io.Writer) error {
	steps, err := landedSteps(ctx, root, since)
	if err != nil {
		return err
	}
	if err := processlock.EnsureStateDirectory(root); err != nil {
		return err
	}
	state := filepath.Join(root, processlock.StateDirectory)
	lock, err := processlock.Acquire(filepath.Join(state, replayLock), clioptions.PrivateFileMode)
	if err != nil {
		return fmt.Errorf("another replay holds the replay checkout: %w", err)
	}
	defer lock.Close()
	checkout, err := filepath.Abs(filepath.Join(state, replayCheckout))
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(out)
	var mutants, refused, taken int
	for _, step := range steps {
		if taken == limit {
			break
		}
		record, err := mutate(ctx, root, checkout, step)
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if err != nil {
			record.Skipped = err.Error()
		}
		if record.Skipped == "" {
			taken++
			mutants, refused = mutants+record.Mutants, refused+record.Refused
		}
		if err := encoder.Encode(record); err != nil {
			return err
		}
	}
	if taken == 0 {
		return errors.New("loop: no landed step had a bad alternative its check could be put to")
	}
	fmt.Fprintf(os.Stderr, "loop: the gate refused %d of %d bad alternatives of %d landed steps\n", refused, mutants, taken)
	return nil
}
