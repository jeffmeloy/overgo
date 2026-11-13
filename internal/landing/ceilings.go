package landing

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/testevidence"
)

// lowerCeilings writes each ceiling a count fell under, as the invariant tests said it (testevidence.CeilingFell),
// into the checkout, and takes it into the commit and the candidate the next stage reads: a ceiling only falls, and a
// refusal whose fix the tool knows is not a refusal (refusal-advice/ceilings-fall). A file the checkout holds a change
// of its own to, which the commit leaves out, is not written over.
func (l *landing) lowerCeilings(ctx context.Context, output string) error {
	opening, _, _ := strings.Cut(testevidence.CeilingFell, "%")
	lowered := false
	for line := range strings.Lines(output) {
		var event struct{ Output string }
		_ = json.Unmarshal([]byte(line), &event)
		_, fall, found := strings.Cut(event.Output, opening)
		if !found {
			continue
		}
		var name, file string
		var ceiling, count int
		if _, err := fmt.Sscanf(opening+fall, testevidence.CeilingFell, &name, &file, &ceiling, &count); err != nil {
			return err
		}
		read, err := os.ReadFile(filepath.Join(l.stageRoot(), filepath.FromSlash(file)))
		if err != nil {
			return err
		}
		checkout := filepath.Join(l.Repo, filepath.FromSlash(file))
		held, _ := os.ReadFile(checkout)
		was := fmt.Appendf(nil, "const %s = %d", name, ceiling)
		if !bytes.Equal(held, read) || bytes.Count(read, was) != 1 {
			return fmt.Errorf("%s fell to %d under its ceiling of %d, which %s does not hold as one declaration this commit alone changes; edit: lower it there", name, count, ceiling, file)
		}
		if err := os.WriteFile(checkout, bytes.Replace(read, was, fmt.Appendf(nil, "const %s = %d", name, count), 1), clioptions.OutputFileMode); err != nil {
			return err
		}
		if l.scope != nil {
			l.scope = append(l.scope, file)
		}
		fmt.Fprintf(l.Log, "land: %s fell to %d under its ceiling of %d, so %s holds %d in this commit\n", name, count, ceiling, file, count)
		lowered = true
	}
	if !lowered {
		return nil
	}
	return l.shipSet(ctx)
}
