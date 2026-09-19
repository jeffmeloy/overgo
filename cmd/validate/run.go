package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/processcontrol"
)

// Guard re-acquisition policy: the fixed corpus and budgets a text guard cell
// re-acquires under, matching the accepted guard-catalog protocol so a re-run
// binds evidence at the current surface.
const (
	guardCorpus      = "cmd/longform/testdata/guard-corpus.txt"
	guardBudget      = "90m"
	guardModelBudget = "25m"
)

// reacquireCommand builds the command that re-acquires one validation cell's
// evidence. It binds only the commands whose invocation is fully determined by
// the cell today; an unbound kind returns an error so -run reports incomplete
// coverage rather than fabricating a wrong command. Binding the remaining
// modality producers (media proofs, accuracy benchmarks) is incremental.
func reacquireCommand(cell ModelValidation) (string, []string, error) {
	switch cell.Validation {
	case "guard":
		if cell.Location == "" {
			return "", nil, errors.New("guard re-acquisition needs a served model location")
		}
		return "go", []string{
			"run", "./cmd/longform", "-guard", "-publish",
			"-corpus", guardCorpus, "-budget", guardBudget, "-model-budget", guardModelBudget,
			cell.Location,
		}, nil
	default:
		return "", nil, fmt.Errorf("no re-acquisition command bound for %s/%s", cell.Kind, cell.Validation)
	}
}

// executeRun runs the selected cells' re-acquisition commands smallest-first
// through the process owner, stopping on the first failure so a broken run never
// masquerades as accepted. Cells without a bound command are reported as
// incomplete coverage. An empty run set is the clean "nothing to re-acquire"
// result the dry run predicts for a commit that moves no surface.
func executeRun(ctx context.Context, output io.Writer, root string, plan Plan) error {
	if len(plan.Run) == 0 {
		fmt.Fprintf(output, "validate -run: nothing to re-acquire; %d reused, %d operator-owned, %d registered-inactive\n",
			len(plan.Reuse), len(plan.OperatorOwned), len(plan.RegisteredInactive))
		return nil
	}
	var unbound []string
	acquired := 0
	for _, sel := range plan.Run {
		name, args, err := reacquireCommand(sel.ModelValidation)
		if err != nil {
			unbound = append(unbound, fmt.Sprintf("%s %s: %v", displayName(sel.ModelValidation), sel.Validation, err))
			continue
		}
		fmt.Fprintf(output, "validate -run %d/%d: %s %s (%s)\n", acquired+1, len(plan.Run), sel.Validation, displayName(sel.ModelValidation), sel.Reason)
		if err := runCell(ctx, output, root, name, args); err != nil {
			return fmt.Errorf("validate -run: %s %s failed: %w", displayName(sel.ModelValidation), sel.Validation, err)
		}
		acquired++
	}
	if len(unbound) > 0 {
		return fmt.Errorf("validate -run: re-acquired %d cell(s); %d have no bound command yet:\n  %s",
			acquired, len(unbound), strings.Join(unbound, "\n  "))
	}
	fmt.Fprintf(output, "validate -run: re-acquired %d cell(s); re-run the dry run to confirm coverage\n", acquired)
	return nil
}

// runCell executes one re-acquisition command at the repository root through the
// process owner; the invoked producer guards its own device admission.
func runCell(ctx context.Context, output io.Writer, root, name string, args []string) error {
	var stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: name, Args: args, Dir: root, Env: os.Environ(), Stdout: output, Stderr: io.MultiWriter(output, &stderr),
	})
	if err != nil {
		return fmt.Errorf("%w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if receipt.ExitCode != 0 {
		return fmt.Errorf("exit=%d: %s", receipt.ExitCode, strings.TrimSpace(stderr.String()))
	}
	return nil
}

// displayName names a cell by its model name when present, else its identity.
func displayName(cell ModelValidation) string {
	if cell.ModelName != "" {
		return cell.ModelName
	}
	return cell.Model.String()
}
