package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"
	"time"

	"overgo/internal/evaluation"
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

// benchmarkBudget bounds one benchmark re-acquisition at evaluate's own
// ceiling on a pass; a benchmark with no measured wall is ordered by it.
const benchmarkBudget = 6 * time.Hour

// reacquireCommand builds the command that re-acquires one validation cell's
// evidence into store, the one store this command selected and read, so the
// producer never resolves a default of its own. It binds only the commands
// whose invocation is fully determined by the cell today; an unbound kind,
// such as a projection or training proof, returns an error so -run reports
// incomplete coverage rather than fabricating a wrong command. A DNA model's
// guard cell is owned by its smoke run, never the text guard.
func reacquireCommand(cell ModelValidation, store string) (string, []string, error) {
	switch cell.Validation {
	case "guard":
		switch {
		case dnaGuard(cell):
			return "go", smokeCommand(cell.Model, store), nil
		case cell.Location == "":
			return "", nil, errors.New("guard re-acquisition needs a served model location")
		}
		return "go", []string{
			"run", "./cmd/longform", "-guard", "-publish", "-repo", store,
			"-corpus", guardCorpus, "-budget", guardBudget, "-model-budget", guardModelBudget,
			cell.Location,
		}, nil
	case "image-proof", "video-proof", "speech-proof", "vqa-proof", "specialized-proof", servingValidation:
		// A media, specialized or inference model is measured the way its
		// users meet it: its active recipe runs the request its verification
		// ran, and the run is recorded as the serving attempt the scorecard
		// reads.
		if cell.Location == "" {
			return "", nil, errors.New("a capability measurement needs the model's location")
		}
		return "go", []string{"run", "./cmd/recipe", "measure", "-repo", store, "-task", string(cell.Kind), cell.Location}, nil
	}
	if !slices.Contains(benchmarkFamilies, cell.Validation) {
		return "", nil, fmt.Errorf("no re-acquisition command bound for %s/%s", cell.Kind, cell.Validation)
	}
	if cell.Location == "" {
		return "", nil, errors.New("a benchmark needs the model's location")
	}
	// A benchmark is one evaluate worker over the model and the family's
	// suite, under the protocol of the record it replaces.
	args := []string{
		"run", "./cmd/evaluate", "-all", "-worker", "-repo", store, "-model-path", cell.Location,
		"-family", cell.Validation, "-budget", benchmarkBudget.String(),
	}
	if cell.Protocol == evaluation.PromptingChatTemplate.Label() {
		args = append(args, "-chat-protocol")
	}
	return "go", args, nil
}

// executeRun runs the selected cells' re-acquisition commands smallest-first
// through the process owner, each into the selected store, then the selected
// cycle lanes. A cell or lane that fails is reported, not accepted, and the
// rest still run: one model's broken measurement withholds no other's, and
// the failure withholds promotion. Cells without a bound command are
// reported as incomplete coverage. An empty run set is the clean "nothing to
// re-acquire" result the dry run predicts when every judged cell's evidence
// and every cycle lane's pass is current.
func executeRun(ctx context.Context, output io.Writer, root, store string, plan Plan) error {
	if len(plan.Run) == 0 && !slices.ContainsFunc(plan.Suites, func(suite SelectedSuite) bool { return suite.Run }) {
		fmt.Fprintf(output, "validate -run: nothing to re-acquire; %d reused, %d unknown, %d operator-owned, %d registered-inactive\n",
			len(plan.Reuse), len(plan.Unknown), len(plan.OperatorOwned), len(plan.RegisteredInactive))
		return nil
	}
	var unbound, failed []string
	acquired := 0
	for index, sel := range plan.Run {
		name, args, err := reacquireCommand(sel.ModelValidation, store)
		if err != nil {
			unbound = append(unbound, fmt.Sprintf("%s %s: %v", displayName(sel.ModelValidation), sel.Validation, err))
			continue
		}
		fmt.Fprintf(output, "validate -run %d/%d: %s %s (%s)\n", index+1, len(plan.Run), sel.Validation, displayName(sel.ModelValidation), sel.Reason)
		if err := runCell(ctx, output, root, name, args); err != nil {
			failed = append(failed, fmt.Sprintf("%s %s: %v", displayName(sel.ModelValidation), sel.Validation, err))
			continue
		}
		acquired++
	}
	for _, suite := range plan.Suites {
		if !suite.Run {
			continue
		}
		fmt.Fprintf(output, "validate -run cycle lane %s (%s)\n", suite.Lane, suite.Reason)
		if err := runCell(ctx, output, root, "go", suite.Arguments, suite.Environment...); err != nil {
			failed = append(failed, fmt.Sprintf("cycle lane %s: %v", suite.Lane, err))
			continue
		}
		acquired++
	}
	var problems []string
	if len(failed) > 0 {
		problems = append(problems, fmt.Sprintf("%d failed:\n  %s", len(failed), strings.Join(failed, "\n  ")))
	}
	if len(unbound) > 0 {
		problems = append(problems, fmt.Sprintf("%d have no bound command yet:\n  %s", len(unbound), strings.Join(unbound, "\n  ")))
	}
	if len(problems) > 0 {
		return fmt.Errorf("validate -run: re-acquired %d cell(s); %s", acquired, strings.Join(problems, "; "))
	}
	fmt.Fprintf(output, "validate -run: re-acquired %d cell(s); re-run the dry run to confirm coverage\n", acquired)
	return nil
}

// runCell executes one re-acquisition command at the repository root through the
// process owner, with environment added to this process's; the invoked
// producer guards its own device admission.
func runCell(ctx context.Context, output io.Writer, root, name string, args []string, environment ...string) error {
	var stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: name, Args: args, Dir: root, Env: append(os.Environ(), environment...), Stdout: output, Stderr: io.MultiWriter(output, &stderr),
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
