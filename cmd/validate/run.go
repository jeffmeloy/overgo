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

	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/processcontrol"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
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
// whose invocation is fully determined by the cell as its records were read;
// an unbound cell, such as a projection proof nothing committed grounds,
// returns why, so -run reports incomplete coverage rather than fabricating a
// wrong command. A DNA model's guard cell is owned by its smoke run, never
// the text guard.
func reacquireCommand(cell ModelValidation, store string) (string, []string, error) {
	switch cell.Validation {
	case trainingProof:
		args, _, _, err := trainingRun(cell, store)
		if err != nil {
			return "", nil, err
		}
		return "go", args, nil
	case projectionProof:
		args, _, err := projectionRun(cell, store)
		if err != nil {
			return "", nil, err
		}
		return "go", args, nil
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
		// reads. An inference runner opens the weights file it serves; every
		// other capability resolves the directory the model was registered
		// from, its configuration beside its weights.
		path := cell.Directory
		if cell.Validation == servingValidation {
			path = cell.Location
		}
		if path == "" {
			return "", nil, errors.New("a capability measurement needs the model's registered directory, or a served model's weights file")
		}
		return "go", []string{"run", "./cmd/recipe", "measure", "-repo", store, "-task", string(cell.Kind), path}, nil
	}
	if !benchmark(cell) {
		return "", nil, fmt.Errorf("no re-acquisition command bound for %s/%s", cell.Kind, cell.Validation)
	}
	if cell.Location == "" {
		return "", nil, errors.New("a benchmark needs the model's location")
	}
	// A benchmark is one evaluate worker over the model and the family's
	// suite, under the protocol of the record it replaces, which it replays
	// first.
	args := []string{
		"run", "./cmd/evaluate", "-all", "-worker", "-repo", store, "-model-path", cell.Location,
		"-family", cell.Validation, "-budget", benchmarkBudget.String(),
	}
	if cell.Protocol == evaluation.PromptingChatTemplate.Label() {
		args = append(args, "-chat-protocol")
	}
	if cell.Replay.Valid() {
		args = append(args, "-replay", cell.Replay.String())
	}
	return "go", args, nil
}

// executeRun runs the cells promotion waits on smallest-first through the
// process owner, each into the selected store, then the cycle lanes; with
// background it runs the background cells alone, which the loop
// asks for after promotion's decision. A cell or lane is acquired only as
// acquire judges its run, so a go test passes by the outcomes it requires,
// never by its exit. One that fails is reported, not accepted, and the rest
// still run: one model's broken measurement withholds no other's, and a
// blocking failure withholds promotion. One whose required tests only
// declared themselves inapplicable here acquired nothing and withholds too,
// reported apart from the failures, as cells without a bound command are
// reported as incomplete coverage. A training run runs under its route's
// environment and publishes its checkpoint beneath a scratch directory
// emptied before and after it, since its proof is the verification it
// records; a projection proof's producer runs under the environment it
// publishes by. An empty run set is the clean "nothing to re-acquire" result.
func executeRun(ctx context.Context, output io.Writer, root, store string, plan Plan, background bool) error {
	cells, suites := plan.Run, plan.Suites
	if background {
		cells, suites = plan.Background, nil
	}
	if len(cells)+len(suites) == 0 {
		fmt.Fprintf(output, "validate -run: nothing to re-acquire; %d reused, %d unknown, %d operator-owned, %d registered-inactive\n",
			len(plan.Reuse), len(plan.Unknown), len(plan.OperatorOwned), len(plan.RegisteredInactive))
		return nil
	}
	var unbound, failed, inapplicable []string
	acquired := 0
	record := func(subject string, err error) {
		switch {
		case err == nil:
			acquired++
		case errors.Is(err, errInapplicable):
			inapplicable = append(inapplicable, fmt.Sprintf("%s: %v", subject, err))
		default:
			failed = append(failed, fmt.Sprintf("%s: %v", subject, err))
		}
	}
	for index, sel := range cells {
		name, args, err := reacquireCommand(sel.ModelValidation, store)
		if err != nil {
			unbound = append(unbound, fmt.Sprintf("%s %s: %v", displayName(sel.ModelValidation), sel.Validation, err))
			continue
		}
		fmt.Fprintf(output, "validate -run %d/%d: %s %s (%s; %s)\n", index+1, len(cells), sel.Validation, displayName(sel.ModelValidation), sel.Reason, sel.Promotion)
		var environment []string
		scratch := ""
		switch sel.Validation {
		case trainingProof:
			_, environment, scratch, _ = trainingRun(sel.ModelValidation, store)
		case projectionProof:
			_, environment, _ = projectionRun(sel.ModelValidation, store)
		}
		err = os.RemoveAll(scratch)
		if err == nil {
			err = acquire(ctx, output, root, name, args, environment...)
		}
		record(fmt.Sprintf("%s %s", displayName(sel.ModelValidation), sel.Validation), errors.Join(err, os.RemoveAll(scratch)))
	}
	for _, suite := range suites {
		fmt.Fprintf(output, "validate -run cycle lane %s\n", suite.Lane)
		record("cycle lane "+suite.Lane, acquire(ctx, output, root, "go", suite.Arguments, suite.Environment...))
	}
	var problems []string
	if len(failed) > 0 {
		problems = append(problems, fmt.Sprintf("%d failed:\n  %s", len(failed), strings.Join(failed, "\n  ")))
	}
	if len(inapplicable) > 0 {
		problems = append(problems, fmt.Sprintf("%d acquired nothing, their required tests %s:\n  %s",
			len(inapplicable), errInapplicable, strings.Join(inapplicable, "\n  ")))
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

// errInapplicable marks a required outcome that only declared exclusions
// answered: the tests it selects stated, through testskip.NotApplicable, why
// they cannot apply where they ran. It is neither a failure nor a pass.
var errInapplicable = errors.New("declared inapplicable here")

// acquire runs one cell's or lane's command and says why it acquired
// nothing. A go test command is acquired by the outcomes it requires, never
// by its exit: it runs once with -json, and testevidence's report of that
// run's events must show each package the command names completing with a
// passing test its -run selector matches, as requiredPass judges. Its
// packages' progress and any failure's retained output go to output. Any
// other command is a producer whose exit is its own verdict: it publishes
// records the next dry run judges, or, like the journeys lane's
// cmd/webui-lane, judges its own typed test report.
func acquire(ctx context.Context, output io.Writer, root, name string, args []string, environment ...string) error {
	if name != "go" || len(args) == 0 || args[0] != "test" {
		return runCell(ctx, output, root, name, args, environment...)
	}
	// The command is read as every verifier is, so its packages and selector
	// mean what they mean to testevidence.
	words := []string{name}
	for _, arg := range args {
		words = append(words, "'"+strings.ReplaceAll(arg, "'", `'\''`)+"'")
	}
	parsed, err := runrecord.ParseVerify(strings.Join(words, " "))
	if err != nil {
		return err
	}
	invocation, selected := parsed.GoTests()[0], func(string) bool { return true }
	if invocation.Target != nil {
		selected = invocation.Target.MatchString
	}
	report, err := testevidence.RunGoTestCommand(ctx, processcontrol.Command{
		Path: name, Args: slices.Insert(slices.Clone(args), 1, "-json"), Dir: root, Env: append(os.Environ(), environment...),
	}, testevidence.GoTestOptions{DiagnosticBytes: clioptions.DiagnosticTailBytes, Progress: func(done testevidence.PackageProgress) {
		fmt.Fprintf(output, "%s\t%s\t%s\n", done.Action, done.Package, done.Elapsed)
	}})
	required := requiredPass(report, invocation.Packages, selected)
	if err == nil {
		return required
	}
	for _, diagnostic := range report.Diagnostics {
		fmt.Fprintf(output, "validate -run: diagnostic: %s\n", diagnostic)
	}
	if required != nil {
		err = fmt.Errorf("%w; %s", err, required)
	}
	return err
}

// requiredPass says why one go test run's report does not prove what the run
// requires of the packages its command names as ./directory: nil when the
// run is complete, every test passing or declaring itself inapplicable, and
// in each package a test selected matches passed. RequireComplete names the
// first test that failed, did not finish or skipped without declaring
// itself inapplicable; PackagePassed is each package's authority, and the
// observed tests say which passed. A package the run never reported, or one
// in which no selected test passed, is absent, and a same-named test of
// another package lends it nothing. When declared exclusions are all that
// answered, the refusal wraps errInapplicable.
func requiredPass(report testevidence.GoTestReport, packages []string, selected func(string) bool) error {
	if err := testevidence.RequireComplete(report); err != nil {
		return err
	}
	var refused, declared []string
	for _, pkg := range packages {
		relative := strings.TrimPrefix(pkg, "./")
		at := slices.IndexFunc(report.Executions, func(execution testevidence.PackageExecution) bool {
			return execution.Package == pkg || strings.HasSuffix(execution.Package, "/"+relative)
		})
		if at < 0 {
			refused = append(refused, pkg+" is absent: the run reported no such package")
			continue
		}
		name := report.Executions[at].Package
		passed := slices.ContainsFunc(report.TestCosts(), func(test testevidence.TestExecution) bool {
			return test.Package == name && test.Action == "pass" && selected(test.Name)
		})
		excluded := slices.DeleteFunc(slices.Clone(report.ClassifiedSkipped), func(skipped string) bool {
			test, found := strings.CutPrefix(skipped, name+": ")
			return !found || !selected(test)
		})
		switch {
		case passed && !report.PackagePassed(name):
			refused = append(refused, name+" did not complete a passing run")
		case !passed && len(excluded) > 0:
			declared = append(declared, strings.Join(excluded, ", "))
		case !passed:
			refused = append(refused, name+" is absent: no test the run selected passed there")
		}
	}
	switch {
	case len(refused) > 0:
		return errors.New(strings.Join(refused, "; "))
	case len(declared) > 0:
		return fmt.Errorf("%w: %s", errInapplicable, strings.Join(declared, "; "))
	case len(packages) == 0:
		return errors.New("the run names no package it requires")
	}
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
