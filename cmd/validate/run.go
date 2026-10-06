package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/landing"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/trainingworkflow"
)

// Guard re-acquisition policy: the fixed corpus and budgets a text guard cell
// re-acquires under, so a re-run binds evidence at the current surface.
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
		args, _, err := trainingRun(cell, store)
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

// runScope is what -run executes: the whole cycle, the background cells alone
// (which the loop asks for after promotion's decision), or one half of the
// cycle so its scorecard can publish between them.
type runScope int

const (
	scopeCycle runScope = iota
	scopeBackground
	scopeCells
	scopeLanes
)

// runScopeOf names the scope the flags select; the three narrowing flags
// exclude one another.
func runScopeOf(background, cellsOnly, lanesOnly bool) (runScope, error) {
	switch {
	case background && (cellsOnly || lanesOnly) || cellsOnly && lanesOnly:
		return scopeCycle, errors.New("validate: -background, -cells and -lanes select different runs; give at most one")
	case background:
		return scopeBackground, nil
	case cellsOnly:
		return scopeCells, nil
	case lanesOnly:
		return scopeLanes, nil
	}
	return scopeCycle, nil
}

// executeRun runs the cells promotion waits on smallest-first through the
// process owner, each into the selected store, then the cycle lanes; the scope
// narrows that to the background cells, the blocking cells or the lanes. A cell or lane is acquired only as
// acquire judges its run, so a go test passes by the outcomes it requires,
// never by its exit. One that fails is reported, not accepted, and the rest
// still run: one model's broken measurement withholds no other's, and a
// blocking failure withholds promotion. One whose required tests only
// declared themselves inapplicable here acquired nothing and withholds too,
// reported apart from the failures, as cells without a bound command are
// reported as incomplete coverage. A training run runs under its route's
// environment and publishes its checkpoint beneath private invocation scratch.
// Its exact fresh gate/run handoff is checked after actual process completion
// and store refresh; a projection proof's producer runs under the environment it
// publishes by. An empty run set is the clean "nothing to re-acquire" result. A timed cell
// runs while no landing does and none starts, and the quality cells and the lanes, whose
// results are not times, run beside landings.
func executeRun(ctx context.Context, output io.Writer, root, storeRoot string, store *overgodb.Store, plan Plan, scope runScope) error {
	cells, suites := plan.Run, plan.Suites
	switch scope {
	case scopeBackground:
		cells, suites = plan.Background, nil
	case scopeCells:
		suites = nil
	case scopeLanes:
		cells = nil
	}
	if len(cells)+len(suites) == 0 {
		fmt.Fprintf(output, "validate -run: nothing to re-acquire; %d operator-owned, %d registered-inactive\n",
			len(plan.OperatorOwned), len(plan.RegisteredInactive))
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
	credit := cycleCellCredit(ctx, root, storeRoot)
	for index, sel := range cells {
		var name string
		var args, environment []string
		var err error
		switch sel.Validation {
		case trainingProof:
			name = "go"
			args, environment, err = trainingRun(sel.ModelValidation, storeRoot)
		case projectionProof:
			name = "go"
			args, environment, err = projectionRun(sel.ModelValidation, storeRoot)
		default:
			name, args, err = reacquireCommand(sel.ModelValidation, storeRoot)
		}
		if err != nil {
			unbound = append(unbound, fmt.Sprintf("%s %s: %v", displayName(sel.ModelValidation), sel.Validation, err))
			continue
		}
		if credit.has(sel.ModelValidation) {
			fmt.Fprintf(output, "validate -run %d/%d: %s %s credited: an earlier run of this cycle acquired it, so it is not run again\n", index+1, len(cells), sel.Validation, displayName(sel.ModelValidation))
			acquired++
			continue
		}
		fmt.Fprintf(output, "validate -run %d/%d: %s %s (%s; %s)\n", index+1, len(cells), sel.Validation, displayName(sel.ModelValidation), sel.Reason, sel.Promotion)
		acquireCell := func() error {
			if sel.Validation == trainingProof {
				return acquireTraining(ctx, output, root, store, sel.ModelValidation, name, args, environment...)
			}
			return acquire(ctx, output, root, name, args, environment...)
		}
		if timedValidation(sel.Validation) {
			err = landing.Measuring(ctx, output, acquireCell)
		} else {
			err = acquireCell()
		}
		record(fmt.Sprintf("%s %s", displayName(sel.ModelValidation), sel.Validation), err)
		if err == nil {
			if creditErr := credit.add(sel.ModelValidation); creditErr != nil {
				fmt.Fprintf(output, "validate -run: the cycle's credit for this cell was not kept: %v\n", creditErr)
			}
		}
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
	fmt.Fprintf(output, "validate -run: re-acquired %d cell(s)\n", acquired)
	return nil
}

// timedValidation says whether a cell's result is a time the scorecard compares: the
// serving and capability measurements read first output, throughput and load, and the
// proofs that train read load, so a build or a test beside one slows what it reads. The
// benchmarks, the guards and the lanes read answers and memory, which a busy machine does not change.
func timedValidation(validation string) bool {
	switch validation {
	case trainingProof, projectionProof, servingValidation, "image-proof", "video-proof", "speech-proof", "vqa-proof", "specialized-proof":
		return true
	}
	return false
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
// records, or, like the journeys lane's cmd/webui-lane, judges its own typed
// test report.
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

// acquireTraining owns this invocation's scratch until actual child completion,
// then credits only the exact fresh proof the child handed off. Active aliases
// and another concurrent same-recipe run cannot substitute for that result.
func acquireTraining(ctx context.Context, output io.Writer, root string, store *overgodb.Store, cell ModelValidation, name string, args []string, environment ...string) (err error) {
	if store == nil {
		return errors.New("training proof acquisition requires the selected store")
	}
	at := slices.Index(args, "-out")
	if at < 0 || at+1 >= len(args) {
		return errors.New("training proof command names no checkpoint output")
	}
	scratch, err := os.MkdirTemp("", "overgo-validate-proof-")
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, os.RemoveAll(scratch)) }()
	if err := store.Refresh(ctx); err != nil {
		return err
	}
	_, before := store.Head()
	args = slices.Clone(args)
	args[at+1] = filepath.Join(scratch, "checkpoint")
	path := filepath.Join(scratch, "verification.json")
	environment = append(slices.Clone(environment), trainingworkflow.VerificationOutputEnvironment+"="+path)
	err = runCell(ctx, output, root, name, args, environment...)
	if err = errors.Join(err, store.Refresh(context.WithoutCancel(ctx))); err != nil {
		return err
	}
	var handoff modelrecipe.Verification
	if err := jsonfile.DecodeStrict(path, &handoff); err != nil {
		return fmt.Errorf("training proof handoff: %w", err)
	}
	if _, err := runrecord.VerifyGateRun(ctx, store, cell.Recipe, handoff.Gate, handoff.Run); err != nil {
		return fmt.Errorf("training proof handoff: %w", err)
	}
	for _, id := range []artifact.ID{handoff.Gate, handoff.Run} {
		introduced, found, err := store.ArtifactIntroduction(ctx, id)
		if err != nil {
			return err
		}
		if !found || introduced.Sequence <= before {
			return errors.New("training proof handoff predates this producer attempt")
		}
	}
	fmt.Fprintf(output, "validate -run: acquired training proof %s/%s\n", handoff.Gate, handoff.Run)
	return nil
}

// displayName names a cell by its model name when present, else its identity.
func displayName(cell ModelValidation) string {
	if cell.ModelName != "" {
		return cell.ModelName
	}
	return cell.Model.String()
}
