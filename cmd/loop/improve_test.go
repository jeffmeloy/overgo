package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/loop"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// scoredRow is one fully explained scorecard row of the model named name:
// every cell measured by source, with the held-out accuracy, the failure
// rate and the load cell the caller gives.
func scoredRow(t *testing.T, source artifact.ID, name string, task recipe.Task, accuracy, failures float64, load scorecard.Cell) scorecard.Row {
	cells := map[metric.Name]scorecard.Cell{}
	for _, column := range metric.Columns {
		cells[column] = scorecard.Cell{Value: 1, Source: source}
	}
	// The reading is one a test compares with a reference, so a row files the
	// gaps these tests are of and not the comparison every unchecked reading asks for.
	cells[metric.Quality] = scorecard.Cell{Value: accuracy, Label: "accuracy, store/mmlu (raw-completion)", Source: source, Direction: runrecord.DirectionMaximize,
		Compared: "internal/fixture TestServedPathMatchesItsReference",
		Contract: scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: "store/mmlu", Protocol: "raw-completion", Comparison: testutil.ArtifactID(t, artifact.KindProfile, "mmlu-raw-environment")}}
	cells[metric.Failures] = scorecard.Cell{Value: failures, Label: "attempts", Source: source}
	cells[metric.Load] = load
	return scorecard.Row{Model: testutil.ArtifactID(t, artifact.KindModel, name), Name: name, Task: task, Cells: cells}
}

// emptyPlan writes a lane's plan with no rows at a path of its own.
func emptyPlan(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "plan.json")
	if err := plan.Save(path, plan.Plan{Campaign: "improvement", Doctrine: "the scorecard's gaps are the work", Lane: "master-lead"}); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestScorecardGapsBecomeProposals writes a published scorecard's gaps into
// the plan as plain rows of its lane: failed attempts first, then a held-out
// quality below the baseline, then each task's unmeasured metric, one row
// per task and metric listing every model it names with its reason. A
// metric that does not apply proposes nothing. Each row dispatches and is
// verified by the gap closing on the scorecard the store holds; a second
// pass over the same gaps adds nothing.
func TestScorecardGapsBecomeProposals(t *testing.T) {
	t.Parallel()
	source := testutil.ArtifactID(t, artifact.KindEvidence, "loop-proposals")
	measured := scorecard.Cell{Value: 1, Source: source}
	unmeasured := scorecard.Cell{Reason: "no benchmark load time and no recorded load phase"}
	baseline := scorecard.Record{Commit: "a", Rows: []scorecard.Row{scoredRow(t, source, "alpha", recipe.TaskInference, 0.5, 0, measured)}}
	image := scoredRow(t, source, "gamma", recipe.TaskImageGen, 0.5, 0, unmeasured)
	image.Cells[metric.Throughput] = scorecard.Cell{Reason: "image-gen returns its output whole", NotApplicable: true}
	published := scorecard.Record{Commit: "b", Name: "scorecard-20260929T120000Z.json", Rows: []scorecard.Row{
		scoredRow(t, source, "alpha", recipe.TaskInference, 0.4, 0.25, unmeasured),
		scoredRow(t, source, "beta", recipe.TaskInference, 0.7, 0, unmeasured),
		image,
	}}
	improver := improvement{root: t.TempDir(), plan: emptyPlan(t)}
	added, err := improver.Propose(published, published.Gaps(&baseline))
	if err != nil || added != 4 {
		t.Fatalf("proposed %d rows, %v; want four", added, err)
	}
	document, err := plan.Load(improver.plan)
	if err != nil {
		t.Fatal(err)
	}
	var decline string
	for _, gap := range published.Gaps(&baseline) {
		if gap.Restore {
			decline = gap.Key
		}
	}
	var keys []string
	for _, item := range document.Items {
		keys = append(keys, item.ID)
		// A decline's row carries the value its cell is to be restored to, which no later pass rewrites.
		verify := "go run ./cmd/validate -scorecard -gap " + item.ID
		if item.ID == decline {
			verify += " -target 0.5"
		}
		if len(item.Steps) != 1 || item.Steps[0].Verify != verify || !strings.Contains(item.Steps[0].Rationale, published.Name) {
			t.Fatalf("row %+v is not verified by its gap closing on the store's scorecard", item)
		}
		if item.Owner != document.Lane {
			t.Fatalf("row %s is owned by %q, want the plan's lane %q", item.ID, item.Owner, document.Lane)
		}
	}
	want := []string{
		"scorecard-failing-inference",
		decline,
		"scorecard-unmeasured-image-gen",
		"scorecard-unmeasured-inference",
	}
	if !slices.Equal(keys, want) {
		t.Fatalf("proposed rows %q, want %q", keys, want)
	}
	if rationale := document.Items[3].Steps[0].Rationale; !strings.Contains(rationale, "- alpha load-time: "+unmeasured.Reason) || !strings.Contains(rationale, "- beta load-time: "+unmeasured.Reason) {
		t.Fatalf("the inference load row does not list both models with their reason:\n%s", rationale)
	}
	if !strings.Contains(document.Items[1].Steps[0].Rationale, "alpha held-out-quality") || !strings.Contains(document.Items[1].Steps[0].Rationale, "0.5 to 0.4 (-0.1, -20.0%") {
		t.Fatalf("the regression row does not state the fall:\n%s", document.Items[1].Steps[0].Rationale)
	}
	if item, step, open := plan.Current(document, ""); !open || item.ID != want[0] || step.ID != proposalStep {
		t.Fatalf("the plan dispatches %s/%s (open %v), want the failed attempts first", item.ID, step.ID, open)
	}
	if added, err := improver.Propose(published, published.Gaps(&baseline)); err != nil || added != 0 {
		t.Fatalf("a second pass proposed %d rows, %v; want none", added, err)
	}
}

// gitFixture runs git in directory as a fixture author and returns its
// trimmed output.
func gitFixture(t *testing.T, directory string, arguments ...string) string {
	t.Helper()
	command := exec.Command("git", append([]string{"-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "-c", "commit.gpgsign=false"}, arguments...)...)
	command.Dir = directory
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", arguments, err, output)
	}
	return strings.TrimSpace(string(output))
}

// TestCycleComparesWithTheNewestEarlierScorecardWithoutAMain publishes two scorecards of a
// commit in a repository that holds no main, the second worse than the first. The cell is
// compared with its newest earlier measurement, the first record, whatever any branch points
// at: the decline is proposed as the row that restores it, to the value it declined from.
func TestCycleComparesWithTheNewestEarlierScorecardWithoutAMain(t *testing.T) {
	t.Parallel()
	repository := t.TempDir()
	gitFixture(t, repository, "init", "-q", "-b", "master")
	gitFixture(t, repository, "commit", "-q", "--allow-empty", "-m", "measured")
	measured := gitFixture(t, repository, "rev-parse", "HEAD")

	source := testutil.ArtifactID(t, artifact.KindEvidence, "loop-no-main-reference")
	records := filepath.Join(t.TempDir(), "scorecards")
	published := time.Date(2026, time.September, 30, 12, 0, 0, 0, time.UTC)
	publish := func(accuracy float64) error {
		published = published.Add(time.Hour)
		_, err := scorecard.Publish(records, published, scorecard.Record{Commit: measured, Rows: []scorecard.Row{scoredRow(t, source, "alpha", recipe.TaskInference, accuracy, 0, scorecard.Cell{Value: 1, Source: source})}})
		return err
	}
	if err := publish(0.5); err != nil {
		t.Fatal(err)
	}
	improver := improvement{root: repository, plan: emptyPlan(t), records: records, measure: func(string, func() error) error { return publish(0.4) }, fill: func(string) error { return nil }}
	added, err := loop.Improve(improver)
	if err != nil || added != 1 {
		t.Fatalf("a cycle against the record before it proposed %d row(s), %v; want the one decline", added, err)
	}
	document, err := plan.Load(improver.plan)
	if err != nil || len(document.Items) != 1 || !strings.HasPrefix(document.Items[0].ID, "scorecard-regressed-") || !strings.HasSuffix(document.Items[0].Steps[0].Verify, " -target 0.5") {
		t.Fatalf("plan = %+v (%v), want the decline's row, to be restored to 0.5", document.Items, err)
	}
}

// TestCycleRunsInItsOwnCheckout runs improvement cycles while rows land in
// the main checkout, one during the measurement and one during the fill.
// The scorecard is measured and the background filled in one cycle
// checkout kept in the main checkout's process state, at the commit the
// main checkout's HEAD held when the cycle began, clean throughout, whether
// the fill passed or failed. The next cycle reuses that directory, moved to
// the main checkout's new HEAD, and the main checkout stays clean beside
// it. A cycle checkout another cycle holds refuses the next measurement,
// and so does one a run left a change in, which stays.
func TestCycleRunsInItsOwnCheckout(t *testing.T) {
	t.Parallel()
	repository := t.TempDir()
	git := func(directory string, arguments ...string) string { return gitFixture(t, directory, arguments...) }
	git(repository, "init", "-q", "-b", "master")
	planPath := filepath.Join(repository, "plan.json")
	if err := plan.Save(planPath, plan.Plan{Campaign: "improvement", Doctrine: "the scorecard's gaps are the work", Lane: "master-lead"}); err != nil {
		t.Fatal(err)
	}
	git(repository, "add", "plan.json")
	git(repository, "commit", "-q", "-m", "measured")
	mainCheckout := git(repository, "rev-parse", "--show-toplevel")

	source := testutil.ArtifactID(t, artifact.KindEvidence, "loop-cycle-worktree")
	records := filepath.Join(t.TempDir(), "scorecards")
	published := time.Date(2026, time.October, 2, 12, 0, 0, 0, time.UTC)
	improver := improvement{root: repository, plan: planPath, records: records}
	var cycle string
	for round, failure := range []error{nil, errors.New("a background suite failed")} {
		measured := git(repository, "rev-parse", "HEAD")
		var ran []string
		// inCycle checks the stage runs where and at what it should, then
		// lands a row in the main checkout while it runs.
		inCycle := func(stage, checkout string) {
			ran = append(ran, stage)
			cycle = cmp.Or(cycle, checkout)
			if relative, err := filepath.Rel(repository, checkout); checkout != cycle || err != nil || strings.HasPrefix(relative, "..") {
				t.Errorf("round %d: the %s ran in %s; want the cycle checkout %s inside the main checkout every time", round, stage, checkout, cycle)
			}
			if top := git(checkout, "rev-parse", "--show-toplevel"); top == mainCheckout {
				t.Errorf("round %d: the %s ran in the main checkout %s", round, stage, top)
			}
			if head, status := git(checkout, "rev-parse", "HEAD"), git(checkout, "status", "--porcelain"); head != measured || status != "" {
				t.Errorf("round %d: the %s ran at %s with status %q while the main checkout landed; want the measured commit %s, clean", round, stage, head, status, measured)
			}
			landed := fmt.Sprintf("landed-during-%s-%d", stage, round)
			if added, err := improver.Propose(scorecard.Record{Name: "another session"}, []scorecard.Gap{{Key: landed, Title: "a row another session lands", Detail: "written while the cycle runs"}}); err != nil || added != 1 {
				t.Errorf("round %d: the main checkout's plan took %d row(s) during the %s, %v; want one", round, added, stage, err)
			}
			git(repository, "commit", "-q", "-am", "landed during the "+stage)
		}
		improver.measure = func(checkout string, _ func() error) error {
			inCycle("measure", checkout)
			published = published.Add(time.Hour)
			_, err := scorecard.Publish(records, published, scorecard.Record{Commit: git(checkout, "rev-parse", "HEAD"), Rows: []scorecard.Row{
				scoredRow(t, source, "alpha", recipe.TaskInference, 0.5, 0, scorecard.Cell{Value: 1, Source: source})}})
			return err
		}
		improver.fill = func(checkout string) error {
			inCycle("fill", checkout)
			return failure
		}
		if _, err := loop.Improve(improver); err != nil {
			t.Fatalf("round %d: %v", round, err)
		}
		if !slices.Equal(ran, []string{"measure", "fill"}) {
			t.Fatalf("round %d: ran %q; want the measure, then the fill", round, ran)
		}
		if status := git(repository, "status", "--porcelain"); status != "" {
			t.Fatalf("round %d: the main checkout shows %q beside its cycle checkout; want it clean", round, status)
		}
		if registered := strings.Count(git(repository, "worktree", "list", "--porcelain"), "worktree "); registered != 2 {
			t.Fatalf("round %d: %d worktrees registered after the cycle; want the main checkout and the cycle checkout", round, registered)
		}
	}
	improver.measure = func(string, func() error) error {
		t.Error("measured in a cycle checkout another cycle holds or a run left a change in")
		return nil
	}
	held, err := processlock.Acquire(filepath.Join(repository, processlock.StateDirectory, cycleLock), cycleLockMode)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := loop.Improve(improver); !errors.Is(err, processlock.ErrBusy) {
		t.Fatalf("a cycle checkout another cycle holds gave %v; want a refusal", err)
	}
	if err := held.Close(); err != nil {
		t.Fatal(err)
	}
	stray := filepath.Join(cycle, "stray.txt")
	if err := os.WriteFile(stray, []byte("a run left this"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loop.Improve(improver); err == nil || !strings.Contains(err.Error(), "stray.txt") {
		t.Fatalf("a cycle checkout holding a change gave %v; want a refusal naming the change", err)
	}
	if _, err := os.Stat(stray); err != nil {
		t.Fatalf("the change the cycle checkout held is gone: %v", err)
	}
}

// TestUnfinishedCycleContinuesAtItsOwnCommit holds a cycle a run began and did
// not finish to the commit it began at. A run measures at one commit, keeps
// the cells it measured there as validate -run does, and is cut short before
// it publishes; rows land in the main checkout meanwhile. The run after it
// measures in the cycle checkout at the first commit, where the kept cells are
// credited, publishes that commit's record and ends the cycle, which removes
// what was kept; and the cycle after that measures the commit the checkout has
// moved to.
func TestUnfinishedCycleContinuesAtItsOwnCommit(t *testing.T) {
	t.Parallel()
	repository := t.TempDir()
	git := func(directory string, arguments ...string) string { return gitFixture(t, directory, arguments...) }
	git(repository, "init", "-q", "-b", "master")
	planPath := filepath.Join(repository, "plan.json")
	if err := plan.Save(planPath, plan.Plan{Campaign: "improvement", Doctrine: "the scorecard's gaps are the work", Lane: "master-lead"}); err != nil {
		t.Fatal(err)
	}
	git(repository, "add", "plan.json")
	git(repository, "commit", "-q", "-m", "begun")
	begun := git(repository, "rev-parse", "HEAD")

	source := testutil.ArtifactID(t, artifact.KindEvidence, "loop-cycle-continued")
	records := filepath.Join(t.TempDir(), "scorecards")
	published := time.Date(2026, time.October, 9, 12, 0, 0, 0, time.UTC)
	kept := filepath.Join(repository, processlock.StateDirectory, cycleCreditFile)
	cut := errors.New("the run was cut short")
	var measured []string
	improver := improvement{root: repository, plan: planPath, records: records, fill: func(string) error { return nil }}
	improver.measure = func(checkout string, _ func() error) error {
		at := git(checkout, "rev-parse", "HEAD")
		measured = append(measured, at)
		// The first run keeps a cell and ends before it publishes.
		if len(measured) == 1 {
			credit := fmt.Sprintf(`{"commit": %q, "store": "store", "cells": ["a measured cell"]}`, at)
			return errors.Join(cut, os.WriteFile(kept, []byte(credit), cycleStateMode))
		}
		published = published.Add(time.Hour)
		_, err := scorecard.Publish(records, published, scorecard.Record{Commit: at, Rows: []scorecard.Row{
			scoredRow(t, source, "alpha", recipe.TaskInference, 0.5, 0, scorecard.Cell{Value: 1, Source: source})}})
		return err
	}
	if _, err := loop.Improve(improver); !errors.Is(err, cut) {
		t.Fatalf("the run that was cut short ended with %v", err)
	}
	git(repository, "commit", "-q", "--allow-empty", "-m", "landed while the cycle was unfinished")
	moved := git(repository, "rev-parse", "HEAD")
	if _, err := loop.Improve(improver); err != nil {
		t.Fatalf("the run that continues the cycle: %v", err)
	}
	if _, err := os.Stat(kept); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("a cycle that reached its end left its measured cells kept (%v), so the next would continue it again", err)
	}
	if _, err := loop.Improve(improver); err != nil {
		t.Fatalf("the cycle after the continued one: %v", err)
	}
	if want := []string{begun, begun, moved}; !slices.Equal(measured, want) {
		t.Fatalf("the runs measured at %v; want the unfinished cycle continued at the commit it began at, then the commit the checkout moved to: %v", measured, want)
	}
	held, err := scorecard.Records(records)
	if err != nil || len(held) != 2 || held[0].Commit != begun || held[1].Commit != moved {
		t.Fatalf("the published records are of %+v (%v); want one of the commit the cycle began at and one of the commit after it", held, err)
	}
}

// planWorld drives the loop over a plan file: its worker lands the
// dispatched row the way the gate does, by removing it, and records the
// verify command of the row it landed.
type planWorld struct {
	improvement
	worked []string
	parked []string
}

func (w *planWorld) Current() (loop.Step, bool, error) {
	document, err := plan.Load(w.plan)
	if err != nil {
		return loop.Step{}, false, err
	}
	item, step, open := plan.Current(document, "")
	return loop.Step{Item: item.ID, ID: step.ID}, open, nil
}

func (w *planWorld) Prompt(step loop.Step) (string, error) { return "TASK " + step.Key(), nil }

func (w *planWorld) RunWorker(step loop.Step, _, _ string) (string, error) {
	document, err := plan.Load(w.plan)
	if err == nil {
		for _, item := range document.Items {
			for _, candidate := range item.Steps {
				if item.ID == step.Item && candidate.ID == step.ID {
					w.worked = append(w.worked, candidate.Verify)
				}
			}
		}
		document, err = plan.Advance(document, step.Item, step.ID)
	}
	if err == nil {
		err = plan.Save(w.plan, document)
	}
	return "", err
}

func (w *planWorld) Verify(loop.Step) (string, error) { return "", nil }

func (w *planWorld) Park(step loop.Step, reason string) error {
	w.parked = append(w.parked, step.Key()+": "+reason)
	return nil
}

func (w *planWorld) Paused() bool { return false }

// TestADeclineBecomesTheRowTheLoopWorks runs the loop over a real repository. The first
// scorecard of the new commit has a lower held-out accuracy than the one before it: the
// decline becomes the row the worker lands, to be restored to the value it fell from. The next
// scorecard has the accuracy back, compared with the record before it: nothing declined, no
// row is proposed, and the plan stays empty. Each cycle's background validations fail, which
// neither stops the loop nor discards a record. No branch moves and the remote receives nothing.
func TestADeclineBecomesTheRowTheLoopWorks(t *testing.T) {
	t.Parallel()
	repository, remote := t.TempDir(), t.TempDir()
	git := func(directory string, arguments ...string) string { return gitFixture(t, directory, arguments...) }
	git(remote, "init", "-q", "--bare")
	git(repository, "init", "-q", "-b", "master")
	git(repository, "commit", "-q", "--allow-empty", "-m", "earlier")
	earlierCommit := git(repository, "rev-parse", "HEAD")
	git(repository, "remote", "add", "origin", remote)
	git(repository, "commit", "-q", "--allow-empty", "-m", "measured")
	measuredCommit := git(repository, "rev-parse", "HEAD")

	source := testutil.ArtifactID(t, artifact.KindEvidence, "loop-decline")
	records := filepath.Join(t.TempDir(), "scorecards")
	published := time.Date(2026, time.September, 29, 12, 0, 0, 0, time.UTC)
	scripted := []scorecard.Record{
		{Commit: earlierCommit, Rows: []scorecard.Row{scoredRow(t, source, "alpha", recipe.TaskInference, 0.5, 0, scorecard.Cell{Value: 1, Source: source})}},
		{Commit: measuredCommit, Rows: []scorecard.Row{scoredRow(t, source, "alpha", recipe.TaskInference, 0.4, 0, scorecard.Cell{Value: 1, Source: source})}},
		{Commit: measuredCommit, Rows: []scorecard.Row{scoredRow(t, source, "alpha", recipe.TaskInference, 0.5, 0, scorecard.Cell{Value: 1, Source: source})}},
	}
	publish := func() error {
		count, err := scorecard.Records(records)
		if err == nil {
			_, err = scorecard.Publish(records, published.Add(time.Duration(len(count))*time.Hour), scripted[len(count)])
		}
		return err
	}
	if err := publish(); err != nil {
		t.Fatal(err)
	}
	fills := 0
	fill := func(string) error {
		fills++
		return errors.New("a background suite failed")
	}
	world := &planWorld{improvement: improvement{root: repository, plan: emptyPlan(t), records: records, measure: func(string, func() error) error { return publish() }, fill: fill}}
	outcome, err := loop.Run(world, loop.Config{MaxAttemptsPerStep: 1, MaxInvocations: 2})
	if err != nil || outcome.Reason != loop.ReasonPlanComplete || len(world.parked) != 0 || fills != len(scripted)-1 {
		t.Fatalf("outcome=%+v err=%v parked=%v fills=%d", outcome, err, world.parked, fills)
	}
	if len(world.worked) != 1 || !strings.Contains(world.worked[0], "-gap scorecard-regressed-") || !strings.HasSuffix(world.worked[0], " -target 0.5") {
		t.Fatalf("the worker ran %q; want once, on the decline, to be restored to 0.5", world.worked)
	}
	if refs := git(remote, "for-each-ref"); refs != "" {
		t.Fatalf("the remote holds %q; a cycle never pushes", refs)
	}
	if all, err := scorecard.Records(records); err != nil || len(all) != len(scripted) {
		t.Fatalf("records after the cycles = %d (%v), want all %d: a failed fill discards none", len(all), err, len(scripted))
	}
}

// TestPublishedCycleCutShortIsStillWorthFinishing holds the trigger to a cycle that published its
// scorecard and was cut short before its end. A run publishes at a commit, keeps a cell there as
// validate -run does, and its process ends before the fill; nothing a cycle measures lands
// afterwards. The loop that starts next with an empty plan finishes that cycle: it measures at
// the same commit, fills, and removes what was kept, and the loop after that starts nothing. A
// cycle that returned its failure has reached its end and is not started again, and cells kept
// by a cycle that published nothing do not make the newest record's cycle look cut short.
func TestPublishedCycleCutShortIsStillWorthFinishing(t *testing.T) {
	t.Parallel()
	repository, records := t.TempDir(), filepath.Join(t.TempDir(), "scorecards")
	git := func(directory string, arguments ...string) string { return gitFixture(t, directory, arguments...) }
	git(repository, "init", "-q", "-b", "master")
	git(repository, "commit", "-q", "--allow-empty", "-m", "begun")
	begun := git(repository, "rev-parse", "HEAD")
	kept := filepath.Join(repository, processlock.StateDirectory, cycleCreditFile)
	published := time.Date(2026, time.October, 11, 12, 0, 0, 0, time.UTC)
	var ran []string
	var failure error
	publishes := true
	world := &planWorld{improvement: improvement{watch: true, root: repository, plan: emptyPlan(t), records: records,
		surface: func(context.Context, string) (map[string]bool, error) { return map[string]bool{}, nil },
		fill: func(string) error {
			ran = append(ran, "fill")
			return nil
		}}}
	// A run publishes the commit's record and keeps a cell, as validate -run does; a run that
	// finds the newest record is the commit's already stands for the credited publication.
	world.measure = func(checkout string, _ func() error) error {
		ran = append(ran, "measure")
		at := git(checkout, "rev-parse", "HEAD")
		if held, err := scorecard.Records(records); err != nil || len(held) > 0 && held[len(held)-1].Commit == at {
			return err
		}
		if publishes {
			published = published.Add(time.Hour)
			if _, err := scorecard.Publish(records, published, scorecard.Record{Commit: at}); err != nil {
				return err
			}
		}
		credit := fmt.Sprintf(`{"commit": %q, "store": "store", "cells": ["a measured cell"]}`, at)
		return errors.Join(failure, os.WriteFile(kept, []byte(credit), cycleStateMode))
	}
	// The run that is cut short: it publishes, and its process ends before the fill.
	if record, _, err := world.Measure(); err != nil || record == nil || record.Commit != begun {
		t.Fatalf("the run that is cut short published %+v, %v; want the record of %s", record, err, begun)
	}
	if worth, reason, err := world.CycleWorthRunning(); err != nil || !worth {
		t.Fatalf("a cycle that published and was cut short is worth finishing = %t (%s, %v), want true", worth, reason, err)
	}
	// The loop that starts next finds the plan empty and finishes the cycle; the one after starts nothing.
	budget := loop.Config{MaxAttemptsPerStep: 1, MaxInvocations: 1}
	for _, start := range []string{"the loop that finishes the cycle", "the loop after the cycle finished"} {
		if outcome, err := loop.Run(world, budget); err != nil || outcome.Reason != loop.ReasonPlanComplete {
			t.Fatalf("%s: outcome=%+v err=%v", start, outcome, err)
		}
		if want := []string{"measure", "measure", "fill"}; !slices.Equal(ran, want) {
			t.Fatalf("%s: ran %q, want %q", start, ran, want)
		}
		if _, err := os.Stat(kept); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("%s: the finished cycle left its cells kept (%v)", start, err)
		}
	}
	// A cycle that returns its failure has reached its end and is not started again.
	git(repository, "commit", "-q", "--allow-empty", "-m", "landed")
	failure = errors.New("a lane failed")
	if _, err := loop.Improve(world); !errors.Is(err, failure) {
		t.Fatalf("the cycle whose lane failed ended with %v", err)
	}
	if worth, reason, err := world.CycleWorthRunning(); err != nil || worth {
		t.Fatalf("a cycle that returned its failure is worth running again = %t (%s, %v), want false", worth, reason, err)
	}
	// Cells kept at another commit than the newest record's are a cycle that published nothing:
	// what moved to start that one starts it again, and here nothing moved.
	git(repository, "commit", "-q", "--allow-empty", "-m", "landed again")
	publishes = false
	if _, err := loop.Improve(world); !errors.Is(err, failure) {
		t.Fatalf("the cycle that published nothing ended with %v", err)
	}
	if _, err := os.Stat(kept); err != nil {
		t.Fatalf("the cycle that published nothing kept no cell: %v", err)
	}
	if worth, reason, err := world.CycleWorthRunning(); err != nil || worth {
		t.Fatalf("cells kept by a cycle that published nothing make a cycle worth running = %t (%s, %v), want false", worth, reason, err)
	}
}
