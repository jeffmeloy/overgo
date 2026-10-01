package main

import (
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/loop"
	"overgo/internal/plan"
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
	cells[metric.Quality] = scorecard.Cell{Value: accuracy, Label: "accuracy, store/mmlu (raw-completion)", Source: source, Direction: runrecord.DirectionMaximize,
		Contract: scorecard.Contract{Metric: "accuracy", Origin: "evaluation", Suite: "store/mmlu", Protocol: "raw-completion"}}
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
	var keys []string
	for _, item := range document.Items {
		keys = append(keys, item.ID)
		if len(item.Steps) != 1 || item.Steps[0].Verify != "go run ./cmd/validate -scorecard -gap "+item.ID || !strings.Contains(item.Steps[0].Rationale, published.Name) {
			t.Fatalf("row %+v is not verified by its gap closing on the store's scorecard", item)
		}
		if item.Owner != document.Lane {
			t.Fatalf("row %s is owned by %q, want the plan's lane %q", item.ID, item.Owner, document.Lane)
		}
	}
	want := []string{
		"scorecard-failing-inference",
		"scorecard-regressed-inference",
		"scorecard-unmeasured-image-gen",
		"scorecard-unmeasured-inference",
	}
	if !slices.Equal(keys, want) {
		t.Fatalf("proposed rows %q, want %q", keys, want)
	}
	if rationale := document.Items[3].Steps[0].Rationale; !strings.Contains(rationale, "- alpha load-time: "+unmeasured.Reason) || !strings.Contains(rationale, "- beta load-time: "+unmeasured.Reason) {
		t.Fatalf("the inference load row does not list both models with their reason:\n%s", rationale)
	}
	if !strings.Contains(document.Items[1].Steps[0].Rationale, "- alpha held-out-quality: 0.5 to 0.4") {
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

// TestNoScorecardOfMainPromotesNothing publishes two scorecards of a commit
// main does not hold, the second better than the first. Neither measured
// main, so there is no baseline: main stays and the outcome says why, rather
// than judging the commit against its own earlier record.
func TestNoScorecardOfMainPromotesNothing(t *testing.T) {
	t.Parallel()
	repository := t.TempDir()
	gitFixture(t, repository, "init", "-q", "-b", "master")
	gitFixture(t, repository, "commit", "-q", "--allow-empty", "-m", "main")
	gitFixture(t, repository, "branch", "main")
	held := gitFixture(t, repository, "rev-parse", "main")
	gitFixture(t, repository, "commit", "-q", "--allow-empty", "-m", "measured")
	measured := gitFixture(t, repository, "rev-parse", "HEAD")

	source := testutil.ArtifactID(t, artifact.KindEvidence, "loop-no-main-baseline")
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
	improver := improvement{root: repository, plan: emptyPlan(t), records: records, measure: func() error { return publish(0.6) }}
	var outcome loop.Outcome
	if _, err := loop.Improve(improver, &outcome); err != nil {
		t.Fatal(err)
	}
	if outcome.Promoted != "" || !strings.Contains(outcome.Withheld, "no earlier scorecard measured the commit") || gitFixture(t, repository, "rev-parse", "main") != held {
		t.Fatalf("promoted %q, withheld %q, main %s; want main unmoved at %s and the missing baseline stated", outcome.Promoted, outcome.Withheld, gitFixture(t, repository, "rev-parse", "main"), held)
	}
}

// planWorld drives the loop over a plan file: its worker lands the
// dispatched row the way the gate does, by removing it, and records which
// commit the promoted branch held when it ran.
type planWorld struct {
	improvement
	promoted func() string
	heldAt   []string
	parked   []string
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
	w.heldAt = append(w.heldAt, w.promoted())
	document, err := plan.Load(w.plan)
	if err == nil {
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

// TestPromotionWaitsForAHoldingScorecard runs the loop over a real
// repository whose main holds the last promoted commit. The first scorecard
// of the new commit has a lower held-out accuracy: main stays, and the fall
// becomes the row the worker lands. The next scorecard holds, and main
// fast-forwards to the measured commit, which the remote never receives.
// A main holding a commit the measured one lacks is refused, not rewound.
func TestPromotionWaitsForAHoldingScorecard(t *testing.T) {
	t.Parallel()
	repository, remote := t.TempDir(), t.TempDir()
	git := func(directory string, arguments ...string) string { return gitFixture(t, directory, arguments...) }
	git(remote, "init", "-q", "--bare")
	git(repository, "init", "-q", "-b", "master")
	git(repository, "commit", "-q", "--allow-empty", "-m", "promoted")
	promotedCommit := git(repository, "rev-parse", "HEAD")
	git(repository, "branch", "main")
	git(repository, "remote", "add", "origin", remote)
	git(repository, "push", "-q", "origin", "main")
	git(repository, "commit", "-q", "--allow-empty", "-m", "measured")
	measuredCommit := git(repository, "rev-parse", "HEAD")
	promoted := func() string { return git(repository, "rev-parse", "main") }

	source := testutil.ArtifactID(t, artifact.KindEvidence, "loop-promotion")
	records := filepath.Join(t.TempDir(), "scorecards")
	published := time.Date(2026, time.September, 29, 12, 0, 0, 0, time.UTC)
	scripted := []scorecard.Record{
		{Commit: promotedCommit, Rows: []scorecard.Row{scoredRow(t, source, "alpha", recipe.TaskInference, 0.5, 0, scorecard.Cell{Value: 1, Source: source})}},
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
	world := &planWorld{promoted: promoted, improvement: improvement{root: repository, plan: emptyPlan(t), records: records, measure: publish}}
	outcome, err := loop.Run(world, loop.Config{MaxAttemptsPerStep: 1, MaxInvocations: 2})
	if err != nil || outcome.Reason != loop.ReasonPlanComplete || len(world.parked) != 0 {
		t.Fatalf("outcome=%+v err=%v parked=%v", outcome, err, world.parked)
	}
	if !slices.Equal(world.heldAt, []string{promotedCommit}) {
		t.Fatalf("the worker ran %d times with main at %v; want once, on the fall, with main unmoved at %s", len(world.heldAt), world.heldAt, promotedCommit)
	}
	if promoted() != measuredCommit || outcome.Promoted != measuredCommit {
		t.Fatalf("main = %s, promoted %q; want the measured commit %s once its scorecard held", promoted(), outcome.Promoted, measuredCommit)
	}
	if pushed := git(remote, "rev-parse", "main"); pushed != promotedCommit {
		t.Fatalf("the remote's main moved to %s; promotion never pushes", pushed)
	}
	git(repository, "checkout", "-q", "--orphan", "elsewhere")
	git(repository, "commit", "-q", "--allow-empty", "-m", "unrelated")
	unrelated := git(repository, "rev-parse", "HEAD")
	git(repository, "update-ref", "refs/heads/main", unrelated)
	if err := world.Promote(measuredCommit); err == nil || promoted() != unrelated {
		t.Fatalf("promoting over a main the measured commit does not contain = %v, main %s; want a refusal", err, promoted())
	}
}
