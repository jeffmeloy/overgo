package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gitauthority"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestALanesRunTakesFirstWhatItsCommitChanges holds a run of the lanes to taking first what the commits it runs say
// matters, read from them by the tool (lanes-first):
//   - over moss-attention/do (97714132e), which landed at 00:46 on 2026-11-19 and whose model's checkpoints failed at
//     13:57, two hours into a run that reached them in its own order, internal/latentimage's checkpoint tests rank
//     before its other tests, its files among those the commit changed
//   - over moss-elementwise/do (9acbc157a), the tests its check named, in the packages its check ran, rank first, in
//     the check's order
//   - a lane's steps run in that order: a package of three tests runs the one a landed check names, then its
//     checkpoint, then the rest
func TestALanesRunTakesFirstWhatItsCommitChanges(t *testing.T) {
	t.Parallel()
	repo := testutil.RepoRoot(t)
	attention := testcheck.Must(readLanesFirst(t.Context(), repo, "97714132e^", "97714132e"))(t)
	const latentimage = "./internal/latentimage"
	for _, checkpoint := range []string{"TestMOSSSamplingCheckpoint", "TestMOSSGenerationCheckpoint"} {
		if rank, other := attention.rank(latentimage, checkpoint), attention.rank(latentimage, "TestDepthControlMeetsItsSectionFloor"); rank >= other {
			t.Errorf("over moss-attention/do, %s ranks %d and a test no check names %d; want the changed package's checkpoints first", checkpoint, rank, other)
		}
	}
	elementwise := testcheck.Must(readLanesFirst(t.Context(), repo, "9acbc157a^", "9acbc157a"))(t)
	checked := []string{"TestAMOSSBlocksQueryAndKeyProductsNormAndRotateInTheirEpilogue", "TestANormTwoRotationsReadIsComputedForBoth", "TestTheMOSSPassFusesWithoutChangingABit"}
	var named []string
	for _, check := range elementwise.checks {
		named = append(named, check.name)
		if !slices.Equal(check.packages, []string{"./internal/cuda/executor", latentimage}) {
			t.Errorf("moss-elementwise/do's check runs %s in %v; want the packages its check named", check.name, check.packages)
		}
	}
	if !slices.Equal(named, checked) {
		t.Fatalf("over moss-elementwise/do the landed checks name %v; want %v", named, checked)
	}
	if first, checkpoint := elementwise.rank("./internal/cuda/executor", checked[0]), elementwise.rank("./internal/cuda/executor", "TestAnInt8ConvRotProductIsComfyKitchens"); first != 0 || checkpoint <= first {
		t.Errorf("over moss-elementwise/do its check's first test ranks %d and another %d; want it first", first, checkpoint)
	}

	root := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "go.mod"), []byte("module fixture\n\ngo 1.26\n"), 0o644))
	testcheck.NoError(t, os.MkdirAll(filepath.Join(root, "p"), 0o755))
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "p", "p_test.go"), []byte(
		"package p\n\nimport \"testing\"\n\nfunc TestAlpha(t *testing.T) {}\n\nfunc TestBetaCheckpoint(t *testing.T) {}\n\nfunc TestGamma(t *testing.T) {}\n"), 0o644))
	first := lanesFirst{checks: []checkedTest{{name: "TestGamma", packages: []string{"./p"}}}, packages: map[string]bool{"./p": true}}
	var order []string
	for _, step := range testSteps(t.Context(), root, [][]string{{"test", "-count=1", "./p"}}, nil, first) {
		order = append(order, strings.Trim(step.arguments[slices.Index(step.arguments, "-run")+1], "^$"))
	}
	if !slices.Equal(order, []string{"TestGamma", "TestBetaCheckpoint", "TestAlpha"}) {
		t.Errorf("a package of three tests runs them as %v; want the one a landed check names, then its checkpoint, then the rest", order)
	}
}

// TestAFirstListedTestPassedSinceItsRowIsNotTakenFirst holds a run of the lanes to repeating less (lanes-first-kept):
// over moss-elementwise/do (9acbc157a), whose check named three tests in two packages, each test carries the commit
// that landed its row; one with a pass kept in both packages at a commit since that one runs in its package's own
// order, while one passed in a single package, and one passed only at a commit before the row's, stay first, in the
// check's order. The models lane, unpassed whole since 2026-11-16, had taken 89 tests first at every close, two hours
// and more, though the run before had kept passes of them (the owner's review of 2026-11-21).
func TestAFirstListedTestPassedSinceItsRowIsNotTakenFirst(t *testing.T) {
	t.Parallel()
	repo := testutil.RepoRoot(t)
	first := testcheck.Must(readLanesFirst(t.Context(), repo, "9acbc157a^", "9acbc157a"))(t)
	landed := strings.TrimSpace(string(testcheck.Must(gitauthority.Query(t.Context(), repo, "rev-parse", "9acbc157a"))(t)))
	if len(first.checks) != 3 {
		t.Fatalf("moss-elementwise/do's check names %d tests; want 3", len(first.checks))
	}
	for _, check := range first.checks {
		if check.commit != landed {
			t.Errorf("%s is read as named at %q; want the commit that landed its row, %s", check.name, check.commit, landed)
		}
	}
	const executor, latentimage = "overgo/internal/cuda/executor", "overgo/internal/latentimage"
	names := []string{first.checks[0].name, first.checks[1].name, first.checks[2].name}
	passes := map[string]plan.TestPass{
		plan.PassedTest(executor, names[0]): {Commit: "later"}, plan.PassedTest(latentimage, names[0]): {Commit: "later"},
		plan.PassedTest(executor, names[1]): {Commit: "later"},
		plan.PassedTest(executor, names[2]): {Commit: "earlier"}, plan.PassedTest(latentimage, names[2]): {Commit: "earlier"},
	}
	since := func(earlier, later string) bool { return earlier == landed && later == "later" }
	if left := first.passedSince(passes, since); left != 1 {
		t.Errorf("the first list left out %d tests; want the one passed since in both its packages", left)
	}
	var kept []string
	for _, check := range first.checks {
		kept = append(kept, check.name)
	}
	if !slices.Equal(kept, names[1:]) {
		t.Errorf("the first list keeps %v; want %v, the test passed since its row named it left to its package's order", kept, names[1:])
	}
	if rank := first.rank("./internal/cuda/executor", names[0]); rank < len(first.checks) {
		t.Errorf("%s ranks %d; want it among its package's other tests", names[0], rank)
	}
}
