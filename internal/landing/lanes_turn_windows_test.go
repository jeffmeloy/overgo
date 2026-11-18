//go:build windows

package landing

import (
	"bufio"
	"errors"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
)

// TestTheLanesGiveTheirTurnToAReadyCommit holds a run of the cycle's lanes to taking turns with a commit (lanes-turns):
// a commit's tests that come to the test turn while a lane's package runs take it as the package ends, before the
// lane's next package, and the models lane runs its packages one by one, each alone with the lane's flags and tags.
// Beside the models lane as one run, a commit was refused for memory and its lanes waited the lane's hours.
func TestTheLanesGiveTheirTurnToAReadyCommit(t *testing.T) {
	// Serial: it names the test turn the lanes and the commits take.
	turn := testTurn
	testTurn = func() string { return "overgo-test-lanes-turns" }
	t.Cleanup(func() { testTurn = turn })
	commit := exec.Command(os.Args[0], "-test.run=^TestLaneTurnHolder$")
	commit.Env = append(os.Environ(), laneTurnHolderEnvironment+"="+testTurn())
	input := testcheck.Must(commit.StdinPipe())(t)
	output := testcheck.Must(commit.StdoutPipe())(t)
	var tookIt atomic.Bool
	ended := make(chan error, 1)
	steps := []func() error{
		func() error {
			// The commit comes to the turn while the lane's first package holds it.
			if err := commit.Start(); err != nil {
				return err
			}
			go func() {
				lines := bufio.NewScanner(output)
				for lines.Scan() {
					if lines.Text() == "taken" {
						tookIt.Store(true)
						_ = input.Close()
					}
				}
				ended <- commit.Wait()
			}()
			for {
				waited, err := processcontrol.TurnWaited()
				if err != nil || waited {
					return err
				}
				select {
				case err := <-ended:
					return errors.Join(errors.New("the commit ended before it waited for the turn"), err)
				default:
					runtime.Gosched()
				}
			}
		},
		func() error {
			if !tookIt.Load() {
				return errors.New("the lane's next package took the turn before the commit waiting for it")
			}
			return nil
		},
	}
	testcheck.NoError(t, HoldTestTurnEach("a fixture lane", func(work func() error) error { return work() }, steps))
	testcheck.NoError(t, <-ended)

	index := slices.IndexFunc(CycleLanes(), func(candidate CycleLane) bool { return candidate.Name == "models" })
	if index < 0 {
		t.Fatal("no cycle lane is named models")
	}
	models := CycleLanes()[index]
	if len(models.Each) != len(modelPackages) {
		t.Fatalf("the models lane runs %d packages one by one; want its %d", len(models.Each), len(modelPackages))
	}
	for position, arguments := range models.Each {
		if arguments[0] != "test" || !slices.Contains(arguments, CycleTags) || !slices.Contains(arguments, "-count=1") || arguments[len(arguments)-1] != modelPackages[position] ||
			slices.ContainsFunc(modelPackages, func(other string) bool { return other != modelPackages[position] && slices.Contains(arguments, other) }) {
			t.Errorf("the models lane runs %s as %q; want it alone with the lane's flags and tags", modelPackages[position], arguments)
		}
	}
}

// TestALanesPackageWaitingForTheCardHoldsNoTestTurn holds a lanes package on the card to taking the card's turn before
// the test turn (lanes-card-order): while it waits for the card, a commit coming to the test turn takes it at once,
// and the package runs once it holds both. A package that took the test turn first and then waited 20 min 47 s for the
// card held a commit's tests, which do not use the card, the whole of it.
func TestALanesPackageWaitingForTheCardHoldsNoTestTurn(t *testing.T) {
	// Serial: it names the test turn the lanes and the commits take.
	turn := testTurn
	testTurn = func() string { return "overgo-test-lanes-card-order" }
	t.Cleanup(func() { testTurn = turn })
	waitingForTheCard := func(work func() error) error {
		commit := exec.Command(os.Args[0], "-test.run=^TestLaneTurnHolder$")
		// A commit's tests are no process of the lanes', so they hold none of the turns the lanes hold (TurnEnvironment).
		commit.Env = append(slices.DeleteFunc(os.Environ(), func(setting string) bool {
			return strings.HasPrefix(setting, processcontrol.TurnEnvironment+"=")
		}), laneTurnHolderEnvironment+"="+testTurn())
		input := testcheck.Must(commit.StdinPipe())(t)
		output := testcheck.Must(commit.StdoutPipe())(t)
		testcheck.NoError(t, commit.Start())
		said := bufio.NewScanner(output)
		taken := said.Scan() && said.Text() == "taken"
		if !taken {
			t.Errorf("a commit coming to the test turn while a lanes package waited for the card said %q; want the turn taken at once", said.Text())
			testcheck.NoError(t, commit.Process.Kill())
		}
		closed, ended := input.Close(), commit.Wait()
		if taken {
			testcheck.NoError(t, errors.Join(closed, ended))
		}
		return work()
	}
	ran := false
	testcheck.NoError(t, HoldTestTurnEach("a fixture lane", waitingForTheCard, []func() error{func() error { ran = true; return nil }}))
	if !ran {
		t.Error("the package did not run once it held the card and the test turn")
	}
}
