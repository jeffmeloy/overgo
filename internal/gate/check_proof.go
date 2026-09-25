package gate

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/codeprofile"
	"overgo/internal/plan"
	"overgo/internal/planverify"
)

// measuredProfiles is a candidate's code profile and its base's.
type measuredProfiles struct {
	candidate, base codeprofile.Profile
}

// profilePair measures the candidate and its base once per gate. The profile
// step reads both, and acceptance reads how much production code the
// candidate adds, so the two never disagree about what the landing is.
func (g *gateContext) profilePair() (codeprofile.Profile, codeprofile.Profile, error) {
	if g.profiles != nil {
		return g.profiles.candidate, g.profiles.base, nil
	}
	snapshot, err := g.sourceSnapshot()
	if err != nil {
		return codeprofile.Profile{}, codeprofile.Profile{}, err
	}
	candidate, err := g.sourceProfile(snapshot)
	if err != nil {
		return codeprofile.Profile{}, codeprofile.Profile{}, err
	}
	baseSource, err := g.baseSnapshot(snapshot)
	if err != nil {
		return codeprofile.Profile{}, codeprofile.Profile{}, err
	}
	base, err := g.sourceProfile(baseSource)
	if err != nil {
		return codeprofile.Profile{}, codeprofile.Profile{}, err
	}
	g.profiles = &measuredProfiles{candidate: candidate, base: base}
	return candidate, base, nil
}

// proveCheck holds a row's check to meaning something about the landing it
// accepts. The check is the one the row held at the base commit, so a landing
// cannot rewrite what it is judged by. And a landing that adds production
// code must carry a check that fails or is absent without it: one that
// already passes at the base commit proves nothing about the code added. A
// landing that removes or holds production code only has to pass, because
// what it has to show is that nothing broke.
func (g *gateContext) proveCheck(verify string) error {
	data, err := command(g.repo, "git", "show", "HEAD:"+plan.Path)
	if err != nil {
		return err
	}
	base, err := plan.Parse([]byte(data))
	if err != nil {
		return err
	}
	if err := checkUnchangedSinceBase(base, g.planRef, verify); err != nil {
		return err
	}
	if len(g.plannedGoFiles()) == 0 {
		return nil
	}
	candidate, baseProfile, err := g.profilePair()
	if err != nil {
		return err
	}
	growth := candidate.Runtime.Nodes + candidate.Automation.Nodes - baseProfile.Runtime.Nodes - baseProfile.Automation.Nodes
	if growth <= 0 {
		return nil
	}
	if covered, err := g.sourceProvenMergeGo(); err != nil {
		return err
	} else if covered {
		return nil
	}
	return requireProvenCheck(growth, g.runAtBase(verify))
}

// runAtBase runs a check in its own checkout of the base commit, apart from
// the candidate the gate is judging: that candidate's worktree and the state
// built on it stay as they are for the checks that follow. The error is the
// check's own outcome, or what kept it from running.
func (g *gateContext) runAtBase(verify string) (err error) {
	temporary, err := gateTemporary("base-proof-*")
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, os.RemoveAll(temporary)) }()
	worktree := filepath.Join(temporary, "base")
	if _, err := gitWriterCommand(g.repo, "worktree", "add", "--quiet", "--detach", worktree, "HEAD"); err != nil {
		return err
	}
	defer func() {
		_, cleanupErr := gitWriterCommand(g.repo, "worktree", "remove", "--force", worktree)
		err = errors.Join(err, cleanupErr)
	}()
	environment, err := g.sourceEnvironment()
	if err != nil {
		return err
	}
	_, err = planverify.Execute(context.Background(), worktree, verify, environment)
	return err
}

// checkUnchangedSinceBase holds a row that already existed at the base commit
// to the check it had there. A row the landing adds has no earlier check, and
// its proof is that the check fails or is absent without the change.
func checkUnchangedSinceBase(base plan.Plan, reference, verify string) error {
	itemID, stepID, _ := strings.Cut(reference, "/")
	for _, item := range base.Items {
		if item.ID != itemID {
			continue
		}
		for _, step := range item.Steps {
			if step.ID == stepID && step.Verify != verify {
				return fmt.Errorf("acceptance: %s changes its own check in the landing it is judged by; land the check change on its own first", reference)
			}
		}
	}
	return nil
}

// requireProvenCheck decides a growing landing's check from how that check
// ran at the base commit: a clean pass there proves nothing, a failure or an
// absent test is the proof, and anything else is an error the gate reports.
func requireProvenCheck(growth int, baseErr error) error {
	switch {
	case growth <= 0:
		return nil
	case baseErr == nil:
		return fmt.Errorf("acceptance: the row's check already passes at the base commit, so it proves nothing about the %d production nodes this landing adds; give the row a check that fails or is absent without the change", growth)
	case errors.Is(baseErr, planverify.ErrFailed), errors.Is(baseErr, planverify.ErrVacuous):
		return nil
	default:
		return baseErr
	}
}
