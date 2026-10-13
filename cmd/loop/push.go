package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/quietmachine"
)

// pushedBranch is the branch of the remote that a push publishes to. The local branch of that
// name is the staging copy of it, so a listing of leftovers holds it however far the trunk is past it.
const pushedBranch = "refs/heads/main"

// pushNamed publishes the commit the owner names to the remote's main, which only the owner
// does, only when asked: it refuses from a worker's environment and without the explicit
// confirmation, and only fast-forwards the remote, never forcing it. The loop itself never calls
// it; it is the command the owner runs to hand GitHub a commit.
func pushNamed(root, name, remote string, confirmed, worker bool) (string, error) {
	switch {
	case worker:
		return "", errors.New("the loop and its workers never push: the owner runs -push")
	case !confirmed:
		return "", errors.New("pushing publishes " + name + " to " + remote + "/main: pass -confirm-push to say it is wanted")
	}
	ctx := context.Background()
	resolved, err := gitauthority.Query(ctx, root, "rev-parse", "--verify", "--end-of-options", name+"^{commit}")
	if err != nil {
		return "", fmt.Errorf("%s names no commit in this repository: %w", name, err)
	}
	commit := strings.TrimSpace(string(resolved))
	listed, err := gitauthority.Query(ctx, root, "ls-remote", remote, pushedBranch)
	if err != nil {
		return "", err
	}
	tip, _, _ := strings.Cut(strings.TrimSpace(string(listed)), "\t")
	if tip == commit {
		return commit, nil
	}
	if tip != "" {
		if _, err := gitauthority.Query(ctx, root, "merge-base", "--is-ancestor", tip, commit); err != nil {
			return "", fmt.Errorf("%s/main holds %.12s, which %.12s does not contain: a push only fast-forwards: %w", remote, tip, commit, err)
		}
	}
	push := func() error { return gitWrite(root, "push", remote, commit+":"+pushedBranch) }
	if err := quietmachine.Workload("push", os.Stderr, push)(); err != nil {
		return "", err
	}
	return commit, nil
}
