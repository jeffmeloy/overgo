package main

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/gitauthority"
)

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
	listed, err := gitauthority.Query(ctx, root, "ls-remote", remote, "refs/heads/main")
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
	if err := gitWrite(root, "push", remote, commit+":refs/heads/main"); err != nil {
		return "", err
	}
	return commit, nil
}
