package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/scorecard"
)

// pushHeld publishes the commit the newest held scorecard names to the remote's main, which
// only the owner does, only when asked: it refuses from a worker's environment and without
// the explicit confirmation, and only fast-forwards the remote, never forcing it. The loop
// itself never calls it; it is the command the owner runs to hand GitHub what the cycles held.
func pushHeld(root, records, remote string, confirmed, worker bool) (string, error) {
	switch {
	case worker:
		return "", errors.New("the loop and its workers never push: the owner runs -push-held")
	case !confirmed:
		return "", errors.New("pushing publishes the held commit to " + remote + "/main: pass -confirm-push to say it is wanted")
	}
	all, err := scorecard.Records(records)
	if err != nil {
		return "", err
	}
	var commit string
	for _, record := range slices.Backward(all) {
		if record.Held {
			commit = record.Commit
			break
		}
	}
	if commit == "" {
		return "", errors.New("no scorecard is recorded as held, so there is nothing to push")
	}
	ctx := context.Background()
	if _, err := gitauthority.Query(ctx, root, "cat-file", "-e", commit+"^{commit}"); err != nil {
		return "", fmt.Errorf("the held commit %.12s is not in this repository: %w", commit, err)
	}
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
