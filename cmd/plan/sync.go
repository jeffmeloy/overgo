package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"strings"

	"overgo/internal/plan"
)

func syncMaster(root string, local plan.Plan, output io.Writer) error {
	status, err := gitOutput(root, "status", "--porcelain=v1", "--untracked-files=all")
	if err != nil {
		return err
	}
	if strings.TrimSpace(string(status)) != "" {
		return errors.New("sync-master requires a clean worktree")
	}
	contains := exec.Command("git", "merge-base", "--is-ancestor", "master", "HEAD")
	contains.Dir = root
	if contains.Run() == nil {
		fmt.Fprintln(output, "sync-master: HEAD already contains master")
		return nil
	}
	baseRef, err := gitOutput(root, "merge-base", "HEAD", "master")
	if err != nil {
		return err
	}
	base, err := planAtRef(root, strings.TrimSpace(string(baseRef)))
	if err != nil {
		return err
	}
	upstream, err := planAtRef(root, "master")
	if err != nil {
		return err
	}
	merged, err := plan.MergeOpenProjections(base, local, upstream)
	if err != nil {
		return err
	}
	merge := exec.Command("git", "merge", "--no-ff", "--no-commit", "master")
	merge.Dir = root
	mergeOutput, mergeErr := merge.CombinedOutput()
	keepMerge := false
	defer func() {
		if !keepMerge {
			abort := exec.Command("git", "merge", "--abort")
			abort.Dir = root
			_ = abort.Run()
		}
	}()
	if mergeErr != nil {
		conflicts, conflictErr := gitOutput(root, "diff", "--name-only", "--diff-filter=U")
		if conflictErr != nil || strings.TrimSpace(string(conflicts)) != plan.Path {
			return fmt.Errorf("sync-master source conflict; merge aborted: %s", strings.TrimSpace(string(mergeOutput)))
		}
	}
	if err := plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), merged); err != nil {
		return err
	}
	if _, err := gitOutput(root, "add", "--", plan.Path); err != nil {
		return err
	}
	keepMerge = true
	fmt.Fprintln(output, "sync-master: merge prepared; finalize through cmd/gate -merge with the active plan step")
	return nil
}

func planAtRef(root, ref string) (plan.Plan, error) {
	data, err := gitOutput(root, "show", ref+":"+plan.Path)
	if err != nil {
		return plan.Plan{}, err
	}
	return plan.Parse(bytes.TrimSpace(data))
}

func gitOutput(root string, args ...string) ([]byte, error) {
	command := exec.Command("git", args...)
	command.Dir = root
	output, err := command.Output()
	if err != nil {
		return nil, fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	return output, nil
}
