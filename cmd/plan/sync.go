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

const (
	compatibilityDocumentPath         = "docs/COMPATIBILITY.md"
	trainingCompatibilityDocumentPath = "docs/TRAINING_COMPATIBILITY.md"
)

func prepareMerge(root, source string, local plan.Plan, output io.Writer) error {
	status, err := gitOutput(root, "status", "--porcelain=v1", "--untracked-files=all")
	if err != nil {
		return err
	}
	if strings.TrimSpace(string(status)) != "" {
		return errors.New("prepare-merge requires a clean worktree")
	}
	sourceRaw, err := gitOutput(root, "rev-parse", "--verify", source+"^{commit}")
	if err != nil {
		return err
	}
	snapshot := strings.TrimSpace(string(sourceRaw))
	contains := exec.Command("git", "merge-base", "--is-ancestor", snapshot, "HEAD")
	contains.Dir = root
	if contains.Run() == nil {
		fmt.Fprintf(output, "prepare-merge: HEAD already contains %s at %s\n", source, snapshot)
		return nil
	}
	baseRef, err := gitOutput(root, "merge-base", "HEAD", snapshot)
	if err != nil {
		return err
	}
	base, err := planAtRef(root, strings.TrimSpace(string(baseRef)))
	if err != nil {
		return err
	}
	incoming, err := planAtRef(root, snapshot)
	if err != nil {
		return err
	}
	merged, err := plan.MergeOpenProjections(base, local, incoming)
	if err != nil {
		return err
	}
	mergeID := "merge-" + snapshot[:12]
	merged, err = insertItem(merged, mergeID, "Merge "+source+" at "+snapshot[:12], "", "go run ./cmd/compatibility -check")
	if err != nil {
		return err
	}
	merge := exec.Command("git", "merge", "--no-ff", "--no-commit", snapshot)
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
		if conflictErr != nil {
			return conflictErr
		}
		for _, path := range strings.Fields(string(conflicts)) {
			if path != plan.Path && path != compatibilityDocumentPath && path != trainingCompatibilityDocumentPath {
				return fmt.Errorf("prepare-merge source conflict; merge aborted: %s", strings.TrimSpace(string(mergeOutput)))
			}
		}
	}
	if err := plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), merged); err != nil {
		return err
	}
	refresh := exec.Command("go", "run", "./cmd/compatibility", "-refresh-identities")
	refresh.Dir = root
	if result, refreshErr := refresh.CombinedOutput(); refreshErr != nil {
		return fmt.Errorf("refresh merged compatibility: %w: %s", refreshErr, strings.TrimSpace(string(result)))
	}
	latest, err := gitOutput(root, "rev-parse", "--verify", source+"^{commit}")
	if err != nil || strings.TrimSpace(string(latest)) != snapshot {
		return fmt.Errorf("prepare-merge source %s moved from snapshot %s", source, snapshot)
	}
	if _, err := gitOutput(root, "add", "--", plan.Path, "compatibility.json", compatibilityDocumentPath, trainingCompatibilityDocumentPath); err != nil {
		return err
	}
	keepMerge = true
	fmt.Fprintf(output, "prepare-merge: %s@%s staged; finalize with cmd/gate -merge -plan %s/do\n", source, snapshot, mergeID)
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
