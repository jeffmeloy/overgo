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
	merged, err := plan.MergeDocuments(base, local, incoming)
	if err != nil {
		return err
	}
	mergeID := "merge-" + snapshot[:12]
	merged, err = insertItem(merged, mergeID, "Merge "+source+" at "+snapshot[:12], "", "go run ./cmd/compatibility -check")
	if err != nil {
		return err
	}
	_, mergeErr := commandOutput(root, "git", "merge", "--no-ff", "--no-commit", snapshot)
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
				return fmt.Errorf("prepare-merge source conflict; merge aborted: %w", mergeErr)
			}
		}
	}
	if err := plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), merged); err != nil {
		return err
	}
	if _, err := commandOutput(root, "go", "run", "./cmd/compatibility", "-refresh-identities"); err != nil {
		return fmt.Errorf("refresh merged compatibility: %w", err)
	}
	latest, err := gitOutput(root, "rev-parse", "--verify", source+"^{commit}")
	if err != nil || strings.TrimSpace(string(latest)) != snapshot {
		return fmt.Errorf("prepare-merge source %s moved from snapshot %s", source, snapshot)
	}
	if _, err := gitOutput(root, "add", "--", plan.Path, "compatibility.json", compatibilityDocumentPath, trainingCompatibilityDocumentPath); err != nil {
		return err
	}
	closureStore, err := sourceRepoDB(root, snapshot)
	if err != nil {
		return err
	}
	if closureStore != "" {
		result, err := commandOutput(root, "go", "run", "./cmd/closure-scan", "-import-store", closureStore)
		if err != nil {
			return fmt.Errorf("prepare-merge closure evidence: %w", err)
		}
		fmt.Fprintf(output, "prepare-merge: closure evidence %s\n", strings.TrimSpace(string(result)))
	} else {
		fmt.Fprintln(output, "prepare-merge: closure evidence unavailable (source snapshot has no local RepoDB worktree)")
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
	return commandOutput(root, "git", args...)
}

func commandOutput(root, name string, args ...string) ([]byte, error) {
	command := exec.Command(name, args...)
	command.Dir = root
	output, err := command.CombinedOutput()
	if err != nil {
		return output, fmt.Errorf("%s %s: %w: %s", name, strings.Join(args, " "), err, strings.TrimSpace(string(output)))
	}
	return output, nil
}
