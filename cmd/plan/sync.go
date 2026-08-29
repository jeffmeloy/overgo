package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"path/filepath"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

const (
	compatibilityDocumentPath         = "docs/COMPATIBILITY.md"
	trainingCompatibilityDocumentPath = "docs/TRAINING_COMPATIBILITY.md"
)

func prepareMerge(root, source string, local plan.Plan, output io.Writer) error {
	canonicalRoot, err := gitauthority.RepositoryRoot(context.Background(), root)
	if err != nil {
		return err
	}
	root = canonicalRoot
	status, err := gitOutput(root, "status", "--porcelain=v1", "--untracked-files=all")
	if err != nil {
		return err
	}
	if strings.TrimSpace(string(status)) != "" {
		return errors.New("prepare-merge requires a clean worktree")
	}
	localRaw, err := gitOutput(root, "rev-parse", "--verify", "HEAD^{commit}")
	if err != nil {
		return err
	}
	localRevision := strings.TrimSpace(string(localRaw))
	sourceRaw, err := gitOutput(root, "rev-parse", "--verify", source+"^{commit}")
	if err != nil {
		return err
	}
	snapshot := strings.TrimSpace(string(sourceRaw))
	closureSnapshot, err := captureClosureEvidence(root, snapshot)
	if err != nil {
		return err
	}
	if closureSnapshot.cleanup != nil {
		defer closureSnapshot.cleanup()
	}
	contains := exec.Command("git", "--no-replace-objects", "merge-base", "--is-ancestor", snapshot, "HEAD")
	contains.Dir = root
	contains.Env = gitauthority.RepositoryEnvironment()
	if contains.Run() == nil {
		fmt.Fprintf(output, "prepare-merge: HEAD already contains %s at %s\n", source, snapshot)
		return nil
	}
	baseRef, err := uniqueMergeBase(root, localRevision, snapshot)
	if err != nil {
		return err
	}
	base, err := planAtRef(root, baseRef)
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
	if err := verifyProspectiveMergeAuthority(
		root, localRevision, snapshot, local, incoming, merged,
	); err != nil {
		return fmt.Errorf("prepare-merge completion authority: %w", err)
	}
	latestLocal, err := gitOutput(root, "rev-parse", "--verify", "HEAD^{commit}")
	if err != nil || strings.TrimSpace(string(latestLocal)) != localRevision {
		return fmt.Errorf("prepare-merge local HEAD moved from snapshot %s", localRevision)
	}
	_, mergeErr := commandOutput(root, "git", "merge", "--no-ff", "--no-commit", snapshot)
	postMergeHead, headErr := gitOutput(root, "rev-parse", "--verify", "HEAD^{commit}")
	if headErr != nil || strings.TrimSpace(string(postMergeHead)) != localRevision {
		return fmt.Errorf("prepare-merge local HEAD moved during merge from snapshot %s", localRevision)
	}
	keepMerge := false
	defer func() {
		if !keepMerge {
			abort := exec.Command("git", "--no-replace-objects", "merge", "--abort")
			abort.Dir = root
			abort.Env = gitauthority.RepositoryEnvironment()
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
	if err := verifySourceSnapshot(source, snapshot, latest, err); err != nil {
		return err
	}
	if _, err := gitOutput(root, "add", "--", plan.Path, "compatibility.json", compatibilityDocumentPath, trainingCompatibilityDocumentPath); err != nil {
		return err
	}
	if closureSnapshot.store != "" {
		result, err := commandOutput(root, "go", "run", "./cmd/closure-scan", "-import-store", closureSnapshot.store)
		if err != nil {
			return fmt.Errorf("prepare-merge closure evidence: %w", err)
		}
		fmt.Fprintf(output, "prepare-merge: closure evidence %s from %s@%d bound to git %s\n",
			strings.TrimSpace(string(result)), closureSnapshot.head, closureSnapshot.sequence, snapshot)
	} else {
		fmt.Fprintln(output, "prepare-merge: closure evidence unavailable (source snapshot has no local OvergoDB worktree)")
	}
	keepMerge = true
	fmt.Fprintf(output, "prepare-merge: %s@%s staged; finalize with cmd/gate -merge -plan %s/do\n", source, snapshot, mergeID)
	return nil
}

func uniqueMergeBase(root, localRevision, incomingRevision string) (string, error) {
	output, err := gitOutput(root, "merge-base", "--all", localRevision, incomingRevision)
	if err != nil {
		return "", err
	}
	bases := strings.Fields(string(output))
	if len(bases) != 1 {
		return "", fmt.Errorf("prepare-merge requires exactly one merge base, found %d", len(bases))
	}
	return bases[0], nil
}

func verifyProspectiveMergeAuthority(
	root, localRevision, incomingRevision string,
	local, incoming, merged plan.Plan,
) (returnErr error) {
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		return err
	}
	defer func() { returnErr = errors.Join(returnErr, store.Close()) }()
	localAuthority, err := plan.ResolveCompletionAuthority(
		context.Background(), root, localRevision, local, store,
	)
	if err != nil {
		return fmt.Errorf("local parent %.12s: %w", localRevision, err)
	}
	incomingAuthority, err := plan.ResolveCompletionAuthority(
		context.Background(), root, incomingRevision, incoming, store,
	)
	if err != nil {
		return fmt.Errorf("incoming parent %.12s is not proven by the target authority store: %w", incomingRevision, err)
	}
	return plan.VerifyProspectiveMergeAuthority(
		root, localRevision, incomingRevision,
		local, incoming, merged, localAuthority, incomingAuthority,
	)
}

func verifySourceSnapshot(source, snapshot string, latest []byte, resolveErr error) error {
	if resolveErr != nil || strings.TrimSpace(string(latest)) != snapshot {
		return fmt.Errorf("prepare-merge source %s moved from snapshot %s", source, snapshot)
	}
	return nil
}

func planAtRef(root, ref string) (plan.Plan, error) {
	data, err := gitOutput(root, "show", ref+":"+plan.Path)
	if err != nil {
		return plan.Plan{}, err
	}
	return plan.ParseHistorical(bytes.TrimSpace(data))
}

func gitOutput(root string, args ...string) ([]byte, error) {
	return commandOutput(root, "git", args...)
}

func commandOutput(root, name string, args ...string) ([]byte, error) {
	commandArgs := args
	if name == "git" {
		commandArgs = append([]string{"--no-replace-objects"}, args...)
	}
	command := exec.Command(name, commandArgs...)
	command.Dir = root
	if name == "git" {
		command.Env = gitauthority.RepositoryEnvironment()
	}
	output, err := command.CombinedOutput()
	if err != nil {
		return output, fmt.Errorf("%s %s: %w: %s", name, strings.Join(args, " "), err, strings.TrimSpace(string(output)))
	}
	return output, nil
}
