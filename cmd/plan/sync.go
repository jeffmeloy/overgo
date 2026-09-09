package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

const (
	compatibilityDocumentPath         = "docs/COMPATIBILITY.md"
	trainingCompatibilityDocumentPath = "docs/TRAINING_COMPATIBILITY.md"
	apiManifestDocumentPath           = "docs/API_MANIFEST.md"
	apiManifestJSONPath               = "docs/api_manifest.json"
	modernGoBaselinePath              = "docs/modern_go_baseline.json"
	modernGoCensusPath                = "docs/modern_go_census.json"
)

func prepareMerge(root, source string, output io.Writer) error {
	return prepareMergeWithProjection(root, source, plan.MergeProjectionSemanticUnion, output)
}

func prepareMergeWithProjection(
	root, source string,
	projection plan.MergeProjection,
	output io.Writer,
) error {
	parsedProjection, err := plan.ParseMergeProjection(string(projection))
	if err != nil {
		return fmt.Errorf("prepare-merge projection: %w", err)
	}
	if parsedProjection != projection {
		return errors.New("prepare-merge projection is not canonical")
	}
	canonicalRoot, err := gitauthority.RepositoryRoot(context.Background(), root)
	if err != nil {
		return err
	}
	root = canonicalRoot
	return withPlanMutation(root, false, func(local plan.Plan) error {
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
		closureSnapshot, err := captureClosureEvidence(root, source, snapshot)
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
		liveSourceStore := ""
		if projection == plan.MergeProjectionFirstParentTarget {
			liveSourceStore, err = firstParentTargetSourceStore(
				context.Background(), root, snapshot, closureSnapshot.source,
			)
			if err != nil {
				return err
			}
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
		var localAuthority, incomingAuthority plan.CompletionAuthority
		if projection == plan.MergeProjectionSemanticUnion {
			localAuthority, incomingAuthority, err = resolveMergeAuthorities(
				root, localRevision, snapshot, local, incoming,
			)
			if err != nil {
				return fmt.Errorf("prepare-merge completion authority: %w", err)
			}
		}
		merged, err := projectMergePlan(base, local, incoming, localAuthority, incomingAuthority, projection)
		if err != nil {
			return err
		}
		mergeID := "merge-" + snapshot[:12]
		merged, err = insertItem(merged, mergeID, "Merge "+source+" at "+snapshot[:12], "", "go run ./cmd/compatibility -check")
		if err != nil {
			return err
		}
		if projection == plan.MergeProjectionFirstParentTarget {
			if err := verifyFirstParentTargetMergeSources(
				context.Background(), root, localRevision, snapshot, local, incoming, merged,
				closureSnapshot.store,
			); err != nil {
				return fmt.Errorf("prepare-merge completion authority: %w", err)
			}
		} else if err := plan.VerifyProspectiveMergeAuthorityWithProjection(
			root, localRevision, snapshot, local, incoming, merged, localAuthority, incomingAuthority,
			projection,
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
			for path := range strings.FieldsSeq(string(conflicts)) {
				if !mergeOwnedDocument(path) {
					return fmt.Errorf("prepare-merge source conflict; merge aborted: %w", mergeErr)
				}
			}
		}
		if err := plan.Save(filepath.Join(root, filepath.FromSlash(plan.Path)), merged); err != nil {
			return err
		}
		if err := regenerateMergeOwnedDocuments(root, localRevision); err != nil {
			return err
		}
		if _, err := commandOutput(root, "go", "run", "./cmd/compatibility", "-refresh-identities"); err != nil {
			return fmt.Errorf("refresh merged compatibility: %w", err)
		}
		latest, err := gitOutput(root, "rev-parse", "--verify", source+"^{commit}")
		if err := verifySourceSnapshot(source, snapshot, latest, err); err != nil {
			return err
		}
		if _, err := gitOutput(root, "add", "--",
			plan.Path,
			"compatibility.json",
			compatibilityDocumentPath,
			trainingCompatibilityDocumentPath,
			apiManifestDocumentPath,
			apiManifestJSONPath,
			modernGoBaselinePath,
			modernGoCensusPath,
		); err != nil {
			return err
		}
		if _, err := gitOutput(root, "update-index", "--clear-resolve-undo"); err != nil {
			return fmt.Errorf("clear resolved merge metadata: %w", err)
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
		projectionArguments, err := mergeFinalizeProjectionArguments(projection, liveSourceStore)
		if err != nil {
			return err
		}
		fmt.Fprintf(
			output,
			"prepare-merge: %s@%s staged; finalize with cmd/gate -merge%s -plan %s/do\n",
			source, snapshot, projectionArguments, mergeID,
		)
		return nil
	})
}

func mergeOwnedDocument(path string) bool {
	switch path {
	case plan.Path,
		compatibilityDocumentPath,
		trainingCompatibilityDocumentPath,
		apiManifestDocumentPath,
		apiManifestJSONPath,
		modernGoBaselinePath,
		modernGoCensusPath:
		return true
	default:
		return false
	}
}

func regenerateMergeOwnedDocuments(root, localRevision string) error {
	baseline, err := gitOutput(root, "show", localRevision+":"+modernGoBaselinePath)
	if err != nil {
		return fmt.Errorf("restore target modern-Go baseline: %w", err)
	}
	if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(modernGoBaselinePath)), baseline, clioptions.OutputFileMode); err != nil {
		return fmt.Errorf("restore target modern-Go baseline: %w", err)
	}
	commands := []struct {
		label string
		args  []string
	}{
		{"lower merged modern-Go baseline", []string{"run", "./cmd/modern-census", "-lower-baseline"}},
		{"publish merged modern-Go census", []string{"run", "./cmd/modern-census", "-publish-census"}},
		{"refresh merged API manifest", []string{"run", "./cmd/api-manifest", "-update"}},
	}
	for _, command := range commands {
		if _, err := commandOutput(root, "go", command.args...); err != nil {
			return fmt.Errorf("%s: %w", command.label, err)
		}
	}
	return nil
}

func projectMergePlan(
	base, local, incoming plan.Plan,
	localAuthority, incomingAuthority plan.CompletionAuthority,
	projection plan.MergeProjection,
) (plan.Plan, error) {
	if projection == plan.MergeProjectionFirstParentTarget {
		return local, nil
	}
	return plan.MergeDocumentsWithCompletion(base, local, incoming, localAuthority, incomingAuthority)
}

func uniqueMergeBase(root, localRevision, incomingRevision string) (string, error) {
	base, err := plan.CompletionMergeBase(context.Background(), root, localRevision, incomingRevision)
	if err != nil {
		return "", fmt.Errorf("prepare-merge: %w", err)
	}
	return base, nil
}

func resolveMergeAuthorities(
	root, localRevision, incomingRevision string,
	local, incoming plan.Plan,
) (localAuthority, incomingAuthority plan.CompletionAuthority, returnErr error) {
	store, err := overgodb.OpenReadOnly(filepath.Join(root, "overgodb-store"))
	if err != nil {
		return plan.CompletionAuthority{}, plan.CompletionAuthority{}, err
	}
	defer func() { returnErr = errors.Join(returnErr, store.Close()) }()
	localAuthority, err = plan.ResolveCompletionAuthority(
		context.Background(), root, localRevision, local, store,
	)
	if err != nil {
		return plan.CompletionAuthority{}, plan.CompletionAuthority{}, fmt.Errorf("local parent %.12s: %w", localRevision, err)
	}
	incomingAuthority, err = plan.ResolveCompletionAuthority(
		context.Background(), root, incomingRevision, incoming, store,
	)
	if err != nil {
		return plan.CompletionAuthority{}, plan.CompletionAuthority{}, fmt.Errorf(
			"incoming parent %.12s is not proven by the target authority store: %w", incomingRevision, err,
		)
	}
	return localAuthority, incomingAuthority, nil
}

func verifyFirstParentTargetMergeSources(
	ctx context.Context,
	root, localRevision, incomingRevision string,
	local, incoming, merged plan.Plan,
	sourceSnapshotPath string,
) (returnErr error) {
	if strings.TrimSpace(sourceSnapshotPath) == "" {
		return errors.New("first-parent-target requires a replay-verified source store snapshot")
	}
	targetStore, err := overgodb.OpenReadOnly(filepath.Join(root, gitauthority.CanonicalOvergoDBDirectory))
	if err != nil {
		return err
	}
	defer func() { returnErr = errors.Join(returnErr, targetStore.Close()) }()
	sourceStore, err := overgodb.OpenReadOnly(sourceSnapshotPath)
	if err != nil {
		return err
	}
	defer func() { returnErr = errors.Join(returnErr, sourceStore.Close()) }()

	localAuthority, err := plan.ResolveCompletionAuthority(ctx, root, localRevision, local, targetStore)
	if err != nil {
		return fmt.Errorf("local parent %.12s: %w", localRevision, err)
	}
	incomingAuthority, err := plan.ResolveCompletionAuthority(ctx, root, incomingRevision, incoming, sourceStore)
	if err != nil {
		return fmt.Errorf("incoming parent %.12s is not proven by its source authority store: %w", incomingRevision, err)
	}
	_, err = plan.VerifyFirstParentTargetMergeSources(
		ctx, root, localRevision, incomingRevision, local, incoming, merged,
		localAuthority, incomingAuthority, targetStore, sourceStore,
	)
	return err
}

func firstParentTargetSourceStore(
	ctx context.Context,
	root, incomingRevision, sourceStorePath string,
) (string, error) {
	if strings.TrimSpace(sourceStorePath) == "" {
		return "", errors.New("prepare-merge: first-parent-target requires a registered source worktree OvergoDB store")
	}
	store, err := gitauthority.RequireRegisteredWorktreeStore(ctx, root, sourceStorePath, incomingRevision)
	if err != nil {
		return "", fmt.Errorf("prepare-merge: first-parent-target source store: %w", err)
	}
	return store, nil
}

func mergeFinalizeProjectionArguments(projection plan.MergeProjection, sourceStore string) (string, error) {
	switch projection {
	case plan.MergeProjectionSemanticUnion:
		return "", nil
	case plan.MergeProjectionFirstParentTarget:
		if strings.TrimSpace(sourceStore) == "" {
			return "", errors.New("prepare-merge: first-parent-target finalize command requires its live source store")
		}
		if strings.ContainsAny(sourceStore, "\"\r\n") {
			return "", errors.New("prepare-merge: source store path cannot be rendered safely")
		}
		return " -plan-projection " + string(projection) + ` -merge-source-store "` + sourceStore + `"`, nil
	default:
		return "", errors.New("prepare-merge: invalid merge projection")
	}
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
