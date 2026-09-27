package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/gitauthority"
	"overgo/internal/lanestore"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

const (
	compatibilityDocumentPath         = "docs/COMPATIBILITY.md"
	trainingCompatibilityDocumentPath = "docs/TRAINING_COMPATIBILITY.md"
	apiManifestJSONPath               = "docs/api_manifest.json"
	modernGoBaselinePath              = "docs/modern_go_baseline.json"
	harnessSurfaceBaselinePath        = "docs/harness_surface_baseline.json"
)

func prepareMerge(root, source string, output io.Writer) error {
	return prepareMergeWithProjection(root, source, plan.MergeProjectionSemanticUnion, "", output)
}

func prepareMergeWithProjection(
	root, source string,
	projection plan.MergeProjection,
	sourceStoreSelector string,
	output io.Writer,
) error {
	parsedProjection, err := plan.ParseMergeProjection(string(projection))
	if err != nil {
		return fmt.Errorf("prepare-merge projection: %w", err)
	}
	if parsedProjection != projection {
		return errors.New("prepare-merge projection is not canonical")
	}
	// The lane case is measured from the request to the staged merge.
	started := time.Now()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
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
		contains := exec.Command("git", "--no-replace-objects", "merge-base", "--is-ancestor", snapshot, "HEAD")
		contains.Dir = root
		contains.Env = gitauthority.RepositoryEnvironment()
		if err := contains.Run(); err == nil {
			fmt.Fprintf(output, "prepare-merge: HEAD already contains %s at %s\n", source, snapshot)
			return nil
		} else {
			exit, found := errors.AsType[*exec.ExitError](err)
			if !found || exit.ExitCode() != 1 {
				return fmt.Errorf("prepare-merge ancestry: %w", err)
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
		if err := preflightMergeConflicts(root, localRevision, snapshot); err != nil {
			return err
		}
		closureSnapshot, err := captureClosureEvidence(ctx, root, source, snapshot, sourceStoreSelector)
		if err != nil {
			return err
		}
		if closureSnapshot.cleanup != nil {
			defer closureSnapshot.cleanup()
			fmt.Fprintf(output, "prepare-merge: evidence snapshot IO=%+v; store-file counters exclude seal IO and replay\n", closureSnapshot.backup)
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
		merged, err := projectMergePlan(base, local, incoming, projection)
		if err != nil {
			return err
		}
		mergeID := "merge-" + snapshot[:12]
		lane := laneMerge(projection, local)
		// The lane dispatches only rows it owns; the build proves a lane's row.
		merged, err = insertItem(merged, mergeID, "Merge "+source+" at "+snapshot[:12], plan.ClassMerge, "", mergeVerify(lane))
		if err != nil {
			return err
		}
		if lane {
			merged.Items[0].Owner = local.Lane
		}
		latestLocal, err := gitOutput(root, "rev-parse", "--verify", "HEAD^{commit}")
		if err != nil || strings.TrimSpace(string(latestLocal)) != localRevision {
			return fmt.Errorf("prepare-merge local HEAD moved from snapshot %s", localRevision)
		}
		if err := mergeCapturedPlan(root, localRevision, snapshot, local, merged); err != nil {
			return err
		}
		keepMerge := false
		defer func() {
			if !keepMerge {
				abortMerge(root)
			}
		}()
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
			apiManifestJSONPath,
			modernGoBaselinePath,
			harnessSurfaceBaselinePath,
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
		if lane {
			// The lane's store rebinds its own decisions over the imported
			// ones and mirrors the source's activations, so the one gate
			// reads the same activation truth as the source's.
			if _, err := commandOutput(root, "go", "run", "./cmd/closure-scan", "-store", gitauthority.CanonicalOvergoDBDirectory, "-import-store", gitauthority.CanonicalOvergoDBDirectory); err != nil {
				return fmt.Errorf("prepare-merge lane closure rebind: %w", err)
			}
			report, err := mirrorLaneActivations(root, liveSourceStore)
			if err != nil {
				return fmt.Errorf("prepare-merge lane activations: %w", err)
			}
			fmt.Fprintf(output, "prepare-merge: lane store mirrors %d source activation(s): closure=%d copied=%v bindings=%d released=%d\n",
				report.SourceActive, report.ClosureRecords, report.Copied, report.Bindings, len(report.Released))
		}
		keepMerge = true
		fmt.Fprintf(
			output,
			"prepare-merge: %s@%s staged; land it with go run ./cmd/loop -land %s/do\n",
			source, snapshot, mergeID,
		)
		if lane {
			fmt.Fprintf(output, "prepare-merge: lane merge prepared in %s from the request to the staged merge\n", time.Since(started).Round(time.Millisecond))
		}
		return nil
	})
}

// restoreMergeTargetBaselines retains this lane's reviewed ratchets across a
// generated-document conflict. The gate later recomputes them from merged code
// and may lower a ceiling, while a raise still requires explicit paydown.
func restoreMergeTargetBaselines(root, localRevision string) error {
	for _, path := range []string{modernGoBaselinePath, harnessSurfaceBaselinePath} {
		baseline, err := gitOutput(root, "show", localRevision+":"+path)
		if err != nil {
			return fmt.Errorf("restore target %s: %w", path, err)
		}
		if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(path)), baseline, clioptions.OutputFileMode); err != nil {
			return fmt.Errorf("restore target %s: %w", path, err)
		}
	}
	return nil
}

// mergeCapturedPlan performs the textual merge and restores the validated plan
// projection. Its caller holds withPlanMutation's lock from capture through
// staging. Failed publication aborts the merge; later preparation failures are
// rolled back by prepareMergeWithProjection.
func mergeCapturedPlan(root, localRevision, snapshot string, local, merged plan.Plan) (err error) {
	_, mergeErr := commandOutput(root, "git", "merge", "--no-ff", "--no-commit", snapshot)
	postMergeHead, headErr := gitOutput(root, "rev-parse", "--verify", "HEAD^{commit}")
	if headErr != nil || strings.TrimSpace(string(postMergeHead)) != localRevision {
		return fmt.Errorf("prepare-merge local HEAD moved during merge from snapshot %s", localRevision)
	}
	defer func() {
		if err != nil {
			abortMerge(root)
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
	return saveCapturedPlanMutation(root, local, merged)
}

func abortMerge(root string) {
	abort := exec.Command("git", "--no-replace-objects", "merge", "--abort")
	abort.Dir = root
	abort.Env = gitauthority.RepositoryEnvironment()
	_ = abort.Run()
}

// laneMerge reports the lane case: a first-parent-target merge into a plan
// that names its lane.
func laneMerge(projection plan.MergeProjection, local plan.Plan) bool {
	return projection == plan.MergeProjectionFirstParentTarget && local.Lane != ""
}

// mergeVerify names the merge row's verify: a lane's merge row is proven by
// the build, since the compatibility check regenerates on the target's
// own line, not the lane's.
func mergeVerify(lane bool) string {
	if lane {
		return "go build ./..."
	}
	return "go run ./cmd/compatibility -check"
}

// mirrorLaneActivations mirrors the live source store's active inference
// activations into the lane's canonical store.
func mirrorLaneActivations(root, liveSourceStore string) (lanestore.Report, error) {
	source, err := overgodb.OpenReadOnly(liveSourceStore)
	if err != nil {
		return lanestore.Report{}, err
	}
	defer source.Close()
	target, err := overgodb.Open(filepath.Join(root, gitauthority.CanonicalOvergoDBDirectory))
	if err != nil {
		return lanestore.Report{}, err
	}
	defer target.Close()
	return lanestore.MirrorActivations(context.Background(), source, target, lanestore.MirrorDepth)
}

// Preflight writes Git objects only; HEAD, index and worktree stay unchanged.
func preflightMergeConflicts(root, target, source string) error {
	out, err := gitOutput(root, "merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", target, source)
	if err == nil {
		return nil
	}
	exit, found := errors.AsType[*exec.ExitError](err)
	if !found || exit.ExitCode() != 1 {
		return fmt.Errorf("prepare-merge conflict preflight: %w", err)
	}
	_, paths, found := bytes.Cut(out, []byte{0})
	if !found || len(bytes.Trim(paths, "\x00")) == 0 {
		return fmt.Errorf("prepare-merge conflict preflight returned no conflict paths: %w", err)
	}
	for path := range bytes.SplitSeq(paths, []byte{0}) {
		if len(path) != 0 && !mergeOwnedDocument(string(path)) {
			return fmt.Errorf("prepare-merge source conflict in %q; evidence snapshot not copied", path)
		}
	}
	return nil
}

func mergeOwnedDocument(path string) bool {
	switch path {
	case plan.Path,
		compatibilityDocumentPath,
		trainingCompatibilityDocumentPath,
		apiManifestJSONPath,
		modernGoBaselinePath,
		harnessSurfaceBaselinePath:
		return true
	default:
		return false
	}
}

func regenerateMergeOwnedDocuments(root, localRevision string) error {
	if err := restoreMergeTargetBaselines(root, localRevision); err != nil {
		return err
	}
	commands := []struct {
		label string
		args  []string
	}{
		{"lower merged modern-Go baseline", []string{"run", "./cmd/modern-census", "-lower-baseline"}},
		{"refresh merged API manifest", []string{"run", "./cmd/api-manifest", "-update"}},
	}
	for _, command := range commands {
		if _, err := commandOutput(root, "go", command.args...); err != nil {
			return fmt.Errorf("%s: %w", command.label, err)
		}
	}
	return nil
}

func projectMergePlan(base, local, incoming plan.Plan, projection plan.MergeProjection) (plan.Plan, error) {
	if projection == plan.MergeProjectionFirstParentTarget {
		return local, nil
	}
	return plan.MergeDocuments(base, local, incoming)
}

// uniqueMergeBase is the one merge base of the two revisions. Criss-cross
// history with several bases takes the one on the target's first-parent
// chain; without exactly one there, the plan merge has no base and refuses.
func uniqueMergeBase(root, localRevision, incomingRevision string) (string, error) {
	output, err := gitOutput(root, "merge-base", "--all", localRevision, incomingRevision)
	if err != nil {
		return "", fmt.Errorf("prepare-merge: merge base: %w", err)
	}
	bases := strings.Fields(string(output))
	if len(bases) == 1 {
		return bases[0], nil
	}
	output, err = gitOutput(root, "rev-list", "--first-parent", localRevision)
	if err != nil {
		return "", err
	}
	chain := map[string]bool{}
	for revision := range strings.FieldsSeq(string(output)) {
		chain[revision] = true
	}
	var selected []string
	for _, base := range bases {
		if chain[base] {
			selected = append(selected, base)
		}
	}
	if len(selected) != 1 {
		return "", fmt.Errorf("prepare-merge: needs one merge base on the target's first-parent chain, found %d of %d", len(selected), len(bases))
	}
	return selected[0], nil
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
	return commandInput(root, nil, name, args...)
}

// commandInput is commandOutput with the bytes the command reads.
func commandInput(root string, input []byte, name string, args ...string) ([]byte, error) {
	commandArgs := args
	if name == "git" {
		commandArgs = append([]string{"--no-replace-objects"}, args...)
	}
	command := exec.Command(name, commandArgs...)
	command.Dir = root
	command.Stdin = bytes.NewReader(input)
	if name == "git" {
		command.Env = gitauthority.RepositoryEnvironment()
	}
	output, err := command.CombinedOutput()
	if err != nil {
		return output, fmt.Errorf("%s %s: %w: %s", name, strings.Join(args, " "), err, strings.TrimSpace(string(output)))
	}
	return output, nil
}
