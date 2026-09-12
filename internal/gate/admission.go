package gate

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/protection"
	"overgo/internal/repoanalysis"
)

func gateWritesGit(
	reconcile, recordFailure, recoverInterrupted, admitReview, watchdog, inspectPlan, merge bool,
) bool {
	return reconcile || recoverInterrupted || inspectPlan || merge ||
		!recordFailure && !admitReview && !watchdog
}

func requireCanonicalGateStore(path string, writesAuthority bool) error {
	if writesAuthority && filepath.ToSlash(filepath.Clean(path)) != gateStorePath {
		return fmt.Errorf("gate: completion authority requires the canonical %s store", gateStorePath)
	}
	return nil
}

func requireExclusiveGateMode(
	reconcile, recordFailure, recoverInterrupted, admitReview, watchdog, inspectPlan, merge bool,
) error {
	selected := 0
	for _, enabled := range []bool{reconcile, recordFailure, recoverInterrupted, admitReview, watchdog, inspectPlan} {
		if enabled {
			selected++
		}
	}
	if selected > 1 || selected != 0 && merge {
		return errors.New("gate: operation modes are mutually exclusive")
	}
	return nil
}

func gatePlanProjection(value string, merge bool) (plan.MergeProjection, error) {
	projection, err := plan.ParseMergeProjection(value)
	if err != nil {
		return plan.MergeProjectionSemanticUnion, fmt.Errorf("gate: -plan-projection: %w", err)
	}
	if projection == plan.MergeProjectionFirstParentTarget && !merge {
		return plan.MergeProjectionSemanticUnion, errors.New("gate: -plan-projection requires -merge")
	}
	return projection, nil
}

func gateMergeSourceStore(value string, merge bool, projection plan.MergeProjection) (string, error) {
	value = strings.TrimSpace(value)
	if projection != plan.MergeProjectionFirstParentTarget {
		if value != "" {
			return "", errors.New("gate: -merge-source-store requires -merge with first-parent-target")
		}
		return "", nil
	}
	if !merge || value == "" {
		return "", errors.New("gate: first-parent-target requires -merge-source-store")
	}
	resolved, err := filepath.Abs(filepath.Clean(value))
	if err != nil {
		return "", fmt.Errorf("gate: resolve -merge-source-store: %w", err)
	}
	if filepath.Base(resolved) != gitauthority.CanonicalOvergoDBDirectory {
		return "", errors.New("gate: -merge-source-store must name a canonical overgodb-store directory")
	}
	return resolved, nil
}

// checkPlanBinding refuses any commit whose -plan is not the plan's current open
// step. This is the enforcement that makes off-plan work impossible to commit:
// the shared internal/plan.Current is the same "current step" cmd/plan dispatches
// and verifies, so the gate and the dispatcher can never disagree.
func resolvePlanBinding(repo, storePath, ref string) (plan.CompletionAuthority, string, error) {
	store, err := overgodb.OpenReadOnly(filepath.Join(repo, storePath))
	if err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	defer store.Close()
	return resolvePlanBindingWithStore(repo, ref, store)
}

// resolvePlanBindingWithStore derives plan authority through an already-open
// gate admission store. A normal gate opens the canonical store once for
// pending-state validation, plan binding, acceleration, and preparation;
// reopening the full journal between those checks made admission scale with
// the same durable history several times over.
func resolvePlanBindingWithStore(
	repo, ref string,
	store *overgodb.Store,
) (plan.CompletionAuthority, string, error) {
	if ref == "" {
		return plan.CompletionAuthority{}, "", fmt.Errorf("gate: -plan <item>/<step> is required (the plan's current open step; run `go run ./cmd/plan -next`)")
	}
	item, stepID, ok := strings.Cut(ref, "/")
	if !ok || item == "" || stepID == "" {
		return plan.CompletionAuthority{}, "", fmt.Errorf("gate: -plan must be <item>/<step>, got %q", ref)
	}
	document, err := plan.Load(filepath.Join(repo, plan.Path))
	if err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	if err := plan.Validate(document); err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	role, err := plan.AutomationRole("")
	if err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	head, err := command(repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	head = strings.TrimSpace(head)
	authority, err := plan.ResolveCompletionAuthority(context.Background(), repo, head, document, store)
	if err != nil {
		return plan.CompletionAuthority{}, "", err
	}
	it, st, open := plan.Current(document, role, authority)
	if !open {
		return plan.CompletionAuthority{}, "", fmt.Errorf("gate: -plan %s given but the plan is COMPLETE (no open step) -- nothing to commit against", ref)
	}
	if item != it.ID || stepID != st.ID {
		return plan.CompletionAuthority{}, "", fmt.Errorf("gate: -plan %s does NOT match the plan's current open step %s/%s -- commit only the dispatched step (off-plan commit REFUSED). If the plan is wrong, fix the plan first; do not commit around it", ref, it.ID, st.ID)
	}
	return authority, head, nil
}

func (g *gateContext) stepProtection() (bool, error) {
	configured, activated, err := protection.Verify(g.repo)
	if err != nil {
		return false, err
	}
	if stray, err := strayRootExecutables(g.repo); err != nil {
		return false, err
	} else if len(stray) > 0 {
		return false, fmt.Errorf(
			"gate: executables outside bin at the repository root: %s -- a single-package `go build ./cmd/x` drops its binary at the working directory; build with -o bin/<name>.exe or delete the stray",
			strings.Join(stray, ", "),
		)
	}
	g.stepEvidence["protection"] = configured + ";activation=" + activated
	g.note("protection: " + g.stepEvidence["protection"])
	return false, nil
}

// strayRootExecutables lists .exe files at the repository root: the
// gitignore hides them from status and only bin/ holds executables, so
// a root binary is always an accident this gate surfaces at commit
// time instead of leaving it for the release check.
func strayRootExecutables(repo string) ([]string, error) {
	entries, err := os.ReadDir(repo)
	if err != nil {
		return nil, err
	}
	var stray []string
	for _, entry := range entries {
		if !entry.IsDir() && strings.EqualFold(filepath.Ext(entry.Name()), ".exe") {
			stray = append(stray, entry.Name())
		}
	}
	return stray, nil
}

// stepScope refuses staged paths outside the plan (the commit would ship
// them) and reports unstaged co-implementer dirt without blocking on it.
func (g *gateContext) stepScope() (bool, error) {
	staged, err := stagedPaths(g.repo)
	if err != nil {
		return false, err
	}
	var rogue []string
	for _, p := range staged {
		if !slices.Contains(g.paths, filepath.ToSlash(p)) {
			rogue = append(rogue, p)
		}
	}
	if len(rogue) > 0 {
		return false, fmt.Errorf("staged outside -paths (would ship): %s", strings.Join(rogue, ", "))
	}
	status, err := command(g.repo, "git", "status", "--porcelain=v1", "-z", "--untracked-files=all")
	if err != nil {
		return false, err
	}
	dirty, err := repoanalysis.ParseDirtyStatus([]byte(status))
	if err != nil {
		return false, err
	}
	dirtyPaths, unplanned := scopeDirty(g.paths, dirty)
	var verificationInputs []string
	for _, path := range unplanned {
		g.note("unplanned dirty (not shipped): " + path)
		g.profileDirty = g.profileDirty || strings.HasSuffix(path, ".go")
		if unplannedVerificationInput(path) {
			verificationInputs = append(verificationInputs, path)
		}
	}
	if len(verificationInputs) != 0 {
		return false, fmt.Errorf(
			"unplanned verification input could affect evidence without entering the commit: %s",
			strings.Join(verificationInputs, ", "),
		)
	}
	for _, p := range g.paths {
		if _, err := os.Stat(filepath.Join(g.repo, filepath.FromSlash(p))); err != nil {
			if !dirtyPaths[p] {
				return false, fmt.Errorf("planned path %s neither exists nor is a tracked deletion", p)
			}
		}
	}
	return false, nil
}

// stagedPaths includes both the deletion and addition of a rename, independent
// of Git's rename presentation settings. Both affect verification and commit scope.
func stagedPaths(repo string) ([]string, error) {
	return gitLines(repo, "diff", "--cached", "--no-renames", "--name-only")
}

func unplannedVerificationInput(path string) bool {
	return path == "go.mod" || path == "go.sum" || strings.HasSuffix(path, ".go") ||
		(strings.HasPrefix(path, "internal/") || strings.HasPrefix(path, "cmd/") || strings.HasPrefix(path, "kernels/")) &&
			!strings.HasSuffix(path, ".md")
}

func scopeDirty(planned []string, dirty []repoanalysis.DirtyPath) (map[string]bool, []string) {
	visible := map[string]bool{}
	var unplanned []string
	for _, entry := range dirty {
		for _, path := range []string{entry.Path, entry.OriginalPath} {
			if path == "" || visible[path] {
				continue
			}
			visible[path] = true
			if !slices.Contains(planned, path) {
				unplanned = append(unplanned, path)
			}
		}
	}
	return visible, unplanned
}

// applies the plan-only lane commit rule to the operator's planned paths
// before the plan path joins them
func (g *gateContext) refusePlanOnlyCommit(merge bool) error {
	document, err := g.loadPlan()
	if err != nil {
		return err
	}
	return plan.PlanOnlyCommitRefusal(document, g.planRef, g.paths, merge)
}
