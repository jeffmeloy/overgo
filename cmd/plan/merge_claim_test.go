package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/authoritylock"
	"overgo/internal/plan"
	"overgo/internal/worklease"
)

// Exercise prepare-merge's publication transaction with real Git conflict
// markers, index entries and rollback, without copying campaign evidence or
// running unrelated generated-document commands.
func TestPrepareMergeCapturedPlanConflict(t *testing.T) {
	for _, claimedEdit := range []bool{false, true} {
		t.Run(map[bool]string{false: "retain_claimed_local_plan", true: "refuse_claimed_contract_edit"}[claimedEdit], func(t *testing.T) {
			t.Setenv(plan.AutomationRoleEnvironment, worklease.UnassignedRole)
			t.Setenv(plan.AutomationWorkerEnvironment, "")
			base := mutationPlan(t, "base title")
			root := initializePlanTestRepository(t, base)
			git := func(args ...string) string {
				t.Helper()
				out, err := gitOutput(root, args...)
				if err != nil {
					t.Fatal(err)
				}
				return strings.TrimSpace(string(out))
			}
			if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("overgodb-store/\ntmp/\ndocs/.dispatch\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			git("add", ".gitignore")
			git("commit", "-m", "ignore runtime state")
			branch := git("branch", "--show-current")
			git("checkout", "-b", "incoming")
			incoming := clonePlan(t, base)
			incoming.Items[0].Title = "incoming title"
			saveMutationPlan(t, root, incoming)
			git("commit", "-am", "incoming plan")
			snapshot := git("rev-parse", "HEAD")
			git("checkout", branch)
			local := clonePlan(t, base)
			local.Items[0].Title = "local title"
			path := saveMutationPlan(t, root, local)
			git("commit", "-am", "local plan")
			head := git("rev-parse", "HEAD")

			// Prove this fixture reaches the reported failure, rather than a
			// clean merge or a zero-conflict approximation.
			if _, err := gitOutput(root, "merge", "--no-ff", "--no-commit", snapshot); err == nil {
				t.Fatal("fixture did not create a real Git conflict")
			}
			conflicted, err := os.ReadFile(path)
			if err != nil || !bytes.Contains(conflicted, []byte("<<<<<<<")) || git("diff", "--name-only", "--diff-filter=U") != plan.Path {
				t.Fatalf("expected conflicted plan, got %q: %v", conflicted, err)
			}
			git("merge", "--abort")
			// Capture checkout bytes after Git's platform newline conversion.
			before, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			claim, err := plan.ResolveDispatch(t.Context(), root, plan.DispatchRequest{Worker: "merge-fixture", Acquire: true})
			if err != nil || claim.Claim == nil || claim.Item != "row" {
				t.Fatalf("claim = %+v, %v", claim, err)
			}
			err = withPlanMutation(root, false, func(captured plan.Plan) error {
				// Publication must remain in the same critical section as capture.
				if lock, lockErr := authoritylock.Acquire(root); lockErr == nil {
					_ = lock.Close()
					t.Fatal("capture did not retain the mutation lock")
				}
				projected, err := projectMergePlan(base, captured, incoming, plan.CompletionAuthority{}, plan.CompletionAuthority{}, plan.MergeProjectionFirstParentTarget)
				if err != nil {
					return err
				}
				projected, err = insertMergeRow(projected, "merge-"+snapshot[:12], "Merge incoming", "", false)
				if err != nil {
					return err
				}
				if claimedEdit {
					// A candidate contract edit cannot be hidden by the textual
					// conflict or by aliasing the captured plan's nested slices.
					projected = clonePlan(t, projected)
					projected.Items[1].Steps[0].Title = "unauthorized edit"
				}
				return mergeCapturedPlan(root, head, snapshot, captured, projected)
			})
			if claimedEdit {
				if err == nil || !strings.Contains(err.Error(), "claimed by worker merge-fixture") || !strings.Contains(err.Error(), claim.Claim.ID.String()) {
					t.Fatalf("expected exact active-claim refusal, got %v", err)
				}
				after, readErr := os.ReadFile(path)
				if status := git("status", "--porcelain"); readErr != nil || !bytes.Equal(before, after) || status != "" {
					t.Fatalf("refused publication changed state: read=%v equal=%t status=%q", readErr, bytes.Equal(before, after), status)
				}
				if _, err := gitOutput(root, "rev-parse", "--verify", "MERGE_HEAD"); err == nil {
					t.Fatal("refused publication retained MERGE_HEAD")
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				after, err := plan.Load(path)
				if err != nil || len(after.Items) != 2 || after.Items[0].ID != "merge-"+snapshot[:12] {
					t.Fatalf("published plan = %+v, %v", after, err)
				}
				after.Items = after.Items[1:]
				if after.Digest() != local.Digest() {
					t.Fatal("publication substituted or changed the captured local plan")
				}
				git("add", "--", plan.Path)
				if git("rev-parse", "MERGE_HEAD") != snapshot || git("diff", "--name-only", "--diff-filter=U") != "" {
					t.Fatal("publication did not leave the exact merge ready for staging")
				}
				git("merge", "--abort")
			}
			if git("rev-parse", "HEAD") != head {
				t.Fatal("preparation moved HEAD")
			}
			retained, err := plan.ResolveDispatch(t.Context(), root, plan.DispatchRequest{Worker: "merge-fixture"})
			if err != nil || retained.Claim == nil || retained.Claim.ID != claim.Claim.ID {
				t.Fatalf("publication changed active claim: %+v, %v", retained, err)
			}
		})
	}
}
