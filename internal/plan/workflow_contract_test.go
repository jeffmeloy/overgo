package plan

import (
	"crypto/sha256"
	"fmt"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func dispatchLeaseFixture(t *testing.T, task, worktree, worker, acquisition string) WorkLease {
	t.Helper()
	lease, err := NewWorkLease(WorkLease{
		Task: task, Worktree: worktree, Worker: worker, Role: UnassignedRole,
		Branch: "codex/dispatch", TargetHead: orchestrationHead,
		ConflictsWith: []string{}, Claims: WorkspaceClaims{WholeWorktree: true},
		Contract:    fmt.Sprintf("%x", sha256.Sum256([]byte(task))),
		Acquisition: fmt.Sprintf("%x", sha256.Sum256([]byte(acquisition))),
	})
	if err != nil {
		t.Fatal(err)
	}
	return lease
}

func recordClaim(t *testing.T, store *overgodb.Store, lease WorkLease) (WorkLease, error) {
	t.Helper()
	content, err := workLeaseCodec.Content(lease)
	if err != nil {
		t.Fatal(err)
	}
	return RecordWorkLease(t.Context(), store, content.Data)
}

func TestAtomicDispatchClaim(t *testing.T) {
	t.Run("legacy retirement cannot see dispatch claims", func(t *testing.T) {
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		claim := dispatchLeaseFixture(t, "alpha/one", "C:/repo/claim", "worker-one", "head")
		if _, err := recordClaim(t, store, claim); err != nil {
			t.Fatal(err)
		}
		content, err := workLeaseCodec.Content(claim)
		if err != nil {
			t.Fatal(err)
		}
		if content.Descriptor.Schema == WorkLeaseSchema {
			t.Fatal("dispatch claim reused the legacy wire schema")
		}
		if retired, err := RetireLegacyLeases(t.Context(), store, 0); err != nil || len(retired) != 0 {
			t.Fatalf("legacy retirement consumed new claim: %v %v", retired, err)
		}
		if err := ResolveWorkLeaseOwner(t.Context(), store, claim); err != nil {
			t.Fatal(err)
		}
		claim.Version = workLeaseVersion
		if _, err := workLeaseCodec.New(claim); err == nil {
			t.Fatal("legacy schema accepted worker-owned dispatch semantics")
		}
		legacy := workLeaseFixture(t, "beta/one", "C:/repo/advisory")
		legacyContent, err := workLeaseCodec.Content(legacy)
		if err != nil || legacyContent.Descriptor.Schema != WorkLeaseSchema {
			t.Fatalf("legacy contract changed: %v", err)
		}
	})
	t.Run("priority follows real prerequisites", func(t *testing.T) {
		document := Plan{Items: []Item{
			{ID: "publication", Status: StatusOpen, Steps: []Step{
				{ID: "memoize", Status: StatusOpen, Verify: "go test ./internal/gate", DependsOn: []string{"publication/measure", "selection/source-binding"}},
				{ID: "measure", Status: StatusOpen, Verify: "go test ./internal/gate"},
				{ID: "tooling", Status: StatusOpen, Verify: "go test ./cmd/plan"},
			}},
			{ID: "selection", Status: StatusOpen, Steps: []Step{{ID: "source-binding", Status: StatusOpen, Verify: "go test ./internal/automationcheck"}}},
		}}
		before, err := completionJSONDigest(document, "unchanged priority input")
		if err != nil {
			t.Fatal(err)
		}
		ordered, err := dispatchPriority(document, UnassignedRole, testCompletionAuthority(t, document))
		if err != nil {
			t.Fatal(err)
		}
		want := []Ref{{Item: "publication", Step: "measure"}, {Item: "selection", Step: "source-binding"}, {Item: "publication", Step: "tooling"}}
		if !slices.Equal(ordered, want) {
			t.Fatalf("priority=%v want=%v", ordered, want)
		}
		after, err := completionJSONDigest(document, "unchanged priority input")
		if err != nil || after != before {
			t.Fatal("scheduling rewrote the plan")
		}
	})
	t.Run("inspection selection and contract changes", func(t *testing.T) {
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		document := frontierTestPlan()
		authority := testCompletionAuthority(t, document)
		head, sequence := store.Head()
		view, err := selectDispatch(t.Context(), store, document, UnassignedRole, "C:/repo/first", "worker-one", "", authority)
		if err != nil || view.Item != "alpha" || view.Step != "one" || view.Claim != nil {
			t.Fatalf("inspection=%+v %v", view, err)
		}
		if after, count := store.Head(); after != head || count != sequence {
			t.Fatal("inspection acquired ownership")
		}
		lease := dispatchLeaseFixture(t, "alpha/one", "C:/repo/first", "worker-one", "head")
		lease.Contract, err = dispatchContract(document.Items[0], document.Items[0].Steps[0])
		if err != nil {
			t.Fatal(err)
		}
		lease, err = NewWorkLease(lease)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := recordClaim(t, store, lease); err != nil {
			t.Fatal(err)
		}
		for _, worker := range []string{"worker-two", ""} {
			view, err := selectDispatch(t.Context(), store, document, UnassignedRole, lease.Worktree, worker, "", authority)
			if err != nil || view.Waiting == "" || view.Claim == nil || view.Complete {
				t.Fatalf("same-tree inspection=%+v %v", view, err)
			}
		}
		view, err = selectDispatch(t.Context(), store, document, UnassignedRole, "C:/repo/second", "worker-two", "", authority)
		if err != nil || view.Item != "beta" || view.Step != "solo" || view.Waiting != "" {
			t.Fatalf("independent selection=%+v %v", view, err)
		}
		view, err = selectDispatch(t.Context(), store, document, UnassignedRole, lease.Worktree, lease.Worker, "", authority)
		if err != nil || view.Claim == nil || view.Claim.ID != lease.ID || view.Waiting != "" {
			t.Fatalf("resume=%+v %v", view, err)
		}
		document.Items[0].Steps[1].Title = "unrelated sibling edit"
		if err := validateDispatchContract(document, lease, UnassignedRole, testCompletionAuthority(t, document)); err != nil {
			t.Fatalf("sibling invalidated claim: %v", err)
		}
		document.Items[0].Steps[0].Verify = "go test ./internal/artifact"
		view, err = selectDispatch(t.Context(), store, document, UnassignedRole, lease.Worktree, lease.Worker, "", testCompletionAuthority(t, document))
		if err != nil || view.Waiting == "" || view.Complete {
			t.Fatalf("changed contract silently resumed: %+v %v", view, err)
		}
	})
	t.Run("competing workers share one atomic winner", func(t *testing.T) {
		for _, sharedTree := range []bool{false, true} {
			t.Run(fmt.Sprint(sharedTree), func(t *testing.T) {
				store, err := overgodb.Open(t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				defer store.Close()
				first := dispatchLeaseFixture(t, "alpha/one", "C:/repo/first", "worker-one", "head")
				second := dispatchLeaseFixture(t, "alpha/one", "C:/repo/second", "worker-two", "head")
				if sharedTree {
					second.Worktree = first.Worktree
					second, err = NewWorkLease(second)
					if err != nil {
						t.Fatal(err)
					}
				}
				start := make(chan struct{})
				type result struct {
					lease WorkLease
					err   error
				}
				results := make(chan result, 2)
				for _, lease := range []WorkLease{first, second} {
					content, err := workLeaseCodec.Content(lease)
					if err != nil {
						t.Fatal(err)
					}
					go func() {
						<-start
						actual, err := RecordWorkLease(t.Context(), store, content.Data)
						results <- result{actual, err}
					}()
				}
				close(start)
				var winners []WorkLease
				for range []WorkLease{first, second} {
					result := <-results
					if result.err == nil {
						winners = append(winners, result.lease)
					}
				}
				if len(winners) != 1 {
					t.Fatalf("atomic winner count = %d", len(winners))
				}
				if err := ResolveWorkLeaseOwner(t.Context(), store, winners[0]); err != nil {
					t.Fatal(err)
				}
				loser := first
				if winners[0].ID == first.ID {
					loser = second
				}
				for _, alias := range workLeaseAliases(loser)[1:] {
					id, found, err := artifact.ResolveAlias(t.Context(), store, alias)
					if err != nil || found && id == loser.ID {
						t.Fatalf("loser published partial ownership: %s %v", alias, err)
					}
				}
			})
		}
	})
	t.Run("restart release replay and reacquisition", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "store")
		store, err := overgodb.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		lease := dispatchLeaseFixture(t, "alpha/one", "C:/repo/first", "worker-one", "first-head")
		if _, err := recordClaim(t, store, lease); err != nil {
			t.Fatal(err)
		}
		head, sequence := store.Head()
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		store, err = overgodb.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if resumed, err := recordClaim(t, store, lease); err != nil || resumed.ID != lease.ID {
			t.Fatalf("resume = %s %v", resumed.ID, err)
		}
		if after, count := store.Head(); after != head || count != sequence {
			t.Fatal("retry duplicated acquisition")
		}
		if err := releaseClaim(t, store, lease.ID, "other-worker", "cancelled"); err == nil {
			t.Fatal("foreign worker released claim")
		}
		if err := releaseClaim(t, store, lease.ID, lease.Worker, "expired"); err == nil {
			t.Fatal("time-based release admitted")
		}
		if err := releaseClaim(t, store, lease.ID, lease.Worker, "cancelled"); err != nil {
			t.Fatal(err)
		}
		if _, err := recordClaim(t, store, lease); err == nil {
			t.Fatal("released acquisition replay pretended to own work")
		}
		if _, found, err := ReadWorkLease(t.Context(), store, lease.ID); err != nil || !found {
			t.Fatalf("released evidence lost: %v", err)
		}
		next := dispatchLeaseFixture(t, lease.Task, lease.Worktree, lease.Worker, "after-release")
		if _, err := recordClaim(t, store, next); err != nil {
			t.Fatal(err)
		}
		if err := releaseClaim(t, store, lease.ID, lease.Worker, "cancelled"); err != nil {
			t.Fatal(err)
		}
		if err := ResolveWorkLeaseOwner(t.Context(), store, next); err != nil {
			t.Fatalf("old release displaced successor: %v", err)
		}
	})
	t.Run("renewal and legacy writes preserve exact ownership", func(t *testing.T) {
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		lease := dispatchLeaseFixture(t, "alpha/one", "C:/repo/first", "worker-one", "head")
		if _, err := recordClaim(t, store, lease); err != nil {
			t.Fatal(err)
		}
		legacy := workLeaseFixture(t, lease.Task, lease.Worktree)
		if _, err := recordClaim(t, store, legacy); err == nil {
			t.Fatal("legacy write displaced dispatch claim")
		}
		next := lease
		next.Previous = lease.ID
		next.Contract = fmt.Sprintf("%x", sha256.Sum256([]byte("reviewed amendment")))
		next, err = NewWorkLease(next)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := recordClaim(t, store, next); err != nil {
			t.Fatal(err)
		}
		if _, err := recordClaim(t, store, lease); err == nil {
			t.Fatal("stale claim displaced renewal")
		}
		if err := releaseClaim(t, store, lease.ID, lease.Worker, "handoff"); err == nil {
			t.Fatal("old owner released renewal")
		}
		if err := ResolveWorkLeaseOwner(t.Context(), store, next); err != nil {
			t.Fatal(err)
		}
		otherTask := dispatchLeaseFixture(t, "beta/one", lease.Worktree, "worker-two", "head")
		if _, err := recordClaim(t, store, otherTask); err == nil {
			t.Fatal("different task permitted simultaneous worktree mutation")
		}
		otherTree := dispatchLeaseFixture(t, "beta/one", "C:/repo/second", "worker-two", "head")
		if _, err := recordClaim(t, store, otherTree); err != nil {
			t.Fatalf("independent claim refused: %v", err)
		}
		if err := ValidateFrontierLeases([]Ref{{Item: "alpha", Step: "one"}, {Item: "beta", Step: "one"}}, []WorkLease{next, otherTree}); err != nil {
			t.Fatalf("distinct workers sharing unassigned role were serialized: %v", err)
		}
	})
}

func releaseClaim(t *testing.T, store *overgodb.Store, id artifact.ID, worker, reason string) error {
	t.Helper()
	lease, found, err := ReadWorkLease(t.Context(), store, id)
	if err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("lease %s missing", id)
	}
	batch, err := lease.ReleaseBatch(worker, reason)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(t.Context(), store, batch)
	return err
}
