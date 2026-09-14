package gate

import (
	"cmp"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
)

// Ratchets on the package's own suite. Measured 2026-09-14: the retained
// suite-cost record of the previous gate held 938 s wall for 288 tests, 876 s
// of them serial, with thirteen live-tree package graph loads of 22 s each;
// after the shared checkout, 216 parallel tests and the background fixture
// build the gate's test-owners phase measured 349 s for this package and
// isolated short runs held 214 to 398 s over six runs of the same code (106
// to 253 s of serial Setenv and hook tests, then the parallel tests, which
// are process-spawn bound: 8-way and 32-way parallelism give the same wall)
// with one graph load. The budget is half the prior wall and above the
// measured range; the counters are exact: a live-tree graph is loaded once
// per binary through liveRepository and no test adds a candidate worktree
// of the live repository outside it.
const (
	suiteWallBudget          = 480 * time.Second
	suiteLiveGraphLoadBudget = 1
	suiteLiveWorktreeBudget  = 0
	suiteLiveGraphNodes      = 500
	suiteWallLinePrefix      = "suite wall:"
	suiteWallPackage         = "overgo/internal/gate"
)

var (
	suiteLiveGraphLoads atomic.Int64
	suiteLiveWorktrees  atomic.Int64
	suiteGraphLoads     []string
	suiteGraphMutex     sync.Mutex
)

func TestMain(m *testing.M) {
	flag.Parse()
	started := time.Now()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	gatePackageGraphLoadedHook = func(loaded string, nodes int) {
		suiteGraphMutex.Lock()
		suiteGraphLoads = append(suiteGraphLoads, fmt.Sprintf("%d", nodes))
		suiteGraphMutex.Unlock()
		if nodes >= suiteLiveGraphNodes {
			suiteLiveGraphLoads.Add(1)
		}
	}
	gateCandidateWorktreeAddedHook = func(repo string) {
		if repo == root {
			suiteLiveWorktrees.Add(1)
			return
		}
		if !liveStarted.Load() {
			return
		}
		if live, err := liveRepositoryOnce(); err == nil && repo == live.worktree {
			suiteLiveWorktrees.Add(1)
		}
	}
	// A complete run builds the shared checkout while the serial tests run,
	// so the first parallel acceptance finds the graph and base manifest ready.
	if selector := flag.Lookup("test.run"); selector != nil && selector.Value.String() == "" {
		liveStarted.Store(true)
		go liveRepositoryOnce()
	}
	code := m.Run()
	plans := int64(0)
	if liveStarted.Load() {
		live, err := liveRepositoryOnce()
		if err == nil {
			plans = live.plans.Load()
			err = live.teardown()
		}
		if err != nil {
			fmt.Println(err)
			code = 1
		}
	}
	suiteGraphMutex.Lock()
	fmt.Printf("%s %.1fs live_graph_loads=%d live_worktrees=%d live_plans=%d graph_nodes=%s\n",
		suiteWallLinePrefix, time.Since(started).Seconds(), suiteLiveGraphLoads.Load(), suiteLiveWorktrees.Load(), plans, strings.Join(suiteGraphLoads, ","))
	suiteGraphMutex.Unlock()
	if code == 0 && (suiteLiveGraphLoads.Load() > suiteLiveGraphLoadBudget || suiteLiveWorktrees.Load() > suiteLiveWorktreeBudget) {
		fmt.Printf("suite wall ratchet: live-tree work exceeded its budget: graph loads %d > %d or worktrees %d > %d\n",
			suiteLiveGraphLoads.Load(), suiteLiveGraphLoadBudget, suiteLiveWorktrees.Load(), suiteLiveWorktreeBudget)
		code = 1
	}
	os.Exit(code)
}

// TestSuiteWallRatchet holds the package wall the latest gate measured: the
// newest retained suite-cost record whose short invocation ranks this
// package stays under the budget. A complete run measures its own wall in
// TestMain and enforces the live-tree counters here.
func TestSuiteWallRatchet(t *testing.T) {
	selector := flag.Lookup("test.run")
	if selector == nil || !strings.Contains(selector.Value.String(), "TestSuiteWallRatchet") {
		loads, worktrees := suiteLiveGraphLoads.Load(), suiteLiveWorktrees.Load()
		if loads > suiteLiveGraphLoadBudget || worktrees > suiteLiveWorktreeBudget {
			t.Fatalf("live-tree work exceeded its budget: graph loads %d > %d or worktrees %d > %d", loads, suiteLiveGraphLoadBudget, worktrees, suiteLiveWorktreeBudget)
		}
		t.Logf("complete run: live graph loads=%d worktrees=%d so far; the wall is measured in TestMain", loads, worktrees)
		return
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	result, err := store.Query(t.Context(), overgodb.Query{
		Kind: artifact.KindEvidence, Schema: suiteCostContract.Schema, MaxResults: 100000, Projection: overgodb.ProjectArtifacts,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Truncated {
		t.Fatalf("suite-cost catalog exceeds the listing bound: %d", len(result.Artifacts))
	}
	type suite struct {
		Package string   `json:"package"`
		Action  string   `json:"action"`
		Elapsed *float64 `json:"elapsed_seconds"`
	}
	type invocation struct {
		Short  bool    `json:"short"`
		Failed bool    `json:"failed"`
		WallNS uint64  `json:"wall_ns"`
		Suites []suite `json:"suites"`
	}
	// Descriptors sort by identity; the introducing commit orders the records.
	type retained struct {
		id       artifact.ID
		sequence uint64
	}
	records := make([]retained, 0, len(result.Artifacts))
	for _, descriptor := range result.Artifacts {
		introduction, found, err := store.ArtifactIntroduction(t.Context(), descriptor.ID)
		if err != nil || !found {
			t.Fatalf("suite-cost record %s has no introducing commit: %v", descriptor.ID, err)
		}
		records = append(records, retained{descriptor.ID, introduction.Sequence})
	}
	slices.SortFunc(records, func(left, right retained) int { return cmp.Compare(right.sequence, left.sequence) })
	for _, retainedRecord := range records {
		id := retainedRecord.id
		content, found, err := artifact.ReadContent(t.Context(), store, id)
		if err != nil || !found {
			t.Fatalf("suite-cost record %s unreadable: %v", id, err)
		}
		var record struct {
			Result      artifact.ID  `json:"result"`
			Invocations []invocation `json:"invocations"`
		}
		if err := json.Unmarshal(content.Data, &record); err != nil {
			t.Fatal(err)
		}
		for _, invocation := range record.Invocations {
			if !invocation.Short {
				continue
			}
			for _, ranked := range invocation.Suites {
				if ranked.Package != suiteWallPackage || ranked.Elapsed == nil {
					continue
				}
				t.Logf("record %s (gate result %s): %s %s in %.1fs, invocation wall %s over %d ranked package(s)",
					id, record.Result, suiteWallPackage, ranked.Action, *ranked.Elapsed, (time.Duration(invocation.WallNS) * time.Nanosecond).Round(time.Second), len(invocation.Suites))
				if *ranked.Elapsed > suiteWallBudget.Seconds() {
					t.Fatalf("suite wall %.1fs exceeds %s", *ranked.Elapsed, suiteWallBudget)
				}
				return
			}
		}
	}
	t.Fatalf("no retained suite-cost record ranks %s among %d record(s)", suiteWallPackage, len(result.Artifacts))
}
