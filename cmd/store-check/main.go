// store-check proves the published side of the source-data contract: every
// active recipe chain in OvergoDB must still load through the production
// readers under the current document schemas. The gate runs source-only
// checks, so an unversioned schema change ships green while every published
// document becomes unreadable -- this command is the missing half, and the
// gate runs it whenever a schema-owning package changes.
//
// Chains that were already broken before the check existed are enumerated in
// a committed baseline with their exact failure text. The check refuses new
// breakage and refuses a stale baseline entry (a chain that recovered or now
// fails differently), so the baseline can only shrink deliberately.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"slices"
	"sort"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
)

type baselineFile struct {
	Broken []brokenChain `json:"broken"`
}

type brokenChain struct {
	Alias   string `json:"alias"`
	Failure string `json:"failure"`
}

func main() {
	clioptions.MainNamed("store-check", run)
}

func run() error {
	repository := flag.String("repo", "", "OvergoDB store directory; empty resolves via the data-root contract (OVERGO_DATA_ROOT, local-models.json, or ./overgodb-store)")
	baselinePath := flag.String("baseline", "docs/published_debt.json", "committed baseline of already-broken chains")
	printBaseline := flag.Bool("print-baseline", false, "emit the current failures as a baseline document and exit zero")
	flag.Parse()
	if strings.TrimSpace(*repository) == "" {
		// A plan-row verify runs in the gate's candidate worktree, which
		// holds no store; the data-root contract names the canonical one.
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		*repository = roots.Store
	}
	failures, checked, err := activeChainFailures(*repository)
	if err != nil {
		return err
	}
	if *printBaseline {
		return clioptions.WritePrettyJSON(os.Stdout, baselineFile{Broken: failures})
	}
	baseline, err := readBaseline(*baselinePath)
	if err != nil {
		return err
	}
	known := make(map[string]string, len(baseline.Broken))
	for _, entry := range baseline.Broken {
		known[entry.Alias] = entry.Failure
	}
	var fresh, drifted, recovered []string
	for _, failure := range failures {
		recorded, listed := known[failure.Alias]
		switch {
		case !listed:
			fresh = append(fresh, failure.Alias+": "+failure.Failure)
		case recorded != failure.Failure:
			drifted = append(drifted, failure.Alias+": recorded "+recorded+"; now "+failure.Failure)
		}
		delete(known, failure.Alias)
	}
	for alias := range known {
		recovered = append(recovered, alias)
	}
	slices.Sort(fresh)
	slices.Sort(drifted)
	slices.Sort(recovered)
	for _, line := range fresh {
		fmt.Printf("NEW BREAKAGE %s\n", line)
	}
	for _, line := range drifted {
		fmt.Printf("BASELINE DRIFT %s\n", line)
	}
	for _, line := range recovered {
		fmt.Printf("RECOVERED %s (remove from baseline)\n", line)
	}
	fmt.Printf("store-check: chains=%d broken=%d baseline=%d\n", checked, len(failures), len(baseline.Broken))
	if len(fresh) > 0 || len(drifted) > 0 || len(recovered) > 0 {
		return fmt.Errorf("store-check: published chains diverge from source: new=%d drifted=%d recovered=%d",
			len(fresh), len(drifted), len(recovered))
	}
	return nil
}

// activeChainFailures walks every activation alias through the production
// readers and returns the failing chains sorted by alias.
func activeChainFailures(repository string) ([]brokenChain, int, error) {
	ctx := context.Background()
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return nil, 0, err
	}
	defer store.Close()
	var failures []brokenChain
	checked := 0
	err = store.VisitAliases(ctx, "", func(alias overgodb.AliasView) error {
		modelID, task, ok := modelrecipe.ParseActiveAlias(alias.Name)
		if !ok {
			return nil
		}
		checked++
		if err := modelrecipe.VerifyPublishedChain(ctx, store, modelID, task); err != nil {
			failures = append(failures, brokenChain{Alias: alias.Name, Failure: err.Error()})
		}
		return nil
	})
	if err != nil {
		return nil, 0, err
	}
	sort.Slice(failures, func(left, right int) bool { return failures[left].Alias < failures[right].Alias })
	return failures, checked, nil
}

func readBaseline(path string) (baselineFile, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return baselineFile{}, fmt.Errorf("store-check: read baseline: %w", err)
	}
	var baseline baselineFile
	if err := strictjson.DecodeBytes(raw, &baseline); err != nil {
		return baselineFile{}, fmt.Errorf("store-check: decode baseline: %w", err)
	}
	for _, entry := range baseline.Broken {
		if strings.TrimSpace(entry.Alias) == "" || strings.TrimSpace(entry.Failure) == "" {
			return baselineFile{}, fmt.Errorf("store-check: baseline entry lacks alias or failure text")
		}
	}
	return baseline, nil
}
