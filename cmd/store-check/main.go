//overgo:runtime-inputs caller

// store-check proves the published side of the source-data contract: every
// active recipe chain in OvergoDB must still load through the production
// readers under the current document schemas. The gate runs source-only
// checks, so an unversioned schema change ships green while every published
// document becomes unreadable -- this command is the missing half, and the
// gate runs it whenever a schema-owning package changes. Every broken chain
// is refused: the committed list of chains broken before the check existed
// emptied, so the rule holds without one.
package main

import (
	"context"
	"flag"
	"fmt"
	"sort"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

type brokenChain struct {
	Alias   string
	Failure string
}

func main() {
	clioptions.MainNamed("store-check", run)
}

func run() error {
	repository := flag.String("repo", "", "OvergoDB store directory; empty resolves via the data-root contract (OVERGO_DATA_ROOT, local-models.json, or ./overgodb-store)")
	blobs := flag.Bool("blobs", false, "also hold every live blob, loose or packed, to its identity")
	flag.Parse()
	// A plan-row verify runs in the gate's candidate worktree, which holds no
	// store; the data-root contract names the canonical one.
	resolved, err := dataroot.StoreRoot(*repository)
	if err != nil {
		return err
	}
	*repository = resolved
	failures, checked, err := activeChainFailures(*repository)
	if err != nil {
		return err
	}
	if *blobs {
		verified, err := verifyBlobs(*repository)
		if err != nil {
			return err
		}
		fmt.Printf("store-check: blobs=%d verified\n", verified)
	}
	for _, failure := range failures {
		fmt.Printf("BROKEN %s: %s\n", failure.Alias, failure.Failure)
	}
	fmt.Printf("store-check: chains=%d broken=%d\n", checked, len(failures))
	if len(failures) > 0 {
		return fmt.Errorf("store-check: %d published chain(s) no longer load through the production readers", len(failures))
	}
	return nil
}

// verifyBlobs holds every live blob of the store to its identity.
func verifyBlobs(repository string) (int, error) {
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return 0, err
	}
	defer store.Close()
	return store.VerifyBlobs(context.Background())
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
