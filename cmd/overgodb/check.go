package main

import (
	"context"
	"flag"
	"fmt"
	"sort"

	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
)

type brokenChain struct {
	Alias   string
	Failure string
}

// checkStore is overgodb check: it proves the published side of the
// source-data contract. Every active recipe chain in OvergoDB must still load
// through the production readers under the current document schemas;
// source-only checks let an unversioned schema change ship green while every
// published document becomes unreadable, and this is the missing half. Every
// broken chain is refused.
func checkStore(flags *flag.FlagSet, args []string) error {
	repository := flags.String("repo", "", "OvergoDB store directory; empty resolves via the data-root contract (OVERGO_DATA_ROOT, local-models.json, or ./overgodb-store)")
	blobs := flags.Bool("blobs", false, "also hold every live blob, loose or packed, to its identity")
	if err := flags.Parse(args); err != nil {
		return err
	}
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
		fmt.Printf("overgodb check:blobs=%d verified\n", verified)
	}
	for _, failure := range failures {
		fmt.Printf("BROKEN %s: %s\n", failure.Alias, failure.Failure)
	}
	fmt.Printf("overgodb check:chains=%d broken=%d\n", checked, len(failures))
	if len(failures) > 0 {
		return fmt.Errorf("overgodb check:%d published chain(s) no longer load through the production readers", len(failures))
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
