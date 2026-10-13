package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/authoritylock"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/storepolicy"
)

// rebuild is overgodb rebuild: it writes the store's projected state into a
// fresh candidate store, stripping derived caches whose identities stay
// registered. The store keeps recipes, prototypes, activations, and evaluated
// performance; it stops carrying recomputable snapshots (owner ruling
// 2026-08-27, measured: 6.07 GB of code manifests against ~450 MB of
// knowledge). -without-retired also leaves out the records of the families the
// store's code lists as retired, which nothing writes and nothing reads, and
// prints what it left out of each and what it kept of each because a kept
// record names it; what it keeps, the candidate registers in the store's
// order. The store is opened to read it and is not written. -swap puts the
// candidate into service the way precheck does: under the gate's authority,
// only once every consumer check passed, and by renaming the whole store, so
// the superseded store stays beside it and a failed rename restores it.
func rebuild(flags *flag.FlagSet, args []string) error {
	repository := flags.String("repo", "overgodb-store", "OvergoDB store directory")
	destination := flags.String("destination", "", "fresh directory for the rebuilt store (default: <repo>.candidate)")
	withoutRetired := flags.Bool("without-retired", false, "leave out the records of the retired families, which nothing writes and nothing reads, keeping the ones a kept record names")
	checkout := flags.String("checkout", ".", "with -swap: checkout whose committed documents the rebuilt store must satisfy")
	swap := flags.Bool("swap", false, "put the rebuilt store into service once every consumer check passed, keeping the superseded store beside it")
	if err := flags.Parse(args); err != nil {
		return err
	}
	*destination = cmp.Or(*destination, filepath.Clean(*repository)+".candidate")
	if _, err := os.Lstat(*destination); err == nil {
		return fmt.Errorf("overgodb rebuild:%s already exists; use a fresh destination", *destination)
	}
	if *swap {
		// One hold of the gate's authority covers the rebuild, its checks and
		// the swap, as it does for precheck.
		authority, err := authoritylock.Acquire(*checkout)
		if err != nil {
			return err
		}
		defer authority.Close()
	}
	ctx := context.Background()
	source, err := overgodb.OpenReadOnly(*repository)
	if err != nil {
		return err
	}
	head, sequence := source.Head()
	report, err := overgodb.Rebuild(ctx, source, *destination, storepolicy.DerivedCache, runrecord.RetiredGateAliasesAtRest, *withoutRetired)
	if err := errors.Join(err, source.Close()); err != nil {
		return err
	}
	fmt.Printf("rebuilt %s: artifacts=%d contents_kept=%d (%.2f GB) contents_dropped=%d (%.2f GB) manifests=%d lineage=%d locations=%d aliases=%d batches=%d\n",
		*destination, report.Artifacts, report.ContentsKept, float64(report.BytesKept)/1e9,
		report.ContentsDropped, float64(report.BytesDropped)/1e9,
		report.Manifests, report.LineageEdges, report.Locations, report.Aliases, report.Batches)
	for _, line := range report.Retired() {
		fmt.Println(line)
	}
	if !*swap {
		return nil
	}
	result := receipt{OperationProof: overgodb.OperationProof{HeadBefore: head.String(), SequenceBefore: sequence}, Candidate: *destination}
	err = proveRebuilt(ctx, &result, *checkout)
	if err == nil {
		err = conclude(&result, true, func() (string, error) {
			return overgodb.ReplaceStore(ctx, *repository, *destination, result.HeadBefore)
		})
	}
	return errors.Join(err, json.NewEncoder(os.Stdout).Encode(result))
}

// proveRebuilt runs precheck's consumer checks against the rebuilt candidate
// and commits the proof into it when every one passed.
func proveRebuilt(ctx context.Context, result *receipt, checkout string) error {
	store, err := overgodb.Open(result.Candidate)
	if err != nil {
		return err
	}
	head, sequence := store.Head()
	result.HeadAfter, result.SequenceAfter = head.String(), sequence
	return errors.Join(prove(ctx, result, store, checkout), store.Close())
}
