// overgodb-compact is retention's operator entry. Its default operation
// releases, in place, the bytes of the re-derivable cache documents
// (storepolicy.DerivedCache) that nothing reaches -- no alias through lineage,
// manifests or causality, and no identity a committed document pins -- and
// keeps the commit chain whole, so every gate trailer, merge receipt and
// census sequence that names a store commit stays exact. Evidence classes are
// never released however unreachable they look: consumers find them through
// lineage children and memo keys. It refuses to release unless a published
// backup seals the store's current head.
//
// With -dest it instead writes the live set into a fresh store. That is a
// lineage migration: the destination has new commit identities, and every
// commit-coordinate binding outside the store names commits it lacks.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/storepolicy"
)

func main() {
	clioptions.MainNamed("overgodb-compact", run)
}

func run() error {
	repository := flag.String("repo", "overgodb-store", "OvergoDB store")
	destination := flag.String("dest", "", "rewrite the live set into this new store instead of releasing in place")
	pinRoot := flag.String("pin-root", "", "checkout whose committed JSON documents pin store identities kept as roots")
	backup := flag.String("backup", "", "published backup whose seal must name the store's current head before an in-place release")
	flag.Parse()
	if flag.NArg() != 0 {
		return errors.New("usage: overgodb-compact [-repo <path>] [-pin-root <checkout>] (-backup <published-backup> | -dest <new-store>)")
	}
	var roots []artifact.ID
	if *pinRoot != "" {
		pinned, err := pinnedRoots(*pinRoot)
		if err != nil {
			return err
		}
		trailers, err := trailerRoots(*pinRoot)
		if err != nil {
			return err
		}
		roots = mergeRoots(pinned, trailers)
		fmt.Printf("roots: documents=%d trailers=%d union=%d\n", len(pinned), len(trailers), len(roots))
	}
	if strings.TrimSpace(*destination) != "" {
		return rewrite(*repository, *destination, roots)
	}
	if strings.TrimSpace(*backup) == "" {
		return errors.New("overgodb-compact: an in-place release requires -backup naming a published backup of this store")
	}
	return release(*repository, *backup, roots)
}

// release appends release commits for the unreachable derived caches and
// removes their blobs, once the backup seal proves the current chain is
// captured.
func release(repository, backup string, roots []artifact.ID) error {
	seal, err := overgodb.ReadBackupSeal(backup)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	head, sequence := store.Head()
	if seal.Head != head || seal.Sequence != sequence {
		return fmt.Errorf("overgodb-compact: backup %s seals head %s at sequence %d; the store is at %s sequence %d",
			backup, seal.Head, seal.Sequence, head, sequence)
	}
	policy := overgodb.RetentionPolicy{Roots: roots, FollowChildren: storepolicy.FollowChildren}
	report, err := overgodb.Release(context.Background(), store, runrecord.GateLifecycleAtRest(store), policy, storepolicy.DerivedCache)
	if err != nil {
		return err
	}
	fmt.Println(report.String())
	fmt.Printf("classes %s\n", strings.Join(storepolicy.DerivedCacheSchemas, " "))
	fmt.Printf("head %s -> %s\n", report.HeadBefore, report.HeadAfter)
	return nil
}

// rewrite writes the live set into a fresh store and leaves the source as is.
func rewrite(repository, destination string, roots []artifact.ID) error {
	source, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer source.Close()
	report, err := overgodb.CompactWithRoots(context.Background(), source, destination, runrecord.GateLifecycleAtRest(source), roots)
	if err != nil {
		return err
	}
	fmt.Println(report.String())
	fmt.Printf("pinned roots=%d missing=%d\n", report.PinnedRoots, report.PinnedMissing)
	fmt.Println("source untouched; the destination is a new commit lineage, not the store in service")
	return nil
}
