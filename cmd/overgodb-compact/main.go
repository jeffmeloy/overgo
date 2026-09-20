// overgodb-compact is retention's operator entry. It releases, in place, the
// bytes of the re-derivable cache documents (storepolicy.DerivedCache) that
// nothing reaches -- no alias through lineage, manifests or causality, no
// identity a committed document or a commit message pins, and no record the
// gate lifecycle discovers downward -- and keeps the commit chain whole, so
// every gate trailer, merge receipt and census sequence that names a store
// commit stays exact. Evidence classes are never released however unreachable
// they look. It refuses to release unless a published backup seals the
// store's current head. A store is never rewritten to compact it: a fresh
// journal is a new commit chain, and overgodb-rebuild owns that migration.
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
	pinRoot := flag.String("pin-root", "", "checkout whose committed JSON documents and commit messages pin store identities kept as roots")
	backup := flag.String("backup", "", "published backup whose seal must name the store's current head")
	flag.Parse()
	if flag.NArg() != 0 || strings.TrimSpace(*backup) == "" {
		return errors.New("usage: overgodb-compact -backup <published-backup> [-repo <path>] [-pin-root <checkout>]")
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
	seal, err := overgodb.ReadBackupSeal(*backup)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	head, sequence := store.Head()
	if seal.Head != head || seal.Sequence != sequence {
		return fmt.Errorf("overgodb-compact: backup %s seals head %s at sequence %d; the store is at %s sequence %d",
			*backup, seal.Head, seal.Sequence, head, sequence)
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
