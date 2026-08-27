// overgodb-rebuild writes the store's projected state into a fresh
// journal, stripping derived caches whose identities stay registered:
// the store keeps recipes, prototypes, activations, and evaluated
// performance; it stops carrying recomputable snapshots (owner ruling
// 2026-08-27, measured: 6.07 GB of code manifests against ~450 MB of
// knowledge). -swap moves the rebuilt journal into place, keeping the
// old journal alongside; -discard-old removes the old journal after a
// verified swap.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
)

// strippedSchemas are derived-cache document schemas: recomputable
// from git at their recorded source identity, never model knowledge.
var strippedSchemas = map[string]bool{
	"overgo/code-manifest/v1":          true,
	"overgo/code-profile/v4":           true,
	"overgo/code-manifest-analysis/v1": true,
}

func main() {
	clioptions.MainNamed("overgodb-rebuild", run)
}

func run() error {
	repo := flag.String("repo", "overgodb-store", "OvergoDB store directory")
	destination := flag.String("destination", "tmp/store-rebuild", "directory for the rebuilt store; must not already hold a journal")
	swap := flag.Bool("swap", false, "move the rebuilt journal into the store directory, keeping the old journal alongside")
	discardOld := flag.Bool("discard-old", false, "with -swap: remove the old journal after the swap")
	flag.Parse()
	rebuiltExists := false
	if _, err := os.Stat(filepath.Join(*destination, "overgodb.log")); err == nil {
		rebuiltExists = true
	}
	if rebuiltExists && !*swap {
		return fmt.Errorf("overgodb-rebuild: %s already holds a journal; use a fresh destination", *destination)
	}
	if rebuiltExists && *swap {
		// A verified rebuilt journal already stands; swap it without
		// rebuilding again.
		return swapJournals(*repo, *destination, *discardOld)
	}
	source, err := overgodb.OpenReadOnly(*repo)
	if err != nil {
		return err
	}
	report, err := overgodb.Rebuild(context.Background(), source, *destination, func(descriptor artifact.Descriptor) bool {
		return strippedSchemas[descriptor.Schema]
	})
	closeErr := source.Close()
	if err != nil {
		return errors.Join(err, closeErr)
	}
	if closeErr != nil {
		return closeErr
	}
	fmt.Printf("rebuilt: artifacts=%d contents_kept=%d (%.2f GB) contents_dropped=%d (%.2f GB) manifests=%d lineage=%d locations=%d aliases=%d batches=%d\n",
		report.Artifacts, report.ContentsKept, float64(report.BytesKept)/1e9,
		report.ContentsDropped, float64(report.BytesDropped)/1e9,
		report.Manifests, report.LineageEdges, report.Locations, report.Aliases, report.Batches)
	rebuiltJournal := filepath.Join(*destination, "overgodb.log")
	if info, err := os.Stat(rebuiltJournal); err == nil {
		fmt.Printf("rebuilt journal: %s (%.2f GB)\n", rebuiltJournal, float64(info.Size())/1e9)
	}
	if !*swap {
		fmt.Println("re-run with -swap to move the rebuilt journal into place")
		return nil
	}
	return swapJournals(*repo, *destination, *discardOld)
}

func swapJournals(repo, destination string, discardOld bool) error {
	journal := filepath.Join(repo, "overgodb.log")
	rebuiltJournal := filepath.Join(destination, "overgodb.log")
	previous := journal + ".pre-rebuild"
	if _, err := os.Stat(previous); err == nil {
		return fmt.Errorf("overgodb-rebuild: %s already exists; resolve it before another swap", previous)
	}
	if err := os.Rename(journal, previous); err != nil {
		return err
	}
	if err := os.Rename(rebuiltJournal, journal); err != nil {
		return err
	}
	fmt.Printf("swapped: previous journal kept at %s\n", previous)
	if discardOld {
		if err := os.Remove(previous); err != nil {
			return err
		}
		fmt.Println("discarded the previous journal")
	}
	return nil
}
