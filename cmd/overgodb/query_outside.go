package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
)

// writeOutside lists the artifacts the store at repository records a local location of outside the repository (the
// checkout the command runs in) and the data roots, with no live copy inside them and no content of the store's own
// (overgodb.OutsideLocations): a line each, "outside" with its live copies where removing their folders would lose it,
// "lost" with its gone locations where none is left, then the count of each.
func writeOutside(output io.Writer, repository string) error {
	checkout, err := os.Getwd()
	if err != nil {
		return err
	}
	roots, err := dataroot.Resolve(checkout)
	if err != nil {
		return err
	}
	bounds := []string{checkout, filepath.Dir(roots.Store), roots.References()}
	for _, root := range []string{roots.Models, roots.Datasets, roots.Checkpoints, roots.StoreModels} {
		if root != "" {
			bounds = append(bounds, root)
		}
	}
	if roots.AudioReference != "" {
		bounds = append(bounds, filepath.Dir(roots.AudioReference))
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	listed, err := store.OutsideLocations(context.Background(), bounds)
	if err != nil {
		return err
	}
	lost := 0
	for _, entry := range listed {
		if entry.Lost() {
			lost++
			fmt.Fprintf(output, "lost\t%s\t%s\n", entry.Artifact, strings.Join(entry.Gone, " | "))
			continue
		}
		fmt.Fprintf(output, "outside\t%s\t%s\n", entry.Artifact, strings.Join(entry.Live, " | "))
	}
	_, err = fmt.Fprintf(output, "%d artifacts live only outside %s; %d with no copy left\n", len(listed)-lost, strings.Join(bounds, ", "), lost)
	return err
}
