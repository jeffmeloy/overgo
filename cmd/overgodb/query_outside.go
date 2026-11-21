package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
)

// locationBounds are the repository and the data roots: the checkout the command runs in, the store's home and its
// references, the models, datasets and checkpoints roots, the models beside the store and the audio reference store's
// home, as the checkout resolves them.
func locationBounds() ([]string, error) {
	checkout, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	roots, err := dataroot.Resolve(checkout)
	if err != nil {
		return nil, err
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
	return bounds, nil
}

// writeOutside lists the artifacts the store at repository records a local location of outside the repository and
// the data roots (locationBounds), with no live copy inside them and no content of the store's own
// (overgodb.OutsideLocations): a line each, "outside" with its live copies where removing their folders would lose it,
// "lost" with its gone locations where none is left, then the count of each.
func writeOutside(output io.Writer, repository string) error {
	bounds, err := locationBounds()
	if err != nil {
		return err
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

// writeRoots is overgodb roots: it records the repository and the data roots (locationBounds) as the roots the store at
// -repo keeps its local locations in (overgodb.WriteLocationRoots), so the store refuses one outside them (store-roots).
func writeRoots(flags *flag.FlagSet, args []string, output io.Writer) error {
	repository := flags.String("repo", "", "OvergoDB root")
	if err := flags.Parse(args); err != nil {
		return err
	}
	resolved, err := dataroot.StoreRoot(*repository)
	if err != nil {
		return err
	}
	bounds, err := locationBounds()
	if err != nil {
		return err
	}
	if err := overgodb.WriteLocationRoots(resolved, bounds); err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "%s keeps its local locations in %s\n", resolved, strings.Join(bounds, ", "))
	return err
}
