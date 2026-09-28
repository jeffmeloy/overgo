package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"path/filepath"

	"overgo/internal/objectstore"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbimport"
)

// importStream is overgodb import: it ingests a wire-format import stream
// into the catalog through streaming object storage.
func importStream(flags *flag.FlagSet, args []string, input io.Reader, output io.Writer) error {
	flags.SetOutput(io.Discard)
	repositoryPath := flags.String("repo", "", "OvergoDB root")
	artifactRoot := flags.String("root", "", "root for relative artifact paths")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *repositoryPath == "" || *artifactRoot == "" {
		return fmt.Errorf("usage: %s -repo <path> -root <path> < export.jsonl", flags.Name())
	}
	store, err := overgodb.Open(*repositoryPath)
	if err != nil {
		return err
	}
	defer store.Close()
	objects, err := objectstore.New(filepath.Join(*repositoryPath, "objects"), store)
	if err != nil {
		return err
	}
	result, err := overgodbimport.ImportStored(context.Background(), store, *artifactRoot, objects, input)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "commit %s source %s artifacts %d\n", result.Commit, result.Source, len(result.Names))
	return err
}
