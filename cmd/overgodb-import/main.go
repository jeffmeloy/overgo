// Command overgodb-import ingests a wire-format import stream into the
// catalog through streaming object storage.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"overgo/internal/objectstore"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbimport"
)

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, input io.Reader, output io.Writer) error {
	flags := flag.NewFlagSet("repodb-import", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repositoryPath := flags.String("repo", "", "OvergoDB root")
	artifactRoot := flags.String("root", "", "root for relative artifact paths")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *repositoryPath == "" || *artifactRoot == "" {
		return errors.New("usage: repodb-import -repo <path> -root <path> < export.jsonl")
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
