// dataset-catalog publishes the legacy dataset inventory into RepoDB.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
)

func main() {
	clioptions.MainNamed("dataset-catalog", func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("dataset-catalog", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repositoryPath := flags.String("repo", "", "OvergoDB root")
	legacyPath := flags.String("legacy", "", "legacy OvergoDB root")
	contentPath := flags.String("root", "", "dataset content root")
	register := flags.String("register", "", "register one directory dataset under this name (with -root)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("usage: dataset-catalog [-repo <path>] -legacy <path> -root <path> | -register <name> -root <dir>")
	}
	repositoryRoot := strings.TrimSpace(*repositoryPath)
	if repositoryRoot == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		repositoryRoot = roots.Store
	}
	contentRoot := filepath.Clean(strings.TrimSpace(*contentPath))
	if name := strings.TrimSpace(*register); name != "" {
		if contentRoot == "." {
			return errors.New("dataset-catalog: -register requires -root")
		}
		store, err := overgodb.Open(repositoryRoot)
		if err != nil {
			return err
		}
		defer store.Close()
		registered, err := dataset.RegisterDirectoryDataset(context.Background(), store, name, contentRoot)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(output, "registered %s files=%d bytes=%d changed=%t dataset=%s commit=%s\n",
			name, registered.Files, registered.Bytes, registered.Changed, registered.Dataset, registered.Commit)
		return err
	}
	legacyRoot := filepath.Clean(strings.TrimSpace(*legacyPath))
	if legacyRoot == "." || contentRoot == "." {
		return errors.New("dataset-catalog: legacy and content roots are required")
	}
	store, err := overgodb.Open(repositoryRoot)
	if err != nil {
		return err
	}
	defer store.Close()
	publication, err := dataset.PublishLegacyCatalog(context.Background(), store, legacyRoot, contentRoot)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "datasets registered=%d published=%d available=%d changed=%t commit=%s\n",
		publication.Coverage.Registered, publication.Coverage.Published, publication.Coverage.Available,
		publication.Changed, publication.Commit)
	return err
}
