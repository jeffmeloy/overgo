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
	"overgo/internal/repodb"
)

func main() {
	clioptions.MainNamed("dataset-catalog", func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("dataset-catalog", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repositoryPath := flags.String("repo", "", "RepoDB root")
	legacyPath := flags.String("legacy", "", "legacy RepoDB root")
	contentPath := flags.String("root", "", "dataset content root")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("usage: dataset-catalog [-repo <path>] -legacy <path> -root <path>")
	}
	repositoryRoot := strings.TrimSpace(*repositoryPath)
	if repositoryRoot == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		repositoryRoot = roots.Store
	}
	legacyRoot, contentRoot := filepath.Clean(strings.TrimSpace(*legacyPath)), filepath.Clean(strings.TrimSpace(*contentPath))
	if legacyRoot == "." || contentRoot == "." {
		return errors.New("dataset-catalog: legacy and content roots are required")
	}
	store, err := repodb.Open(repositoryRoot)
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
