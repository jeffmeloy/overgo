// profile-catalog: architecture registry publication.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/scratchmodel"
)

func main() {
	clioptions.MainNamed("profile-catalog", func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("profile-catalog", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repository := flags.String("repo", "", "OvergoDB root")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("usage: profile-catalog [-repo <path>]")
	}
	root, err := dataroot.StoreRoot(*repository)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	publication, err := modelrecipe.PublishArchitectureProfileCatalog(context.Background(), store)
	if err != nil {
		return err
	}
	derivation, err := scratchmodel.PublishDerivationProfileCatalog(context.Background(), store)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "profiles registered=%d published=%d changed=%t commit=%s derivation=%s\n",
		publication.Coverage.Registered, publication.Coverage.Published,
		publication.Changed, publication.Commit, derivation.ID)
	return err
}
