package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

// rehearsalStore is the name of a rehearsal copy in its row's scratch folder: not the store's own name, so the
// leftovers' walk does not take it for a store in service, and it carries no backup seal.
const rehearsalStore = "store"

// rehearse is overgodb rehearse: it clones the store in service into tmp/<row>/store of the checkout, for a measure
// that writes records to be rehearsed on a copy, as copies made by hand were (cp -r, about 8.5 GiB each, ten of them
// standing on 2026-11-09, each held by the leftovers' walk as a store that is never deleted). The clone's blobs are hard
// links to the store's, so it costs seconds and little space; its folder, named for the row, is the mark the tmp line
// reads, and it is finished when the row leaves the plan, as every scratch folder is. The row must be one the working
// plan holds, so a copy is never finished as soon as it is made.
func rehearse(flags *flag.FlagSet, args []string, output io.Writer) error {
	flags.SetOutput(io.Discard)
	repository := flags.String("repo", "", "OvergoDB root to copy; empty resolves via the data-root contract")
	row := flags.String("row", "", "the plan row the rehearsal is for; the copy is made in its scratch folder, tmp/<row>/store")
	checkout := flags.String("checkout", ".", "the checkout whose plan holds the row and whose tmp folder holds the copy")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *row == "" || filepath.Base(*row) != *row {
		return fmt.Errorf("usage: %s -row <plan row> [-repo <path>] [-checkout <path>]", flags.Name())
	}
	working, err := plan.Load(filepath.Join(*checkout, filepath.FromSlash(plan.Path)))
	if err != nil {
		return err
	}
	if !slices.ContainsFunc(working.Items, func(item plan.Item) bool { return strings.EqualFold(item.ID, *row) }) {
		return errors.New("overgodb rehearse: the working plan holds no row " + *row + ", so the copy would be finished as soon as it was made")
	}
	root, err := dataroot.StoreRoot(*repository)
	if err != nil {
		return err
	}
	destination := filepath.Join(*checkout, "tmp", *row, rehearsalStore)
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	report, err := overgodb.CloneLiveStore(ctx, root, destination)
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "rehearsal copy of %s for %s: %s, %d blobs linked, %d files copied (%d bytes); it is finished when the row leaves the plan\n",
		root, *row, destination, report.LinkedBlobs, report.CopiedFiles, report.CopiedBytes)
	return nil
}
