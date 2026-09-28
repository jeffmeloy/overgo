package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"

	"overgo/internal/overgodb"
)

// backup is overgodb backup: it publishes a verified copy of an OvergoDB
// store. The destination is written to a partial directory, replayed
// read-only, and renamed into place only when its head commit and sequence
// match the source. The store is the system of record for gate, run, and
// evaluation evidence; this is the owner that makes that evidence survive
// single-disk loss.
func backup(flags *flag.FlagSet, args []string, output io.Writer) error {
	flags.SetOutput(io.Discard)
	repository := flags.String("repo", "", "OvergoDB root to back up")
	destination := flags.String("dest", "", "destination directory (must not exist)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || strings.TrimSpace(*repository) == "" || strings.TrimSpace(*destination) == "" {
		return fmt.Errorf("usage: %s -repo <path> -dest <path>", flags.Name())
	}
	store, err := overgodb.OpenReadOnly(*repository)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	report, err := store.Backup(ctx, *destination)
	if err != nil {
		return fmt.Errorf("backup failed: store-file IO=%+v: %w", report, err)
	}
	fmt.Fprintf(output, "backup published: %s head=%s sequence=%d extent=%d; IO=%+v; audit: byte-verified copy, matching replay head; store-file counters exclude seal IO and replay\n",
		*destination, report.Head, report.Sequence, report.Extent, report)
	return nil
}
