// Command overgodb-backup publishes a verified copy of an OvergoDB store: the
// destination is written to a partial directory, replayed read-only, and
// renamed into place only when its head commit and sequence match the source.
// The store is the system of record for gate, run, and evaluation evidence;
// this is the owner that makes that evidence survive single-disk loss.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
)

func main() {
	clioptions.Main(func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("overgodb-backup", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repository := flags.String("repo", "", "OvergoDB root to back up")
	destination := flags.String("dest", "", "destination directory (must not exist)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || strings.TrimSpace(*repository) == "" || strings.TrimSpace(*destination) == "" {
		return errors.New("usage: overgodb-backup -repo <path> -dest <path>")
	}
	store, err := overgodb.OpenReadOnly(*repository)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	head, sequence, err := store.Backup(*destination)
	if err != nil {
		return err
	}
	fmt.Fprintf(output, "backup published: %s head=%s sequence=%d; audit: replay-verified against the source head, not byte-compared\n",
		*destination, head, sequence)
	return nil
}
