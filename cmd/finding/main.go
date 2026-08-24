// finding records an out-of-scope observation in the typed OvergoDB finding
// register so an agent can park it without growing prompt state or drifting.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/finding"
	"overgo/internal/overgodb"
)

func main() { clioptions.MainNamed("finding", run) }

func run() error { return runArgs(os.Args[1:], os.Stdout) }

func runArgs(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("finding", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repository := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	title := flags.String("title", "", "concise observation")
	severity := flags.String("severity", "", "low, medium, or high")
	closure := flags.String("closure", "", "bounded implementation path")
	check := flags.String("check", "", "command or condition that fails while the finding remains true")
	var owners, evidence []string
	flags.Func("owner", "repeatable owner surface", func(value string) error { owners = append(owners, value); return nil })
	flags.Func("evidence", "repeatable observed evidence", func(value string) error { evidence = append(evidence, value); return nil })
	if err := flags.Parse(args); err != nil || flags.NArg() != 0 {
		return errors.New("usage: finding -title <text> -severity <low|medium|high> -owner <surface> -evidence <observation> -closure <path> -check <failable-check> [-repo <path>]")
	}
	document, batch, err := finding.NewTextBatch(*title, finding.Severity(*severity), owners, evidence, *closure, *check)
	if err != nil {
		return err
	}
	root := strings.TrimSpace(*repository)
	if root == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		root = roots.Store
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return err
	}
	fmt.Fprintf(output, "FINDING parked: %s\n", document.ID)
	return nil
}
