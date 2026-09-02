// finding records an out-of-scope observation in the typed OvergoDB finding
// register so an agent can park it without growing prompt state or drifting,
// and disposes a parked finding once it is fixed, refuted, or deferred.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/artifact"
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
	closeID := flags.String("close", "", "dispose a parked finding by id: publish its closed (or -status refuted|deferred) disposition citing -resolution")
	status := flags.String("status", string(finding.StatusClosed), "disposition status for -close: closed, refuted, or deferred")
	resolution := flags.String("resolution", "", "what resolved, refuted, or defers the finding (commit, owner decision, or measurement) for -close")
	var owners, evidence []string
	flags.Func("owner", "repeatable owner surface", func(value string) error { owners = append(owners, value); return nil })
	flags.Func("evidence", "repeatable observed evidence", func(value string) error { evidence = append(evidence, value); return nil })
	if err := flags.Parse(args); err != nil || flags.NArg() != 0 {
		return errors.New("usage: finding -title <text> -severity <low|medium|high> -owner <surface> -evidence <observation> -closure <path> -check <failable-check> [-repo <path>] | finding -close <finding-id> -resolution <text> [-status closed|refuted|deferred] [-repo <path>]")
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
	ctx := context.Background()
	var line string
	var batch artifact.Batch
	if strings.TrimSpace(*closeID) != "" {
		line, batch, err = dispositionBatch(ctx, store, *closeID, finding.Status(*status), *resolution)
	} else {
		var document finding.Document
		document, batch, err = finding.NewTextBatch(*title, finding.Severity(*severity), owners, evidence, *closure, *check)
		line = "FINDING parked: " + document.ID.String()
	}
	if err != nil {
		return err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	fmt.Fprintln(output, line)
	return nil
}

// dispositionBatch builds the closed, refuted, or deferred disposition of one
// parked finding; the original document stays as the observation that was
// made, and a finding takes one disposition.
func dispositionBatch(ctx context.Context, store *overgodb.Store, closeID string, status finding.Status, resolution string) (string, artifact.Batch, error) {
	var id artifact.ID
	if err := id.UnmarshalText([]byte(strings.TrimSpace(closeID))); err != nil {
		return "", artifact.Batch{}, err
	}
	content, found, err := artifact.ReadContent(ctx, store, id)
	if err != nil {
		return "", artifact.Batch{}, err
	}
	if !found {
		return "", artifact.Batch{}, errors.New("finding: the finding to dispose is not in the store")
	}
	original, err := finding.Parse(content.Data)
	if err != nil {
		return "", artifact.Batch{}, err
	}
	if _, disposed, err := finding.Disposition(ctx, store, id); err != nil {
		return "", artifact.Batch{}, err
	} else if disposed {
		return "", artifact.Batch{}, errors.New("finding: the finding already has a disposition")
	}
	document, batch, err := finding.NewDispositionBatch(ctx, store, original, status, resolution)
	if err != nil {
		return "", artifact.Batch{}, err
	}
	return fmt.Sprintf("FINDING %s: %s -> %s", status, id, document.ID), batch, nil
}
