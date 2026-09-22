package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
)

// receiptVerb publishes a test receipt: the raw go test -json stream of the
// named packages, kept as evidence in the store. A reviewed reconciliation of
// an output-neutral source change binds such a receipt and names the tests it
// must show passing; the lane's own run keeps only a digest of the stream, so
// the receipt is a run of its own that keeps the bytes.
const receiptVerb = "receipt"

// receiptKeyPrefix is the batch key of a published receipt.
const receiptKeyPrefix = "test-lane/receipt/"

// receipt runs the packages and publishes the stream that proves them.
func receipt(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("test-lane receipt", flag.ContinueOnError)
	flags.SetOutput(stderr)
	repository := flags.String("repo", "", "OvergoDB store; empty resolves through the data-root contract")
	if err := flags.Parse(args); err != nil || flags.NArg() == 0 {
		fmt.Fprintln(stderr, "usage: test-lane receipt [-repo STORE] PACKAGE...")
		return 2
	}
	var stream bytes.Buffer
	run, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: append([]string{"test", "-json", "-count=1"}, flags.Args()...), Stdout: &stream, Stderr: stderr,
	})
	if err != nil || run.ExitCode != 0 {
		fmt.Fprintf(stderr, "test-lane receipt: go test failed (exit=%d): %v\n", run.ExitCode, err)
		return 1
	}
	id, err := publishReceipt(ctx, *repository, stream.Bytes())
	if err != nil {
		fmt.Fprintf(stderr, "test-lane receipt: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "test-lane receipt: published %s for %s\n", id, strings.Join(flags.Args(), " "))
	return 0
}

// publishReceipt keeps a complete go test -json stream as evidence and returns
// its identity. An incomplete stream -- a failure, a package that never
// finished -- is refused before anything is written: a receipt proves a pass.
func publishReceipt(ctx context.Context, repository string, stream []byte) (artifact.ID, error) {
	report, err := testevidence.GoTestJSONReport(string(stream))
	if err != nil {
		return artifact.ID{}, err
	}
	if err := testevidence.RequireComplete(report); err != nil {
		return artifact.ID{}, err
	}
	if report.PassedPackages == 0 {
		return artifact.ID{}, errors.New("no package passed; nothing to receipt")
	}
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, stream)
	if err != nil {
		return artifact.ID{}, err
	}
	root, err := dataroot.StoreRoot(repository)
	if err != nil {
		return artifact.ID{}, err
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return artifact.ID{}, err
	}
	defer store.Close()
	batch := artifact.Batch{
		Key:      receiptKeyPrefix + id.DigestHex(),
		Contents: []artifact.Content{{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(stream))}, Data: stream}},
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return artifact.ID{}, err
	}
	return id, nil
}
