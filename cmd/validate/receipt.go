package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
)

// receiptKeyPrefix is the batch key of a published receipt, the one
// receipts have been published under since they were first kept.
const receiptKeyPrefix = "test-lane/receipt/"

// runReceipt is validate -receipt: it publishes a test receipt, the raw go
// test -json stream of the go test arguments args, kept as evidence in the
// store repository names. A reviewed reconciliation of an output-neutral
// source change binds such a receipt and names the tests it must show
// passing, so the receipt keeps the bytes, not a digest of them.
func runReceipt(ctx context.Context, output io.Writer, repository string, args []string) error {
	if len(args) == 0 {
		return errors.New("usage: validate -receipt [-repo STORE] [--] [go test flags] PACKAGE...")
	}
	var stream bytes.Buffer
	run, err := processcontrol.Run(ctx, processcontrol.Command{Path: "go", Args: receiptArgs(args), Stdout: &stream, Stderr: output})
	if err != nil || run.ExitCode != 0 {
		return fmt.Errorf("validate -receipt: go test failed (exit=%d): %v", run.ExitCode, err)
	}
	id, err := publishReceipt(ctx, repository, stream.Bytes())
	if err != nil {
		return fmt.Errorf("validate -receipt: %w", err)
	}
	_, err = fmt.Fprintf(output, "validate -receipt: published %s for %s\n", id, strings.Join(args, " "))
	return err
}

// receiptArgs runs the packages one at a time: a receipt proves device tests,
// and parallel packages reserve the one GPU against each other. A -p the
// operator passes still wins, as go test keeps the last.
func receiptArgs(operator []string) []string {
	return append([]string{"test", "-json", "-count=1", "-p=1"}, operator...)
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
