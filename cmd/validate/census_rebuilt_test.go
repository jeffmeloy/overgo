package main

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestACensusIsTakenAgainOnARebuiltStore holds the acceptance of a census to
// a store that was rebuilt since the census was taken. A census records the
// store commit it was taken at and counts its verifications up to it, and a
// rebuild writes what it carries in commits of its own, so the rebuilt store
// holds the census's document and not its checkpoint: reading the census's
// verifications there fails for the checkpoint, although the registered
// models are the ones the census names.
//
// On the store the census was taken on, a call with the same models is
// refused and writes nothing. On the rebuilt store the same call with no
// reason says nothing would be written, and with a reason it publishes a
// census of the rebuilt store, which names the same models and a checkpoint
// that store holds, and writes a binding the acceptance tests' check passes.
func TestACensusIsTakenAgainOnARebuiltStore(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	sourcePath := t.TempDir()
	source, err := overgodb.Open(sourcePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = source.Close() })
	commit := strings.Repeat("c", 40)
	registerCensusModel(t, source, "frozen", true, true)
	base, err := buildCapabilityCensus(ctx, source, commit, catalogLimit, discovery.NewMemo())
	if err != nil {
		t.Fatal(err)
	}
	batch, err := capabilityCensusBatch(base)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, source, batch); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "binding.json")
	if err := os.WriteFile(path, []byte(`{"Base": "`+base.ID.String()+`"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := runArgs(ctx, []string{"-repo", sourcePath, "-check-census-binding", path}, io.Discard); err != nil {
		t.Fatalf("original census binding: %v", err)
	}
	if err := runArgs(ctx, []string{"-repo", sourcePath, "-check-census-binding", path, "-check-census-binding", path + ".missing"}, io.Discard); err == nil {
		t.Fatal("second missing census binding was ignored")
	}
	beforeBinding, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	rebuiltPath := filepath.Join(t.TempDir(), "rebuilt")
	if _, err := overgodb.Rebuild(ctx, source, rebuiltPath, nil, runrecord.RetiredGateAliasesAtRest, false); err != nil {
		t.Fatal(err)
	}
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}

	if err := acceptCensus(sourcePath, commit, path, "the same models", io.Discard); err == nil || !strings.Contains(err.Error(), "nothing to accept") {
		t.Fatalf("accepting the same models on the store the census was taken on = %v, want it refused: there is nothing to accept", err)
	}
	rebuilt, err := overgodb.OpenReadOnly(rebuiltPath)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := readCensusVerifications(ctx, rebuilt, base); !errors.Is(err, errCensusCheckpointAbsent) {
		t.Fatalf("reading the census's verifications on the rebuilt store = %v, want its checkpoint absent", err)
	}
	if err := rebuilt.Close(); err != nil {
		t.Fatal(err)
	}
	if err := runArgs(ctx, []string{"-repo", rebuiltPath, "-check-census-binding", path}, io.Discard); !errors.Is(err, errCensusCheckpointAbsent) {
		t.Fatalf("candidate binding: %v, want absent checkpoint", err)
	}
	afterBinding, err := os.ReadFile(path)
	if err != nil || string(afterBinding) != string(beforeBinding) {
		t.Fatalf("read-only binding check changed the file: %v", err)
	}
	var printed strings.Builder
	if err := acceptCensus(rebuiltPath, commit, path, "", &printed); err != nil || !strings.Contains(printed.String(), "nothing written") {
		t.Fatalf("a call with no reason on the rebuilt store = %v, printing %q, want nothing written and no refusal", err, printed.String())
	}
	if err := acceptCensus(rebuiltPath, commit, path, "the store was rebuilt", io.Discard); err != nil {
		t.Fatalf("accepting the census again on the rebuilt store: %v", err)
	}
	if err := runArgs(ctx, []string{"-repo", rebuiltPath, "-check-census-binding", path}, io.Discard); err != nil {
		t.Fatalf("reaccepted candidate binding: %v", err)
	}
	var binding acceptedCensus
	if err := jsonfile.DecodeStrict(path, &binding); err != nil {
		t.Fatal(err)
	}
	rebuilt, err = overgodb.OpenReadOnly(rebuiltPath)
	if err != nil {
		t.Fatal(err)
	}
	defer rebuilt.Close()
	if err := checkCensusDisposition(ctx, rebuilt, binding); err != nil {
		t.Fatalf("the binding written on the rebuilt store fails the acceptance tests' check: %v", err)
	}
	current, err := capabilityCensusCodec.Require(ctx, rebuilt, binding.Current)
	if err != nil {
		t.Fatal(err)
	}
	if binding.Base != base.ID || current.ID == base.ID || !sameCensusModels(base.Models, current.Models) {
		t.Fatalf("the census taken again = %+v over base %s, want another census of the same models", binding, base.ID)
	}
	if _, err := readCensusVerifications(ctx, rebuilt, current); err != nil {
		t.Fatalf("the census taken on the rebuilt store names a checkpoint it does not hold: %v", err)
	}
}
