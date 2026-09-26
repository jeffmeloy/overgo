// store-precheck proves a store maintenance operation on a candidate before
// it goes into service. It clones the sealed backup of the live store with
// hard-linked blobs, releases the unreachable derived caches on the clone,
// and runs the consumers that bind to the store against it: the plan
// completion authority, the published recipe chains and the closure ledger.
// It prints a typed receipt and, with -swap, renames the candidate into
// service only when every check passed, keeping the superseded store beside
// it. Two stores that passed hand-picked checks went into service and broke
// the plan authority within minutes; this command is the check that would
// have refused them, run by the operation rather than chosen by its operator.
package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/authoritylock"
	"overgo/internal/clioptions"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/processcontrol"
	"overgo/internal/repoanalysis"
	"overgo/internal/runrecord"
	"overgo/internal/storepolicy"
)

type check = overgodb.ConsumerCheck

// producer is this command's capability to commit a store operation proof:
// the store admits that kind from nothing else, so the gate never reads a
// proof someone typed.
var producer = overgodb.NewProducer("store-precheck")

// receipt is the proof the candidate carries plus what became of the
// candidate, emitted by the command that did it.
type receipt struct {
	overgodb.OperationProof
	Candidate  string `json:"candidate"`
	Swapped    bool   `json:"swapped"`
	Superseded string `json:"superseded,omitzero"`
}

func main() {
	clioptions.MainNamed("store-precheck", run)
}

func run() error {
	repository := flag.String("repo", "overgodb-store", "OvergoDB store in service")
	backup := flag.String("backup", "", "published backup whose seal names the store's current head (default: this command seals one under -backups)")
	backups := flag.String("backups", "", "directory this command seals its own backup under (default: overgo-backups beside the checkout)")
	checkout := flag.String("checkout", ".", "checkout whose plan, committed documents and commit messages the candidate must satisfy")
	candidate := flag.String("candidate", "", "fresh candidate store directory (default: <repo>.candidate)")
	swap := flag.Bool("swap", false, "rename the candidate into service when every check passed, keeping the superseded store beside it")
	leftovers := flag.Bool("leftovers", false, "list the superseded stores and sealed backups earlier operations left and which the newest of each are; removes nothing")
	flag.Parse()
	if flag.NArg() != 0 {
		return errors.New("usage: store-precheck [-backup <published-backup> | -backups <dir>] [-repo <store>] [-checkout <dir>] [-candidate <dir>] [-swap] | -leftovers")
	}
	absolute, err := filepath.Abs(*checkout)
	if err != nil {
		return err
	}
	*backups = cmp.Or(*backups, filepath.Join(filepath.Dir(absolute), "overgo-backups"))
	if *leftovers {
		found, err := listLeftovers(*repository, *backups)
		if err != nil {
			return err
		}
		return clioptions.WritePrettyJSON(os.Stdout, found)
	}
	// One hold of the gate's authority covers the seal, the candidate and the
	// swap: no landing can move the head the backup sealed.
	authority, err := authoritylock.Acquire(*checkout)
	if err != nil {
		return err
	}
	defer authority.Close()
	if strings.TrimSpace(*backup) == "" {
		if *backup, err = sealBackup(context.Background(), *repository, *backups); err != nil {
			return err
		}
	}
	*candidate = cmp.Or(*candidate, filepath.Clean(*repository)+".candidate")
	// The receipt prints whatever happened: a precheck that could not run,
	// a refused verdict and a swap each leave their record.
	result, err := precheck(context.Background(), *repository, *backup, *checkout, *candidate)
	if err != nil {
		return errors.Join(err, json.NewEncoder(os.Stdout).Encode(result))
	}
	err = conclude(&result, *swap, func() (string, error) { return swapStores(*repository, *candidate, result.HeadBefore) })
	return errors.Join(err, json.NewEncoder(os.Stdout).Encode(result))
}

// sealBackup seals a backup of the store in service under directory, named
// for the head and sequence it seals, so a repeat at the same head finds the
// one already there instead of copying the store again.
func sealBackup(ctx context.Context, repository, directory string) (string, error) {
	store, err := overgodb.Open(repository)
	if err != nil {
		return "", err
	}
	defer store.Close()
	head, sequence := store.Head()
	destination := filepath.Join(directory, fmt.Sprintf("overgodb-%.12s-%d", head, sequence))
	if seal, err := overgodb.ReadBackupSeal(destination); err == nil && seal.Head == head && seal.Sequence == sequence {
		return destination, nil
	}
	_, err = store.Backup(ctx, destination)
	return destination, err
}

// precheck builds the candidate and collects every consumer's verdict.
func precheck(ctx context.Context, repository, backup, checkout, candidate string) (receipt, error) {
	result := receipt{OperationProof: overgodb.OperationProof{Backup: backup}, Candidate: candidate}
	seal, err := overgodb.ReadBackupSeal(backup)
	if err != nil {
		return result, err
	}
	live, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return result, err
	}
	head, sequence := live.Head()
	if err := live.Close(); err != nil {
		return result, err
	}
	if seal.Head != head || seal.Sequence != sequence {
		return result, fmt.Errorf("store-precheck: backup %s seals head %s at sequence %d; the store in service is at %s sequence %d",
			backup, seal.Head, seal.Sequence, head, sequence)
	}
	result.HeadBefore, result.SequenceBefore = head.String(), sequence
	cloned, err := overgodb.CloneStore(backup, candidate)
	if err != nil {
		return result, err
	}
	result.LinkedBlobs = cloned.LinkedBlobs
	roots, _, _, err := storepolicy.CheckoutRoots(ctx, checkout)
	if err != nil {
		return result, err
	}
	store, err := overgodb.Open(candidate)
	if err != nil {
		return result, err
	}
	defer store.Close()
	released, err := overgodb.Release(ctx, store, runrecord.GateLifecycleAtRest(store),
		overgodb.RetentionPolicy{Roots: roots, FollowChildren: storepolicy.FollowChildren}, storepolicy.DerivedCache)
	if err != nil {
		return result, err
	}
	result.HeadAfter, result.SequenceAfter = released.HeadAfter.String(), released.SequenceAfter
	result.Roots, result.RootsMissing, result.Retained = released.PinnedRoots, released.PinnedMissing, released.Retained
	result.Unreachable, result.Released, result.ReleasedBytes = released.Unreachable, released.Released, released.ReleasedBytes
	result.BlobsRemoved = released.BlobsRemoved
	// The release decides what is live; the pack then takes the small blobs
	// that are, and the candidate is judged with its blobs where they will
	// be served from.
	if result.Pack, err = overgodb.PackSmallBlobs(ctx, store); err != nil {
		return result, err
	}
	result.Checks = runChecks([]namedCheck{
		{"blob-identity", func() error { _, err := store.VerifyBlobs(ctx); return err }},
		{"completion-authority", func() error { return checkCompletionAuthority(ctx, checkout, store) }},
		{"published-chains", func() error { return checkPublishedChains(ctx, checkout, candidate) }},
		{"closure-ledger", func() error { return checkClosureLedger(ctx, checkout, store) }},
	})
	// A candidate that passed carries its proof into service; one that did
	// not is discarded, so nothing is written to it.
	if !result.Verdict() {
		return result, nil
	}
	batch, err := result.Batch(ctx, store)
	if err != nil {
		return result, err
	}
	_, err = store.CommitAs(ctx, producer, batch)
	return result, err
}

type namedCheck struct {
	name string
	run  func() error
}

// runChecks runs every check: a failed one does not hide the next.
func runChecks(checks []namedCheck) []check {
	verdicts := make([]check, 0, len(checks))
	for _, each := range checks {
		verdict := check{Name: each.name, Passed: true}
		if err := each.run(); err != nil {
			verdict.Passed, verdict.Detail = false, err.Error()
		}
		verdicts = append(verdicts, verdict)
	}
	return verdicts
}

// conclude settles the verdict and swaps only on a clean one.
func conclude(result *receipt, swap bool, swapStores func() (string, error)) error {
	if !result.Verdict() {
		return errors.New("store-precheck: the candidate failed a consumer check; nothing was swapped")
	}
	if !swap {
		return nil
	}
	superseded, err := swapStores()
	result.Swapped, result.Superseded = err == nil, superseded
	return err
}

// swapStores renames the store in service aside and the candidate into its
// place; a failed second rename restores the first.
func swapStores(live, candidate, head string) (string, error) {
	superseded := fmt.Sprintf("%s.superseded-%.12s", filepath.Clean(live), head)
	if _, err := os.Lstat(superseded); err == nil {
		return "", fmt.Errorf("store-precheck: superseded path %q already exists", superseded)
	}
	if err := os.Rename(live, superseded); err != nil {
		return "", err
	}
	if err := os.Rename(candidate, live); err != nil {
		return "", errors.Join(err, os.Rename(superseded, live))
	}
	return superseded, nil
}

// checkCompletionAuthority resolves every landed completion against the candidate.
func checkCompletionAuthority(ctx context.Context, checkout string, store *overgodb.Store) error {
	document, err := plan.Load(filepath.Join(checkout, filepath.FromSlash(plan.Path)))
	if err != nil {
		return err
	}
	_, err = plan.ResolveCompletionAuthority(ctx, checkout, "HEAD", document, store)
	return err
}

// checkPublishedChains runs store-check, which refuses every broken chain, against the candidate.
func checkPublishedChains(ctx context.Context, checkout, candidate string) error {
	absolute, err := filepath.Abs(candidate)
	if err != nil {
		return err
	}
	var output bytes.Buffer
	outcome, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: []string{"run", "./cmd/store-check", "-repo", absolute}, Dir: checkout,
		Env: os.Environ(), Stdout: &output, Stderr: &output,
	})
	if err == nil && outcome.ExitCode != 0 {
		err = fmt.Errorf("store-check: %s", strings.TrimSpace(output.String()))
	}
	return err
}

// checkClosureLedger validates the candidate's active closure bindings against the checkout's sources.
func checkClosureLedger(ctx context.Context, checkout string, store *overgodb.Store) error {
	snapshot, err := repoanalysis.DiscoverGo(checkout, "internal", "cmd")
	if err != nil {
		return err
	}
	documents, aliases, err := closureledger.ActiveBindings(ctx, store)
	if err != nil {
		return err
	}
	_, err = closurescan.ValidatePermanentActiveAuthority(snapshot, documents, aliases)
	return err
}
