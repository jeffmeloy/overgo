package gate

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

const (
	packageReceiptSchema    = "overgo/gate-package-receipt/v1"
	packageReceiptMediaType = "application/vnd.overgo.gate-package-receipt+json"
	packageReceiptAlias     = "gate/package/"
)

// packageReceipt is the current verdict for an immutable input obligation.
// Revocation retains the earlier receipt and advances the alias by CAS.
type packageReceipt struct {
	Version    uint16      `json:"version"`
	ID         artifact.ID `json:"-"`
	Obligation artifact.ID `json:"obligation"`
	Previous   artifact.ID `json:"previous,omitzero"`
	Passed     bool        `json:"passed"`
	// Absent on historical receipts; those carry no named-test evidence.
	Tests map[string]string `json:"tests,omitempty"`
}

var packageReceiptCodec = artifact.JSONDocumentCodec(
	"gate package receipt", artifact.KindEvidence, packageReceiptMediaType, packageReceiptSchema,
	func(value *packageReceipt) error {
		if value.Version != artifact.InitialDocumentVersion || value.Obligation.Kind() != artifact.KindEvidence ||
			value.Previous.Valid() && value.Previous.Kind() != artifact.KindEvidence {
			return errors.New("gate package receipt: invalid binding")
		}
		passedTest := false
		for name, action := range value.Tests {
			if !value.Passed || strings.TrimSpace(name) == "" || action != "pass" && action != "skip" {
				return errors.New("gate package receipt: invalid named verdict")
			}
			passedTest = passedTest || action == "pass"
		}
		if len(value.Tests) != 0 && !passedTest {
			return errors.New("gate package receipt: named verdicts contain no passing test")
		}
		return nil
	}, func(value packageReceipt) artifact.ID { return value.ID },
	func(value *packageReceipt, id artifact.ID) { value.ID = id },
	func(value packageReceipt) packageReceipt { value.Tests = maps.Clone(value.Tests); return value },
)

type packageEvidenceLedger struct {
	store       *overgodb.Store
	environment artifact.ID
	obligations map[string]runrecord.AgentObligation
	// listings: the per-preparation obligations document each package's
	// obligation is listed in; a receipt descends from it.
	listings map[string]artifact.ID
	previous map[string]artifact.ID
	prepared map[string]runrecord.SelectionReuse
}

func (g *gateContext) openPackageEvidence() (*packageEvidenceLedger, error) {
	if g.retryCache == nil {
		cache := g.loadRetryCache()
		g.retryCache = &cache
	}
	store, err := g.openStore()
	if err != nil {
		return nil, err
	}
	return &packageEvidenceLedger{store: store, environment: g.environment.ID,
		obligations: map[string]runrecord.AgentObligation{}, listings: map[string]artifact.ID{}, previous: map[string]artifact.ID{}}, nil
}

// packageObligation is the one obligation a package's tests owe in a mode at an
// input identity in an environment: the receipt's key, computed the same way
// wherever a receipt is written or read.
func packageObligation(pkg, mode string, input, environment artifact.ID) (artifact.ID, runrecord.AgentObligation, error) {
	invocation, err := automationcheck.PackageInvocation(pkg, mode)
	if err != nil {
		return artifact.ID{}, runrecord.AgentObligation{}, err
	}
	obligation, err := runrecord.NewAgentObligation(runrecord.AgentObligation{
		Task: invocation, Name: "go-test", Scope: mode + ":" + pkg, Sources: []artifact.ID{input, environment},
	})
	return invocation, obligation, err
}

// prepare persists every required package before execution, then rebuilds the
// cache projection from exact store receipts. A tmp file is never authority.
func (ledger *packageEvidenceLedger) prepare(ctx context.Context, packages []string, mode string, inputs map[string]artifact.ID, cache *automationcheck.EvidenceCache) error {
	if ledger.prepared == nil {
		ledger.prepared = map[string]runrecord.SelectionReuse{}
	}
	store := ledger.store
	if err := store.Refresh(ctx); err != nil {
		return err
	}
	batch := artifact.Batch{}
	var entries []packageObligationEntry
	for _, pkg := range packages {
		invocation, obligation, err := packageObligation(pkg, mode, inputs[pkg], ledger.environment)
		if err != nil {
			return err
		}
		// The obligation is the receipt's key, computed here and listed in
		// the preparation's one document; it is not a record of its own.
		for _, id := range []artifact.ID{invocation, inputs[pkg], ledger.environment} {
			descriptor, found, err := store.Artifact(ctx, id)
			if err != nil {
				return err
			}
			if !found {
				descriptor = artifact.Descriptor{ID: id}
			}
			batch.Artifacts = append(batch.Artifacts, descriptor)
		}
		entries = append(entries, packageObligationEntry{Package: pkg, Mode: mode, Task: invocation, Input: inputs[pkg], Obligation: obligation.ID})
		ledger.obligations[pkg] = obligation
		prior, found, err := packageReceiptCodec.Resolve(ctx, store, packageReceiptAlias+obligation.ID.String())
		if err != nil {
			return err
		}
		delete(cache.Entries, invocation.String())
		delete(ledger.previous, pkg)
		witness := runrecord.SelectionReuse{Obligation: obligation.ID}
		if found {
			if prior.Obligation != obligation.ID {
				return errors.New("package evidence: receipt obligation mismatch")
			}
			ledger.previous[pkg] = prior.ID
			witness.Receipt, witness.Passed = prior.ID, prior.Passed
			if prior.Passed {
				if err := cache.RecordPackagePass(pkg, mode, inputs[pkg]); err != nil {
					return err
				}
			}
		}
		ledger.prepared[pkg] = witness
	}
	if len(entries) == 0 {
		return nil
	}
	slices.SortFunc(entries, func(a, b packageObligationEntry) int { return strings.Compare(a.Package, b.Package) })
	listing, err := packageObligationsCodec.NewInitial(packageObligations{Environment: ledger.environment, Entries: entries})
	if err != nil {
		return err
	}
	content, err := packageObligationsCodec.Content(listing)
	if err != nil {
		return err
	}
	parents := []artifact.ID{ledger.environment}
	for _, entry := range entries {
		parents = append(parents, entry.Task, entry.Input)
		ledger.listings[entry.Package] = listing.ID
	}
	batch.Key = packageObligationsAlias + listing.ID.String()
	batch.Contents = []artifact.Content{content}
	batch.Lineage = artifact.UniqueDependencyLineage(listing.ID, parents...)
	_, err = store.CommitAs(ctx, gateProducer, batch)
	return err
}

func (ledger *packageEvidenceLedger) record(ctx context.Context, pkg string, passed bool, tests map[string]string) error {
	obligation, found := ledger.obligations[pkg]
	if !found {
		return fmt.Errorf("package evidence: undeclared package %q", pkg)
	}
	// A verdict map carries a skip only for a declared exclusion: in a short
	// run one excluded by -short, in a complete run one that states why it
	// cannot apply here. An unowned skip leaves the package unpassed and
	// never reaches this record.
	previous := ledger.previous[pkg]
	receipt, err := packageReceiptCodec.NewInitial(packageReceipt{Obligation: obligation.ID, Previous: previous, Passed: passed, Tests: tests})
	if err != nil {
		return err
	}
	content, err := packageReceiptCodec.Content(receipt)
	if err != nil {
		return err
	}
	alias := artifact.AliasBinding{Name: packageReceiptAlias + obligation.ID.String(), Target: receipt.ID}
	// The receipt descends from the listing that names its obligation.
	parents := []artifact.ID{ledger.listings[pkg]}
	if previous.Valid() {
		alias = artifact.AliasMove(alias.Name, alias.Target, previous)
		parents = append(parents, previous)
	}
	_, err = ledger.store.CommitAs(ctx, gateProducer, artifact.Batch{
		Key: packageReceiptAlias + receipt.ID.String(), Contents: []artifact.Content{content},
		Lineage: artifact.DependencyLineage(receipt.ID, parents...), Aliases: []artifact.AliasBinding{alias},
	})
	if err == nil {
		ledger.previous[pkg] = receipt.ID
	}
	return err
}
