package gate

import (
	"context"
	"errors"
	"fmt"
	"maps"
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
	previous    map[string]artifact.ID
	prepared    map[string]runrecord.SelectionReuse
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
		obligations: map[string]runrecord.AgentObligation{}, previous: map[string]artifact.ID{}}, nil
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
	var identities []artifact.ID
	for _, pkg := range packages {
		invocation, err := automationcheck.PackageInvocation(pkg, mode)
		if err != nil {
			return err
		}
		obligation, err := runrecord.NewAgentObligation(runrecord.AgentObligation{
			Task: invocation, Name: "go-test", Scope: mode + ":" + pkg,
			Sources: []artifact.ID{inputs[pkg], ledger.environment},
		})
		if err != nil {
			return err
		}
		content, err := obligation.Content()
		if err != nil {
			return err
		}
		batch.Contents = append(batch.Contents, content)
		batch.Lineage = append(batch.Lineage, obligation.Lineage()...)
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
		identities = append(identities, obligation.ID)
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
	if len(identities) == 0 {
		return nil
	}
	key, err := artifact.JSONID(artifact.KindRecipe, identities)
	if err != nil {
		return err
	}
	batch.Key = packageReceiptAlias + key.String()
	_, err = artifact.CommitBatch(ctx, store, batch)
	return err
}

func (ledger *packageEvidenceLedger) record(ctx context.Context, pkg string, passed bool, tests map[string]string) error {
	obligation, found := ledger.obligations[pkg]
	if !found {
		return fmt.Errorf("package evidence: undeclared package %q", pkg)
	}
	for _, action := range tests {
		if action == "skip" && !strings.HasPrefix(obligation.Scope, "short:") {
			return errors.New("package evidence: complete profile cannot contain skipped tests")
		}
	}
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
	parents := []artifact.ID{obligation.ID}
	if previous.Valid() {
		alias.Previous = &previous
		parents = append(parents, previous)
	}
	_, err = artifact.CommitBatch(ctx, ledger.store, artifact.Batch{
		Key: packageReceiptAlias + receipt.ID.String(), Contents: []artifact.Content{content},
		Lineage: artifact.DependencyLineage(receipt.ID, parents...), Aliases: []artifact.AliasBinding{alias},
	})
	if err == nil {
		ledger.previous[pkg] = receipt.ID
	}
	return err
}
