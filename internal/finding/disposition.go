package finding

import (
	"context"
	"errors"
	"strings"

	"overgo/internal/artifact"
)

// DispositionAliasPrefix binds a finding's identity to its latest
// disposition: the closed or refuted document that supersedes it. A
// finding with no disposition alias is live.
const DispositionAliasPrefix = "finding/disposition/"

// GateAdvisoriesTitle names the finding the gate mints on every run from its
// actionable advisories.
const GateAdvisoriesTitle = "Actionable gate advisories"

// GateAdvisoriesAlias names the one live gate-advisory finding; older
// advisory documents are history, not open findings.
const GateAdvisoriesAlias = "finding/active/gate-advisories"

// NewDispositionBatch closes, refutes, or defers a parked finding: the superseding
// document keeps the finding's title, severity, owners, closure path, and
// check, cites the original and the resolution text as evidence, and binds
// the disposition alias so the ledger reads the finding as disposed.
// Findings are immutable, so a disposition is a new document, never an
// edit; the original stays readable as the observation that was made.
func NewDispositionBatch(ctx context.Context, reader artifact.Reader, original Document, status Status, resolution string) (Document, artifact.Batch, error) {
	if status != StatusClosed && status != StatusRefuted && status != StatusDeferred {
		return Document{}, artifact.Batch{}, errors.New("finding: a disposition is closed, refuted, or deferred")
	}
	if original.Status != StatusOpen || !original.ID.Valid() {
		return Document{}, artifact.Batch{}, errors.New("finding: only an open finding takes a disposition")
	}
	if strings.TrimSpace(resolution) == "" {
		return Document{}, artifact.Batch{}, errors.New("finding: a disposition names its resolution")
	}
	resolutionIDs, resolutionContents, err := textContents(artifact.KindEvidence, []string{resolution})
	if err != nil {
		return Document{}, artifact.Batch{}, err
	}
	evidence := append([]artifact.ID{original.ID}, resolutionIDs...)
	document, err := New(original.Title, original.Severity, status, original.OwnerSurfaces, evidence,
		original.ClosurePath, original.FailableCheck)
	if err != nil {
		return Document{}, artifact.Batch{}, err
	}
	content, err := document.Content()
	if err != nil {
		return Document{}, artifact.Batch{}, err
	}
	alias := artifact.AliasBinding{Name: DispositionAliasPrefix + original.ID.String(), Target: document.ID}
	if previous, found, err := artifact.ResolveAlias(ctx, reader, alias.Name); err != nil {
		return Document{}, artifact.Batch{}, err
	} else if found {
		alias.Previous = &previous
	}
	batch, err := artifact.NewDocumentBatch("finding/"+document.ID.String(),
		append(resolutionContents, content), document.Lineage(), []artifact.AliasBinding{alias})
	return document, batch, err
}

// Disposition reads the disposition bound to a finding, if any.
func Disposition(ctx context.Context, reader artifact.Reader, id artifact.ID) (Document, bool, error) {
	target, found, err := artifact.ResolveAlias(ctx, reader, DispositionAliasPrefix+id.String())
	if err != nil || !found {
		return Document{}, false, err
	}
	content, found, err := artifact.ReadContent(ctx, reader, target)
	if err != nil {
		return Document{}, false, err
	}
	if !found {
		return Document{}, false, errors.New("finding: the disposition alias names a document the store does not hold")
	}
	document, err := Parse(content.Data)
	return document, err == nil, err
}
