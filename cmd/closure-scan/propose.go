package main

import (
	"context"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

// proposeTriageRows emits ready-to-apply triage rows for the named
// uncatalogued candidates with the scanner's own exact kind, file, scope,
// and line prefilled — the coordinates the reviewer previously hand-copied.
// A recoverable candidate arrives with its prior decision's text; a
// genuinely new one leaves only the understanding, closure path, and rerank
// trigger to the reviewer. A name the scan cannot find refuses with the
// current unclassified candidates rather than emitting a stale row.
func proposeTriageRows(
	ctx context.Context,
	snapshot repoanalysis.SourceSnapshot,
	store *overgodb.Store,
	names []string,
) (triageFile, error) {
	report, err := buildUnclassifiedPolicyReport(ctx, snapshot, store)
	if err != nil {
		return triageFile{}, err
	}
	index := make(map[string]unclassifiedCandidate, len(report.Candidates))
	current := make([]string, 0, len(report.Candidates))
	for _, candidate := range report.Candidates {
		index[candidate.Current.Name] = candidate
		current = append(current, candidate.Current.Name)
	}
	slices.Sort(current)
	proposal := triageFile{Rows: make([]triageRow, 0, len(names))}
	for _, name := range names {
		candidate, found := index[name]
		if !found {
			if identity, classified, err := identityCandidate(snapshot, name); err != nil {
				return triageFile{}, err
			} else if classified {
				return triageFile{}, refuseIdentityTriage(identity)
			}
			return triageFile{}, fmt.Errorf(
				"triage candidate %q not found by scan; current unclassified candidates: %s",
				name, strings.Join(current, ", "),
			)
		}
		if candidate.Proposed != nil {
			proposal.Rows = append(proposal.Rows, *candidate.Proposed)
			continue
		}
		proposal.Rows = append(proposal.Rows, triageRow{
			Kind: candidate.Current.Kind, Name: candidate.Current.Name,
			File: candidate.Current.File, Scope: candidate.Current.Scope, Line: candidate.Current.Line,
		})
	}
	return proposal, nil
}
