package main

import (
	"cmp"
	"maps"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
)

// closureReview is valid only for its exact snapshot and store head.
// It shares one command's parsed sources and history; it is never persisted.
type closureReview struct {
	source             string
	head               artifact.CommitID
	sequence           uint64
	candidates         []closurescan.Candidate
	index              closurescan.RebindIndex
	documents, history []closureledger.Document
	aliases            map[string]artifact.ID
	recovery           closureRecoveryAnalysis
	projection         closureProjection
}

type closurePending struct {
	document      closureledger.Document
	previousAlias string
	reason        string
}
type closureProjection struct {
	rebound          []closureledger.Document
	resolved         map[artifact.ID]closureledger.Document
	retirements      []artifact.AliasBinding
	retired, claimed map[string]bool
	pending          []closurePending
	unmatched        int
	first            string
}

// projectActiveClosures computes the final active bindings without store writes.
// Import, inventory and admission share its exact successor decisions.
func projectActiveClosures(index closurescan.RebindIndex, candidates []closurescan.Candidate, documents []closureledger.Document, sourceAliases map[string]artifact.ID, sameStore, reviewCallsites bool, priorRetired map[string]bool) (closureProjection, error) {
	var rebound []closureledger.Document
	var retirements []artifact.AliasBinding
	retired := make(map[string]bool, len(priorRetired))
	maps.Copy(retired, priorRetired)
	resolved := map[artifact.ID]closureledger.Document{}
	unmatched, first := 0, ""
	claimedAliases := make(map[string]bool, len(sourceAliases))
	for alias := range sourceAliases {
		claimedAliases[alias] = true
	}
	var contentRetries []closurePending
	projectedClaims := map[string]bool{}
	for _, document := range documents {
		if len(document.Bindings) != 1 {
			unmatched++
			first = cmp.Or(first, document.Name+":bindings")
			continue
		}
		previousAlias, err := closureledger.ActiveAlias(document.Bindings[0])
		if err != nil || sourceAliases[previousAlias] != document.ID {
			unmatched++
			first = cmp.Or(first, document.Name+":source-alias")
			continue
		}
		current, matched, reason, err := rebindClosure(index, document, reviewCallsites)
		if err != nil {
			return closureProjection{}, err
		}
		if !matched {
			contentRetries = append(contentRetries, closurePending{
				document: document, previousAlias: previousAlias, reason: reason,
			})
			continue
		}
		resolved[document.ID] = current
		if sameStore && current.ID == document.ID {
			projectedClaims[previousAlias] = true
			continue
		}
		currentAlias, err := closureledger.ActiveAlias(current.Bindings[0])
		if err != nil {
			return closureProjection{}, err
		}
		claimedAliases[currentAlias] = true
		projectedClaims[currentAlias] = true
		if sameStore && currentAlias != previousAlias && !retired[previousAlias] {
			retirements = append(retirements, artifact.AliasBinding{Name: previousAlias, Target: document.ID, Previous: &document.ID, Remove: true})
			retired[previousAlias] = true
		}
		rebound = append(rebound, current)
	}
	// Content-matched recovery runs after every structural rebind has
	// claimed its alias, so the only free candidates left are genuinely
	// new sites; an offset-shifted successor with the same file, scope,
	// and exact value inherits the reviewed closure instead of being
	// retired and retyped.
	pending := contentRetries
	for {
		var remaining []closurePending
		progress := false
		for _, retry := range pending {
			document, previousAlias := retry.document, retry.previousAlias
			current, matched, _, err := closurescan.ContentMatchedRebind(
				document, candidates, func(alias string) bool { return claimedAliases[alias] },
			)
			if err != nil {
				return closureProjection{}, err
			}
			if matched {
				resolved[document.ID] = current
				currentAlias, err := closureledger.ActiveAlias(current.Bindings[0])
				if err != nil {
					return closureProjection{}, err
				}
				claimedAliases[currentAlias] = true
				projectedClaims[currentAlias] = true
				progress = true
				if sameStore && currentAlias != previousAlias && !retired[previousAlias] {
					retirements = append(retirements, artifact.AliasBinding{Name: previousAlias, Target: document.ID, Previous: &document.ID, Remove: true})
					retired[previousAlias] = true
				}
				rebound = append(rebound, current)
				continue
			}
			remaining = append(remaining, retry)
		}
		// Resolve against the state this transaction will publish. A completed
		// move releases its predecessor slot unless another successor claims it.
		for alias := range retired {
			if !projectedClaims[alias] && claimedAliases[alias] {
				delete(claimedAliases, alias)
				progress = true
			}
		}
		pending = remaining
		if !progress || len(pending) == 0 {
			break
		}
	}
	for _, retry := range pending {
		unmatched++
		first = cmp.Or(first, retry.document.Name+":"+retry.reason)
	}
	if err := requireUniqueClosureDocumentAliases(slices.Collect(maps.Values(resolved))); err != nil {
		return closureProjection{}, err
	}
	return closureProjection{rebound: rebound, resolved: resolved, retirements: retirements, retired: retired, claimed: claimedAliases, pending: pending, unmatched: unmatched, first: first}, nil
}
