package closurescan

import (
	"cmp"
	"fmt"
	"maps"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
)

// Pending is an active decision no candidate rebinds: its previous alias and
// the reason the last match failed.
type Pending struct {
	Document      closureledger.Document
	PreviousAlias string
	Reason        string
}

// Projection is the final active bindings a rebind would publish, computed
// without store writes: the rebound successors, each decision's resolved
// form, the alias removals, and the decisions that stay unmatched.
type Projection struct {
	Rebound          []closureledger.Document
	Resolved         map[artifact.ID]closureledger.Document
	Retirements      []artifact.AliasBinding
	Retired, Claimed map[string]bool
	Pending          []Pending
	Unmatched        int
	First            string
}

// ProjectActiveClosures computes the final active bindings without store
// writes. The import, the inventory, admission and the gate's preflight share
// its exact successor decisions.
func ProjectActiveClosures(index RebindIndex, candidates []Candidate, documents []closureledger.Document, sourceAliases map[string]artifact.ID, sameStore, reviewCallsites bool, priorRetired map[string]bool) (Projection, error) {
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
	var contentRetries []Pending
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
		current, matched, reason, err := index.rebind(document, reviewCallsites)
		if err != nil {
			return Projection{}, err
		}
		if !matched {
			contentRetries = append(contentRetries, Pending{
				Document: document, PreviousAlias: previousAlias, Reason: reason,
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
			return Projection{}, err
		}
		claimedAliases[currentAlias] = true
		projectedClaims[currentAlias] = true
		if sameStore && currentAlias != previousAlias && !retired[previousAlias] {
			retirements = append(retirements, artifact.AliasRemoval(previousAlias, document.ID))
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
		var remaining []Pending
		progress := false
		for _, retry := range pending {
			document, previousAlias := retry.Document, retry.PreviousAlias
			current, matched, _, err := ContentMatchedRebind(
				document, candidates, func(alias string) bool { return claimedAliases[alias] },
			)
			if err != nil {
				return Projection{}, err
			}
			if matched {
				resolved[document.ID] = current
				currentAlias, err := closureledger.ActiveAlias(current.Bindings[0])
				if err != nil {
					return Projection{}, err
				}
				claimedAliases[currentAlias] = true
				projectedClaims[currentAlias] = true
				progress = true
				if sameStore && currentAlias != previousAlias && !retired[previousAlias] {
					retirements = append(retirements, artifact.AliasRemoval(previousAlias, document.ID))
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
		first = cmp.Or(first, retry.Document.Name+":"+retry.Reason)
	}
	if err := RequireUniqueDocumentAliases(slices.Collect(maps.Values(resolved))); err != nil {
		return Projection{}, err
	}
	return Projection{Rebound: rebound, Resolved: resolved, Retirements: retirements, Retired: retired, Claimed: claimedAliases, Pending: pending, Unmatched: unmatched, First: first}, nil
}

// RequireUniqueDocumentAliases refuses two distinct decisions claiming one
// active alias.
func RequireUniqueDocumentAliases(documents []closureledger.Document) error {
	claims := map[string]artifact.ID{}
	for _, document := range documents {
		for _, binding := range document.Bindings {
			alias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return err
			}
			if prior, found := claims[alias]; found && prior != document.ID {
				return fmt.Errorf("closure-scan: conflicting reviewed decisions claim alias %s", alias)
			}
			claims[alias] = document.ID
		}
	}
	return nil
}
