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
	type structuralMatch struct {
		document, current           closureledger.Document
		previousAlias, currentAlias string
	}
	var matches []structuralMatch
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
		currentAlias, err := closureledger.ActiveAlias(current.Bindings[0])
		if err != nil {
			return Projection{}, err
		}
		matches = append(matches, structuralMatch{document, current, previousAlias, currentAlias})
	}
	// A literal removed beside an identical one in its scope falls back by
	// expression onto the survivor. Where the survivor's own decision stayed
	// put it keeps the alias; where every claimant moved, the survivor's
	// ordinal moved too and no claimant proves it owns the survivor, so the
	// alias is contested and each claimant retires unless content finds it
	// another successor.
	held, claimants := map[string]bool{}, map[string]int{}
	for _, match := range matches {
		claimants[match.currentAlias]++
		if match.currentAlias == match.previousAlias {
			held[match.currentAlias] = true
		}
	}
	contested := map[string]bool{}
	for _, match := range matches {
		document, current, previousAlias, currentAlias := match.document, match.current, match.previousAlias, match.currentAlias
		if currentAlias != previousAlias && (held[currentAlias] || claimants[currentAlias] > 1) {
			contested[currentAlias] = true
			contentRetries = append(contentRetries, Pending{
				Document: document, PreviousAlias: previousAlias, Reason: "contested",
			})
			continue
		}
		resolved[document.ID] = current
		if sameStore && current.ID == document.ID {
			projectedClaims[previousAlias] = true
			continue
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
				document, candidates, func(alias string) bool { return claimedAliases[alias] || contested[alias] },
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
	claims := map[string]closureledger.Document{}
	for _, document := range documents {
		for _, binding := range document.Bindings {
			alias, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return err
			}
			if prior, found := claims[alias]; found && prior.ID != document.ID {
				return fmt.Errorf(
					"closure-scan: conflicting reviewed decisions claim alias %s: %s and %s at %s:%d in %s",
					alias, prior.Name, document.Name, binding.File, binding.Line, binding.Scope,
				)
			}
			claims[alias] = document
		}
	}
	return nil
}
