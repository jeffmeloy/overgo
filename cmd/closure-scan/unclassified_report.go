package main

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

const (
	unclassifiedRecoverable = "recoverable-prior-decision"
	unclassifiedNew         = "genuinely-new"
	unclassifiedConflicting = "conflicting-prior-decisions"
	unclassifiedComposite   = "composite-prior-decision"

	historyMatchExact    = "exact"
	historyMatchReviewed = "reviewed-callsite"
)

type unclassifiedReportCounts struct {
	PolicyCandidates   int `json:"policy_candidates,omitzero"`
	Constants          int `json:"constants"`
	Literals           int `json:"literals,omitzero"`
	Assumptions        int `json:"assumptions,omitzero"`
	Classified         int `json:"classified"`
	Unclassified       int `json:"unclassified"`
	RecoverablePrior   int `json:"recoverable_prior_decisions"`
	ReviewedCallsite   int `json:"reviewed_callsite_decisions"`
	GenuinelyNew       int `json:"genuinely_new"`
	ConflictingPrior   int `json:"conflicting_prior_decisions"`
	CompositePrior     int `json:"composite_prior_decisions"`
	AutomaticRecovery  int `json:"automatic_recovery_eligible"`
	RecoveryCandidates int `json:"reactivation_candidates"`
	RebindUnverified   int `json:"rebind_operation_unverified_reactivations"`
	LegacyUnverified   int `json:"legacy_operation_unverified_reactivations"`
	IneligibleRecovery int `json:"lifecycle_ineligible_candidates"`
	TriageProposalRows int `json:"triage_proposal_rows"`
}

type historicalAliasEvent struct {
	Commit   string      `json:"commit"`
	Key      string      `json:"key"`
	Sequence uint64      `json:"sequence"`
	Target   artifact.ID `json:"target"`
	Remove   bool        `json:"remove"`
}

type historicalDecisionProvenance struct {
	Document         artifact.ID           `json:"document"`
	CurrentDocument  artifact.ID           `json:"current_document"`
	Match            string                `json:"match"`
	BindingCount     int                   `json:"binding_count"`
	Fixture          artifact.ID           `json:"pinning_fixture"`
	Alias            string                `json:"alias,omitzero"`
	LatestAlias      *historicalAliasEvent `json:"latest_alias_event,omitempty"`
	SelectedByAlias  bool                  `json:"selected_by_active_alias,omitzero"`
	RecoveryEligible bool                  `json:"automatic_recovery_eligible"`
	RecoveryBlocker  string                `json:"automatic_recovery_blocker,omitzero"`
}

type historicalDecisionMatch struct {
	Version       uint16                         `json:"version"`
	Name          string                         `json:"name"`
	Value         json.RawMessage                `json:"value"`
	Tier          closureledger.Tier             `json:"tier"`
	Status        closureledger.Status           `json:"status"`
	Understanding string                         `json:"understanding"`
	ClosurePath   string                         `json:"closure_path"`
	RerankTrigger string                         `json:"rerank_trigger"`
	Provenance    []historicalDecisionProvenance `json:"provenance"`
}

type unclassifiedCandidate struct {
	Current  closurescan.Candidate     `json:"current"`
	Category string                    `json:"category"`
	History  []historicalDecisionMatch `json:"history,omitempty"`
	Proposed *triageRow                `json:"proposed_triage_row,omitempty"`
}

type recoveredAliasAudit struct {
	Commit         string                  `json:"reactivation_commit"`
	Key            string                  `json:"reactivation_key"`
	Sequence       uint64                  `json:"reactivation_sequence"`
	Provenance     string                  `json:"provenance"`
	Eligible       bool                    `json:"eligible_under_current_successor_lifecycle"`
	Blocker        string                  `json:"blocker,omitzero"`
	ActiveBindings []recoveredAliasBinding `json:"active_bindings"`
}

type recoveredAliasBinding struct {
	Alias              string      `json:"alias"`
	ActiveTarget       artifact.ID `json:"active_target"`
	RetirementSequence uint64      `json:"retirement_sequence"`
	ActiveSequence     uint64      `json:"active_sequence"`
}

type unclassifiedReport struct {
	Version         uint16                   `json:"version"`
	Selection       string                   `json:"selection,omitzero"`
	Source          string                   `json:"source"`
	CatalogHead     string                   `json:"catalog_head"`
	CatalogSequence uint64                   `json:"catalog_sequence"`
	Counts          unclassifiedReportCounts `json:"counts"`
	Candidates      []unclassifiedCandidate  `json:"candidates"`
	RecoveryAudit   []recoveredAliasAudit    `json:"recovery_audit,omitempty"`
	TriageProposal  triageFile               `json:"triage_proposal"`
}

type unclassifiedSelection struct {
	name       string
	kinds      closurescan.CandidateKinds
	policyOnly bool
}

func buildUnclassifiedReport(
	ctx context.Context,
	snapshot repoanalysis.SourceSnapshot,
	store *overgodb.Store,
) (unclassifiedReport, error) {
	return buildUnclassifiedHistoryReport(ctx, snapshot, store, unclassifiedSelection{
		kinds: closurescan.CandidateConstants,
	})
}

func buildUnclassifiedPolicyReport(
	ctx context.Context,
	snapshot repoanalysis.SourceSnapshot,
	store *overgodb.Store,
) (unclassifiedReport, error) {
	return buildUnclassifiedHistoryReport(ctx, snapshot, store, unclassifiedSelection{
		name: "production-policy", kinds: closurescan.CandidateAll, policyOnly: true,
	})
}

func buildUnclassifiedHistoryReport(
	ctx context.Context,
	snapshot repoanalysis.SourceSnapshot,
	store *overgodb.Store,
	selection unclassifiedSelection,
) (unclassifiedReport, error) {
	candidates, err := closurescan.ScanSnapshot(snapshot, nil, selection.kinds)
	if err != nil {
		return unclassifiedReport{}, err
	}
	if selection.policyOnly {
		candidates = slices.DeleteFunc(candidates, func(candidate closurescan.Candidate) bool {
			return !candidate.Policy
		})
	}
	activeDocuments, activeAliases, err := activeClosureDocuments(ctx, store)
	if err != nil {
		return unclassifiedReport{}, err
	}
	historicalDocuments, err := allClosureDocuments(ctx, store)
	if err != nil {
		return unclassifiedReport{}, err
	}
	recoveryAnalysis, err := analyzeClosureRecovery(ctx, store, historicalDocuments)
	if err != nil {
		return unclassifiedReport{}, err
	}
	head, sequence := store.Head()
	report := unclassifiedReport{
		Version: artifact.InitialDocumentVersion, Selection: selection.name, Source: snapshot.Identity(),
		CatalogHead: head.String(), CatalogSequence: sequence,
	}
	recoveryGroups := map[string]int{}
	for _, reactivation := range recoveryAnalysis.Reactivations {
		report.Counts.RecoveryCandidates++
		if reactivation.Provenance == "rebind-operation-unverified" {
			report.Counts.RebindUnverified++
		} else {
			report.Counts.LegacyUnverified++
		}
		if !reactivation.Eligible {
			report.Counts.IneligibleRecovery++
		}
		key := fmt.Sprintf("%s\x00%s\x00%t\x00%s", reactivation.Commit.ID, reactivation.Provenance,
			reactivation.Eligible, reactivation.Blocker)
		groupIndex, found := recoveryGroups[key]
		if !found {
			groupIndex = len(report.RecoveryAudit)
			recoveryGroups[key] = groupIndex
			report.RecoveryAudit = append(report.RecoveryAudit, recoveredAliasAudit{
				Commit: reactivation.Commit.ID.String(), Key: reactivation.Commit.Key,
				Sequence: reactivation.Commit.Sequence, Provenance: reactivation.Provenance,
				Eligible: reactivation.Eligible, Blocker: reactivation.Blocker,
			})
		}
		report.RecoveryAudit[groupIndex].ActiveBindings = append(
			report.RecoveryAudit[groupIndex].ActiveBindings,
			recoveredAliasBinding{
				Alias: reactivation.Alias, ActiveTarget: reactivation.CurrentDocument,
				RetirementSequence: reactivation.RetirementSequence, ActiveSequence: reactivation.CurrentSequence,
			},
		)
	}
	slices.SortFunc(report.RecoveryAudit, func(left, right recoveredAliasAudit) int {
		if order := cmp.Compare(left.Sequence, right.Sequence); order != 0 {
			return order
		}
		if order := cmp.Compare(left.Commit, right.Commit); order != 0 {
			return order
		}
		return cmp.Compare(left.Blocker, right.Blocker)
	})
	if selection.policyOnly {
		report.Counts.PolicyCandidates = len(candidates)
	}
	for _, candidate := range candidates {
		switch candidate.Kind {
		case closureledger.BindingConstant:
			report.Counts.Constants++
		case closureledger.BindingLiteral:
			report.Counts.Literals++
		case closureledger.BindingAssumption:
			report.Counts.Assumptions++
		}
	}
	activeByID := make(map[artifact.ID]closureledger.Document, len(activeDocuments))
	for _, document := range activeDocuments {
		activeByID[document.ID] = document
	}
	bindingOwners := make(map[closureledger.SourceBinding]int, len(candidates))
	activeTargets := make(map[int]artifact.ID, len(candidates))
	unclassified := make(map[int]bool, len(candidates))
	for index, candidate := range candidates {
		binding, err := candidate.Binding()
		if err != nil {
			return unclassifiedReport{}, err
		}
		active, err := inventoryCandidateActive(activeByID, activeAliases, candidate, binding)
		if err != nil {
			return unclassifiedReport{}, err
		}
		if active {
			report.Counts.Classified++
			continue
		}
		bindingOwners[binding] = index
		alias, err := closureledger.ActiveAlias(binding)
		if err != nil {
			return unclassifiedReport{}, err
		}
		if target, found := activeAliases[alias]; found {
			activeTargets[index] = target
		}
		unclassified[index] = true
	}
	matches := make(map[int][]historicalDecisionMatch)
	matchGroups := make(map[int]map[string]int)
	index := closurescan.CompileRebindIndex(candidates)
	for _, document := range historicalDocuments {
		current, matched, _, err := rebindClosure(index, document, false)
		mode := historyMatchExact
		if err != nil {
			return unclassifiedReport{}, err
		}
		if !matched {
			current, matched, _, err = rebindClosure(index, document, true)
			mode = historyMatchReviewed
			if err != nil {
				return unclassifiedReport{}, err
			}
		}
		if !matched {
			continue
		}
		for _, binding := range current.Bindings {
			candidateIndex, found := bindingOwners[binding]
			if !found {
				continue
			}
			match := historicalDecisionMatch{
				Version: document.Version, Name: document.Name, Value: append(json.RawMessage(nil), document.Value...),
				Tier: document.Tier, Status: document.Status, Understanding: document.Understanding,
				ClosurePath: document.ClosurePath, RerankTrigger: document.RerankTrigger,
			}
			provenance := historicalDecisionProvenance{
				Document: document.ID, CurrentDocument: current.ID, Match: mode,
				BindingCount: len(document.Bindings), Fixture: document.Fixture,
				SelectedByAlias: activeTargets[candidateIndex] == document.ID,
			}
			if len(document.Bindings) == 1 {
				provenance.Alias, err = closureledger.ActiveAlias(document.Bindings[0])
				if err != nil {
					return unclassifiedReport{}, err
				}
				if event, found := recoveryAnalysis.History.latest[provenance.Alias]; found {
					provenance.LatestAlias = &historicalAliasEvent{
						Commit: event.Commit.ID.String(), Key: event.Commit.Key, Sequence: event.Commit.Sequence,
						Target: event.Binding.Target, Remove: event.Binding.Remove,
					}
				}
				provenance.RecoveryEligible = recoveryAnalysis.Authorized[provenance.Alias] == document.ID
				if disposition, found := recoveryAnalysis.Dispositions[provenance.Alias]; found && disposition.Target == document.ID {
					provenance.RecoveryBlocker = disposition.Blocker
				} else {
					provenance.RecoveryBlocker = automaticRecoveryBlocker(provenance)
				}
			} else {
				provenance.RecoveryBlocker = "composite-binding"
			}
			key := historicalDecisionKey(match)
			groups := matchGroups[candidateIndex]
			if groups == nil {
				groups = map[string]int{}
				matchGroups[candidateIndex] = groups
			}
			groupIndex, found := groups[key]
			if !found {
				groupIndex = len(matches[candidateIndex])
				groups[key] = groupIndex
				matches[candidateIndex] = append(matches[candidateIndex], match)
			}
			matches[candidateIndex][groupIndex].Provenance = append(
				matches[candidateIndex][groupIndex].Provenance, provenance,
			)
		}
	}
	for candidateIndex, candidate := range candidates {
		if !unclassified[candidateIndex] {
			continue
		}
		entry := unclassifiedCandidate{Current: candidate, History: matches[candidateIndex]}
		decisionHistory := entry.History
		if slices.ContainsFunc(entry.History, decisionSelectedByActiveAlias) {
			decisionHistory = slices.DeleteFunc(slices.Clone(entry.History), func(match historicalDecisionMatch) bool {
				return !decisionSelectedByActiveAlias(match)
			})
		}
		decisionKeys := map[string]bool{}
		composite := false
		exact := false
		automatic := false
		for _, match := range decisionHistory {
			decisionKeys[historicalDecisionKey(match)] = true
			for _, provenance := range match.Provenance {
				composite = composite || provenance.BindingCount != 1
				exact = exact || provenance.Match == historyMatchExact
				automatic = automatic || provenance.RecoveryEligible
			}
		}
		if automatic {
			report.Counts.AutomaticRecovery++
		}
		switch {
		case len(entry.History) == 0:
			entry.Category = unclassifiedNew
			report.Counts.GenuinelyNew++
		case len(decisionKeys) > 1:
			entry.Category = unclassifiedConflicting
			report.Counts.ConflictingPrior++
		case composite:
			entry.Category = unclassifiedComposite
			report.Counts.CompositePrior++
		default:
			entry.Category = unclassifiedRecoverable
			report.Counts.RecoverablePrior++
			if !exact {
				report.Counts.ReviewedCallsite++
			}
			latest := decisionHistory[len(decisionHistory)-1]
			row := triageRow{
				Kind: candidate.Kind, Name: candidate.Name, File: candidate.File, Scope: candidate.Scope, Line: candidate.Line,
				Tier: string(latest.Tier), Status: string(latest.Status), Understanding: latest.Understanding,
				ClosurePath: latest.ClosurePath, RerankTrigger: latest.RerankTrigger,
			}
			entry.Proposed = &row
			report.TriageProposal.Rows = append(report.TriageProposal.Rows, row)
		}
		report.Candidates = append(report.Candidates, entry)
	}
	report.Counts.Unclassified = len(report.Candidates)
	report.Counts.TriageProposalRows = len(report.TriageProposal.Rows)
	return report, nil
}

func decisionSelectedByActiveAlias(match historicalDecisionMatch) bool {
	return slices.ContainsFunc(match.Provenance, func(provenance historicalDecisionProvenance) bool {
		return provenance.SelectedByAlias
	})
}

func historicalDecisionKey(match historicalDecisionMatch) string {
	return fmt.Sprintf("%d\x00%s\x00%s\x00%s\x00%s\x00%s\x00%s\x00%s",
		match.Version, match.Name, match.Value, match.Tier, match.Status, match.Understanding,
		match.ClosurePath, match.RerankTrigger)
}

func inventoryCandidateActive(
	documents map[artifact.ID]closureledger.Document,
	aliases map[string]artifact.ID,
	candidate closurescan.Candidate,
	binding closureledger.SourceBinding,
) (bool, error) {
	alias, err := closureledger.ActiveAlias(binding)
	if err != nil {
		return false, err
	}
	target, found := aliases[alias]
	if !found {
		return false, nil
	}
	document, found := documents[target]
	if !found {
		return false, fmt.Errorf("closure inventory: active alias %s targets an absent closure document", alias)
	}
	_, found = activeCandidateClosure(map[string]closureledger.Document{candidate.DeclarationKey(): document}, candidate)
	return found, nil
}

func automaticRecoveryBlocker(match historicalDecisionProvenance) string {
	if match.RecoveryEligible {
		return ""
	}
	if match.LatestAlias == nil {
		return "no-alias-history"
	}
	if !match.LatestAlias.Remove {
		return "alias-currently-bound"
	}
	if match.LatestAlias.Target != match.Document {
		return "superseded-document"
	}
	if closureOperationKey(match.LatestAlias.Key, string(closureRetireUnmatchedOperation)) {
		return "explicit-retire-unmatched"
	}
	return "unproven-retirement"
}

func writeUnclassifiedReport(destination io.Writer, report unclassifiedReport) error {
	counts := report.Counts
	if _, err := fmt.Fprintf(destination,
		"closure-scan unclassified v%d source=%s catalog=%s@%d constants=%d classified=%d unclassified=%d recoverable=%d reviewed=%d new=%d conflicting=%d composite=%d automatic=%d reactivation_candidates=%d rebind_unverified=%d legacy_unverified=%d lifecycle_ineligible=%d triage=%d\n",
		report.Version, report.Source, report.CatalogHead, report.CatalogSequence,
		counts.Constants, counts.Classified, counts.Unclassified, counts.RecoverablePrior,
		counts.ReviewedCallsite, counts.GenuinelyNew, counts.ConflictingPrior, counts.CompositePrior, counts.AutomaticRecovery,
		counts.RecoveryCandidates, counts.RebindUnverified, counts.LegacyUnverified, counts.IneligibleRecovery,
		counts.TriageProposalRows,
	); err != nil {
		return err
	}
	return writeUnclassifiedCandidates(destination, report, false)
}

func writeUnclassifiedPolicyReport(destination io.Writer, report unclassifiedReport) error {
	counts := report.Counts
	if _, err := fmt.Fprintf(destination,
		"closure-scan unclassified-policy v%d source=%s catalog=%s@%d policy_candidates=%d constants=%d literals=%d assumptions=%d classified=%d unclassified=%d recoverable=%d reviewed=%d new=%d conflicting=%d composite=%d automatic=%d reactivation_candidates=%d rebind_unverified=%d legacy_unverified=%d lifecycle_ineligible=%d triage=%d\n",
		report.Version, report.Source, report.CatalogHead, report.CatalogSequence,
		counts.PolicyCandidates, counts.Constants, counts.Literals, counts.Assumptions,
		counts.Classified, counts.Unclassified, counts.RecoverablePrior,
		counts.ReviewedCallsite, counts.GenuinelyNew, counts.ConflictingPrior, counts.CompositePrior, counts.AutomaticRecovery,
		counts.RecoveryCandidates, counts.RebindUnverified, counts.LegacyUnverified, counts.IneligibleRecovery,
		counts.TriageProposalRows,
	); err != nil {
		return err
	}
	return writeUnclassifiedCandidates(destination, report, true)
}

func writeUnclassifiedCandidates(destination io.Writer, report unclassifiedReport, includeKind bool) error {
	for _, candidate := range report.Candidates {
		mode := "none"
		if len(candidate.History) != 0 {
			latest := candidate.History[len(candidate.History)-1].Provenance
			if len(latest) != 0 {
				mode = latest[len(latest)-1].Match
			}
		}
		var err error
		if includeKind {
			_, err = fmt.Fprintf(destination, "%s %s %s history=%d %s:%d %s=%s\n",
				candidate.Category, mode, candidate.Current.Kind, len(candidate.History), candidate.Current.File,
				candidate.Current.Line, candidate.Current.Name, candidate.Current.Value,
			)
		} else {
			_, err = fmt.Fprintf(destination, "%s %s history=%d %s:%d %s=%s\n",
				candidate.Category, mode, len(candidate.History), candidate.Current.File,
				candidate.Current.Line, candidate.Current.Name, candidate.Current.Value,
			)
		}
		if err != nil {
			return err
		}
	}
	return nil
}
