package plan

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

const (
	completionItemTrailer              = "Overgo-Plan-Item"
	completionStepTrailer              = "Overgo-Plan-Step"
	completionVerifyTrailer            = "Overgo-Verify"
	completionManifestTrailer          = "Overgo-Manifest-Plan"
	completionCodeManifestTrailer      = "Overgo-Code-Manifest"
	completionPreparationTrailer       = "Overgo-Gate-Preparation"
	completionPreparationCommitTrailer = "Overgo-Gate-Preparation-Commit"
	completionSnapshotTrailer          = "Overgo-Plan-Item-Snapshot"
	completionItemIndexTrailer         = "Overgo-Plan-Item-Index"
	maximumCompletionParents           = 2
)

var completionTrailerKeys = [...]string{
	completionItemTrailer,
	completionStepTrailer,
	completionVerifyTrailer,
	completionManifestTrailer,
	completionCodeManifestTrailer,
	completionPreparationTrailer,
	completionPreparationCommitTrailer,
	completionSnapshotTrailer,
	completionItemIndexTrailer,
}

var legacyCompletionTrailerKeys = [...]string{
	completionItemTrailer,
	completionStepTrailer,
	completionVerifyTrailer,
	completionManifestTrailer,
	completionCodeManifestTrailer,
}

// CompletionCommitMessage appends the one canonical completion-trailer block
// used by both the gate writer and the authority reader. Reserved trailers in
// the operator message and multiline verifiers are refused so Git never has
// two competing interpretations of what completed.
func CompletionCommitMessage(
	operatorMessage []byte,
	preAdvance Plan,
	item, step string,
	manifest, codeManifest, preparation artifact.ID,
	preparationCommit artifact.CommitID,
) ([]byte, error) {
	if err := Validate(preAdvance); err != nil {
		return nil, fmt.Errorf("plan: invalid pre-advance completion plan: %w", err)
	}
	snapshot, itemIndex, completedStep, err := completionSnapshot(preAdvance, item, step)
	if err != nil {
		return nil, err
	}
	if manifest.Kind() != artifact.KindRecipe || codeManifest.Kind() != artifact.KindProfile ||
		preparation.Kind() != artifact.KindEvidence || !preparationCommit.Valid() {
		return nil, errors.New("plan: invalid completion commit authority")
	}
	snapshotJSON, err := json.Marshal(snapshot)
	if err != nil {
		return nil, fmt.Errorf("plan: encode completion item snapshot: %w", err)
	}
	snapshotText := base64.RawURLEncoding.EncodeToString(snapshotJSON)
	message := strings.ReplaceAll(string(operatorMessage), "\r\n", "\n")
	if strings.ContainsRune(message, '\x00') {
		return nil, errors.New("plan: completion message contains NUL")
	}
	for _, line := range strings.Split(message, "\n") {
		key, _, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		for _, reserved := range completionTrailerKeys {
			if strings.EqualFold(strings.TrimSpace(key), reserved) {
				return nil, fmt.Errorf("plan: operator message contains reserved trailer %s", reserved)
			}
		}
	}
	message = strings.TrimRight(message, "\n") + "\n\n" +
		completionItemTrailer + ": " + item + "\n" +
		completionStepTrailer + ": " + step + "\n" +
		completionManifestTrailer + ": " + manifest.String() + "\n" +
		completionCodeManifestTrailer + ": " + codeManifest.String() + "\n" +
		completionPreparationTrailer + ": " + preparation.String() + "\n" +
		completionPreparationCommitTrailer + ": " + preparationCommit.String() + "\n" +
		completionSnapshotTrailer + ": " + snapshotText + "\n" +
		completionItemIndexTrailer + ": " + strconv.Itoa(itemIndex) + "\n" +
		completionVerifyTrailer + ": " + completedStep.Verify + "\n"
	return []byte(message), nil
}

func completionSnapshot(document Plan, itemID, stepID string) (Item, int, Step, error) {
	if !validPlanID(itemID) || !validPlanID(stepID) {
		return Item{}, 0, Step{}, errors.New("plan: invalid completion plan reference")
	}
	for itemIndex, item := range document.Items {
		if item.ID != itemID {
			continue
		}
		for _, step := range item.Steps {
			if step.ID != stepID {
				continue
			}
			if step.Status != StatusOpen || step.Verify != strings.TrimSpace(step.Verify) ||
				!validAutomationDetail(step.Verify) {
				return Item{}, 0, Step{}, errors.New("plan: completion row is not exactly open and verifiable")
			}
			return item, itemIndex, step, nil
		}
		return Item{}, 0, Step{}, errors.New("plan: completion item lacks the exact step")
	}
	return Item{}, 0, Step{}, errors.New("plan: completion item is absent")
}

// CompletionAuthority is the derived, in-memory proof that exact pruned plan
// rows completed. It is deliberately opaque and has no JSON representation:
// Git remains the development record and OvergoDB remains the gate-attempt
// record, so plan.json never becomes a second completion ledger.
type CompletionAuthority struct {
	resolved            bool
	protectionSeeds     map[string]bool
	planDigest          [sha256.Size]byte
	repository          string
	revision            string
	completedReferences map[string]completionEvidence
	retiredItems        map[string]completionEvidence
}

// ProtectsRevision reports whether this opaque authority resolved an exact
// revision inside the prepared-format protection epoch. A gate uses it only
// after independently resolving each actual merge parent; every merge parent
// must be protected so pre-activation side history cannot enter later.
func (authority CompletionAuthority) ProtectsRevision() bool {
	return authority.resolved && len(authority.protectionSeeds) != 0 &&
		authority.repository != "" && authority.revision != ""
}

type completionEvidence struct {
	commit       string
	verify       string
	manifest     artifact.ID
	codeManifest artifact.ID
	attempt      artifact.ID
	retiredItem  bool
}

func (authority CompletionAuthority) completed(reference string) bool {
	_, found := authority.completedReferences[reference]
	return found
}

func (authority CompletionAuthority) resolves(document Plan) bool {
	if !authority.resolved || authority.repository == "" || authority.revision == "" {
		return false
	}
	digest, err := completionPlanDigest(document)
	return err == nil && digest == authority.planDigest
}

func (authority CompletionAuthority) resolvesAt(document Plan, repository, revision string) bool {
	if !authority.resolves(document) || strings.TrimSpace(revision) != authority.revision {
		return false
	}
	left, err := filepath.Abs(filepath.Clean(strings.TrimSpace(repository)))
	if err != nil {
		return false
	}
	right, err := filepath.Abs(filepath.Clean(authority.repository))
	if err != nil {
		return false
	}
	if runtime.GOOS == "windows" {
		return strings.EqualFold(left, right)
	}
	return left == right
}

func completionPlanDigest(document Plan) ([sha256.Size]byte, error) {
	encoded, err := json.Marshal(document)
	if err != nil {
		return [sha256.Size]byte{}, fmt.Errorf("plan: encode completion authority plan: %w", err)
	}
	return sha256.Sum256(encoded), nil
}

type completionTrailers struct {
	item, step, verify     string
	manifest               artifact.ID
	codeManifest           artifact.ID
	preparation            artifact.ID
	preparationCommit      artifact.CommitID
	snapshot               Item
	itemIndex              int
	snapshotPresent        bool
	declaresPreparedFormat bool
}

// VerifyCompletionCommitMessage proves that a commit message carries the
// canonical completion tuple and exact pre-advance item snapshot supplied by
// the caller. Recovery uses this before treating an interrupted Git commit as
// the gate's intended commit.
func VerifyCompletionCommitMessage(
	message string,
	preAdvance Plan,
	item, step string,
	manifest, codeManifest, preparation artifact.ID,
	preparationCommit artifact.CommitID,
) error {
	if err := Validate(preAdvance); err != nil {
		return fmt.Errorf("plan: invalid pre-advance completion plan: %w", err)
	}
	trailers, completion, err := parseCompletionTrailers(message)
	if err != nil {
		return err
	}
	if !completion || !trailers.snapshotPresent {
		return errors.New("plan: canonical prepared completion trailers are absent")
	}
	if trailers.item != item || trailers.step != step || trailers.manifest != manifest ||
		trailers.codeManifest != codeManifest || trailers.preparation != preparation ||
		trailers.preparationCommit != preparationCommit {
		return errors.New("plan: completion commit message differs from the expected authority tuple")
	}
	return verifyCompletionSnapshot(preAdvance, trailers)
}

// VerifyProspectiveCompletionTransition applies the history reader's exact
// transition policy before a gate publishes success. Parents must be the
// actual Git parent plan snapshots; a two-parent completion also supplies its
// actual merge-base plan. The prepared message must snapshot preAdvance, keep
// every baseline identity, and produce child by advancing exactly one row.
func VerifyProspectiveCompletionTransition(
	parents []Plan,
	mergeBase *Plan,
	preAdvance, child Plan,
	message string,
) error {
	if err := Validate(preAdvance); err != nil {
		return fmt.Errorf("plan: invalid prospective pre-advance plan: %w", err)
	}
	if err := Validate(child); err != nil {
		return fmt.Errorf("plan: invalid prospective child plan: %w", err)
	}
	baseline, err := prospectiveCompletionBaseline(parents, mergeBase)
	if err != nil {
		return err
	}
	trailers, completion, err := parseCompletionTrailers(message)
	if err != nil {
		return err
	}
	if !completion || !trailers.snapshotPresent {
		return errors.New("plan: prospective completion lacks canonical prepared trailers")
	}
	return verifyPreparedCompletionTransition(baseline, preAdvance, child, trailers)
}

// VerifyProspectiveMergeAuthority proves that a merged projection is safe to
// stage using completion authority derived from the exact two parent
// revisions in the same repository and authority store. It deliberately does
// not manufacture a new authority: the eventual gate must resolve both actual
// parents again while holding its commit transaction.
func VerifyProspectiveMergeAuthority(
	repository, localRevision, incomingRevision string,
	local, incoming, merged Plan,
	localAuthority, incomingAuthority CompletionAuthority,
) error {
	if err := Validate(merged); err != nil {
		return fmt.Errorf("plan: invalid prospective merge plan: %w", err)
	}
	if !localAuthority.resolvesAt(local, repository, localRevision) {
		return errors.New("plan: prospective merge lacks authority for the exact local parent")
	}
	if !incomingAuthority.resolvesAt(incoming, repository, incomingRevision) {
		return errors.New("plan: prospective merge lacks authority for the exact incoming parent")
	}
	if !localAuthority.ProtectsRevision() || !incomingAuthority.ProtectsRevision() {
		return errors.New("plan: prospective merge parent is outside the protected completion epoch")
	}
	commonProtection := false
	for seed := range localAuthority.protectionSeeds {
		if incomingAuthority.protectionSeeds[seed] {
			commonProtection = true
			break
		}
	}
	if !commonProtection {
		return errors.New("plan: prospective merge parents do not share a protected completion epoch")
	}
	if !preservesPlanIdentities(local, merged) || !preservesPlanIdentities(incoming, merged) {
		return errors.New("plan: prospective merge deleted a parent item or step")
	}

	completed := make(map[string]completionEvidence, len(localAuthority.completedReferences)+len(incomingAuthority.completedReferences))
	for reference, evidence := range localAuthority.completedReferences {
		completed[reference] = evidence
	}
	for reference, evidence := range incomingAuthority.completedReferences {
		if previous, found := completed[reference]; found && previous != evidence {
			return fmt.Errorf("plan: prospective merge has ambiguous completion authority for %s", reference)
		}
		completed[reference] = evidence
	}
	retired := make(map[string]completionEvidence, len(localAuthority.retiredItems)+len(incomingAuthority.retiredItems))
	for item, evidence := range localAuthority.retiredItems {
		retired[item] = evidence
	}
	for item, evidence := range incomingAuthority.retiredItems {
		if previous, found := retired[item]; found && previous != evidence {
			return fmt.Errorf("plan: prospective merge has ambiguous retirement authority for %s", item)
		}
		retired[item] = evidence
	}

	wanted, present, presentItems := completionNeeds(merged)
	for item := range presentItems {
		if evidence, reused := retired[item]; reused {
			return fmt.Errorf("plan: prospective merge reuses retired item %s from %.12s", item, evidence.commit)
		}
	}
	for reference := range present {
		if evidence, reused := completed[reference]; reused {
			return fmt.Errorf("plan: prospective merge reuses completed identity %s from %.12s", reference, evidence.commit)
		}
	}
	for reference := range wanted {
		if !present[reference] {
			if _, proven := completed[reference]; !proven {
				return fmt.Errorf("plan: prospective merge pruned dependency %s lacks gated completion evidence", reference)
			}
		}
	}
	return nil
}

func prospectiveCompletionBaseline(parents []Plan, mergeBase *Plan) (Plan, error) {
	parentCount := len(parents)
	if parentCount == 0 || parentCount > maximumCompletionParents {
		return Plan{}, fmt.Errorf("plan: prospective completion requires one or two parents, found %d", parentCount)
	}
	if parentCount < maximumCompletionParents {
		if mergeBase != nil {
			return Plan{}, errors.New("plan: one-parent completion must not supply a merge base")
		}
		if err := validatePlanGraph(parents[0]); err != nil {
			return Plan{}, fmt.Errorf("plan: invalid completion parent: %w", err)
		}
		return parents[0], nil
	}
	if mergeBase == nil {
		return Plan{}, errors.New("plan: two-parent completion requires one merge base")
	}
	baseline, err := MergeDocuments(*mergeBase, parents[0], parents[1])
	if err != nil {
		return Plan{}, fmt.Errorf(
			"plan: prospective completion merge source must be rebased onto the current protected plan: %w", err,
		)
	}
	return baseline, nil
}

func verifyCompletionSnapshot(preAdvance Plan, trailers completionTrailers) error {
	expectedItem, expectedIndex, expectedStep, err := completionSnapshot(
		preAdvance, trailers.item, trailers.step,
	)
	if err != nil {
		return err
	}
	if !trailers.snapshotPresent || trailers.itemIndex != expectedIndex || trailers.verify != expectedStep.Verify {
		return errors.New("plan: completion commit message differs from the expected plan row")
	}
	equal, err := exactCanonicalJSONEqual("completion item", trailers.snapshot, expectedItem)
	if err != nil {
		return err
	}
	if !equal {
		return errors.New("plan: completion commit item snapshot differs from the pre-advance plan")
	}
	return nil
}

func verifyPreparedCompletionTransition(baseline, preAdvance, child Plan, trailers completionTrailers) error {
	if err := verifyCompletionSnapshot(preAdvance, trailers); err != nil {
		return err
	}
	if !preservesPlanIdentities(baseline, preAdvance) {
		return errors.New("plan: completion pre-advance plan deleted a baseline item or step")
	}
	expected, err := Advance(preAdvance, trailers.item, trailers.step)
	if err != nil {
		return fmt.Errorf("plan: derive snapshotted completion transition: %w", err)
	}
	equal, err := exactCanonicalJSONEqual("completion plan", expected, child)
	if err != nil {
		return err
	}
	if !equal {
		return errors.New("plan: completion commit is not the exact plan transition")
	}
	return nil
}

func exactCanonicalJSONEqual(name string, left, right any) (bool, error) {
	leftBytes, err := json.Marshal(left)
	if err != nil {
		return false, fmt.Errorf("plan: encode left %s: %w", name, err)
	}
	rightBytes, err := json.Marshal(right)
	if err != nil {
		return false, fmt.Errorf("plan: encode right %s: %w", name, err)
	}
	return bytes.Equal(leftBytes, rightBytes), nil
}

// ResolveCompletionAuthority derives completion from the exact ancestor graph
// at revision and the successful gate attempts already committed to store. It
// validates both pruned dependencies and current identity reuse before
// returning; callers cannot manufacture authority from plan.json absence.
func ResolveCompletionAuthority(
	ctx context.Context,
	repository, revision string,
	document Plan,
	store *overgodb.Store,
) (CompletionAuthority, error) {
	if ctx == nil || store == nil {
		return CompletionAuthority{}, errors.New("plan: completion authority requires the store")
	}
	if err := Validate(document); err != nil {
		return CompletionAuthority{}, err
	}
	repository = strings.TrimSpace(repository)
	revision = strings.TrimSpace(revision)
	if repository == "" || revision == "" {
		return CompletionAuthority{}, errors.New("plan: completion authority requires a repository and revision")
	}
	repository, revision, err := resolveCompletionRevision(ctx, repository, revision)
	if err != nil {
		return CompletionAuthority{}, err
	}
	wanted, present, presentItems := completionNeeds(document)
	digest, err := completionPlanDigest(document)
	if err != nil {
		return CompletionAuthority{}, err
	}
	authority := CompletionAuthority{
		resolved:            true,
		planDigest:          digest,
		repository:          repository,
		revision:            revision,
		completedReferences: make(map[string]completionEvidence),
		retiredItems:        make(map[string]completionEvidence),
	}
	commits, err := gitCompletionMessages(ctx, repository, revision)
	if err != nil {
		return CompletionAuthority{}, err
	}
	parsed := make([]completionMessageParse, 0, len(commits))
	preparedSeeds := make(map[string]bool)
	for _, commit := range commits {
		trailers, hasCompletion, trailerErr := parseCompletionTrailers(commit.message)
		parsed = append(parsed, completionMessageParse{
			commit: commit, trailers: trailers, hasCompletion: hasCompletion, err: trailerErr,
		})
		if hasCompletion && trailers.declaresPreparedFormat {
			preparedSeeds[commit.hash] = true
		}
	}
	protected, protectionSeeds := protectedCompletionCommits(commits, preparedSeeds)
	protectedSet := make(map[string]bool, len(protected))
	for _, commit := range protected {
		protectedSet[commit.hash] = true
	}
	authority.protectionSeeds = protectionSeeds[revision]
	for _, commit := range protected {
		if len(commit.parents) < maximumCompletionParents {
			continue
		}
		commonSeeds := make(map[string]bool, len(protectionSeeds[commit.parents[0]]))
		for seed := range protectionSeeds[commit.parents[0]] {
			commonSeeds[seed] = true
		}
		for _, parent := range commit.parents[1:] {
			for seed := range commonSeeds {
				if !protectionSeeds[parent][seed] {
					delete(commonSeeds, seed)
				}
			}
		}
		if len(commonSeeds) == 0 {
			return CompletionAuthority{}, fmt.Errorf(
				"plan: protected merge %.12s has no common prepared ancestor; rebase the merge source onto the protected plan",
				commit.hash,
			)
		}
	}
	var candidates []completionCandidate
	for _, parsedMessage := range parsed {
		commit, trailers := parsedMessage.commit, parsedMessage.trailers
		hasCompletion, trailerErr := parsedMessage.hasCompletion, parsedMessage.err
		if !hasCompletion {
			continue
		}
		reference := trailers.item + "/" + trailers.step
		if protectedSet[commit.hash] && !trailers.declaresPreparedFormat {
			return CompletionAuthority{}, fmt.Errorf(
				"plan: completion %s at %.12s uses legacy trailers after prepared activation; rebase onto the protected plan",
				reference, commit.hash,
			)
		}
		relevant := wanted[reference] || presentItems[trailers.item]
		if trailerErr != nil {
			if trailers.declaresPreparedFormat || relevant {
				return CompletionAuthority{}, fmt.Errorf("plan: completion %s at %.12s: %w", reference, commit.hash, trailerErr)
			}
			continue
		}
		if !trailers.preparation.Valid() && !relevant {
			continue
		}
		candidates = append(candidates, completionCandidate{commit: commit, trailers: trailers, relevant: relevant})
	}
	transitionCommits := make([]gitCompletionMessage, 0, len(protected)+len(candidates))
	seenTransition := make(map[string]bool, len(protected)+len(candidates))
	for _, commit := range protected {
		transitionCommits = append(transitionCommits, commit)
		seenTransition[commit.hash] = true
	}
	for _, candidate := range candidates {
		if !seenTransition[candidate.commit.hash] {
			transitionCommits = append(transitionCommits, candidate.commit)
			seenTransition[candidate.commit.hash] = true
		}
	}
	transitions, err := completionTransitionPlans(ctx, repository, transitionCommits)
	if err != nil {
		return CompletionAuthority{}, err
	}
	validatedCompletions := make(map[string]bool, len(candidates))
	for _, candidate := range candidates {
		transition := transitions[candidate.commit.hash]
		reference := candidate.trailers.item + "/" + candidate.trailers.step
		evidence, err := requireCompletionEvidence(
			ctx, candidate.commit.hash, transition.baseline, transition.child, candidate.trailers, store,
		)
		if err != nil {
			return CompletionAuthority{}, fmt.Errorf("plan: completion %s at %.12s: %w", reference, candidate.commit.hash, err)
		}
		validatedCompletions[candidate.commit.hash] = true
		// Prepared-format completions activate the permanent identity ratchet.
		// Register them even when the current plan neither retains nor depends
		// on the row, so an add/complete cycle cannot hide duplicate identity
		// use by ending with the row absent. Legacy history remains scoped to
		// current relevance for backward compatibility.
		if !candidate.relevant && !candidate.trailers.preparation.Valid() {
			continue
		}
		if previous, duplicate := authority.completedReferences[reference]; duplicate {
			return CompletionAuthority{}, fmt.Errorf(
				"plan: completed identity %s is ambiguous at %.12s and %.12s",
				reference, previous.commit, candidate.commit.hash,
			)
		}
		authority.completedReferences[reference] = evidence
		if evidence.retiredItem {
			if previous, duplicate := authority.retiredItems[candidate.trailers.item]; duplicate {
				return CompletionAuthority{}, fmt.Errorf(
					"plan: retired item identity %s is ambiguous at %.12s and %.12s",
					candidate.trailers.item, previous.commit, candidate.commit.hash,
				)
			}
			authority.retiredItems[candidate.trailers.item] = evidence
		}
	}
	for _, commit := range protected {
		if validatedCompletions[commit.hash] {
			continue
		}
		transition := transitions[commit.hash]
		if !preservesPlanIdentities(transition.baseline, transition.child) {
			return CompletionAuthority{}, fmt.Errorf(
				"plan: protected commit %.12s deleted a plan item or step without gated completion", commit.hash,
			)
		}
	}
	if authority.ProtectsRevision() {
		transition, found := transitions[revision]
		if !found {
			return CompletionAuthority{}, errors.New("plan: protected revision transition is absent")
		}
		if !preservesPlanIdentities(transition.child, document) {
			return CompletionAuthority{}, fmt.Errorf(
				"plan: live plan deleted an item or step retained at protected revision %.12s", revision,
			)
		}
	}
	for itemID := range presentItems {
		if evidence, reused := authority.retiredItems[itemID]; reused {
			return CompletionAuthority{}, fmt.Errorf(
				"plan: retired item identity %s from %.12s cannot be reused", itemID, evidence.commit,
			)
		}
	}
	for reference := range present {
		if evidence, reused := authority.completedReferences[reference]; reused {
			return CompletionAuthority{}, fmt.Errorf(
				"plan: completed identity %s from %.12s cannot be reused", reference, evidence.commit,
			)
		}
	}
	for reference := range wanted {
		if present[reference] {
			continue
		}
		if !authority.completed(reference) {
			return CompletionAuthority{}, fmt.Errorf(
				"plan: pruned dependency %s lacks gated ancestor completion evidence", reference,
			)
		}
	}
	return authority, nil
}

func resolveCompletionRevision(ctx context.Context, repository, revision string) (string, string, error) {
	canonicalRepository, err := gitauthority.RepositoryRoot(ctx, repository)
	if err != nil {
		return "", "", fmt.Errorf("plan: resolve completion repository: %w", err)
	}
	if err := gitauthority.RequireCompleteHistory(ctx, canonicalRepository); err != nil {
		return "", "", fmt.Errorf("plan: %w", err)
	}
	commitOutput, err := gitCompletionCommand(
		ctx, canonicalRepository, "rev-parse", "--verify", "--end-of-options", revision+"^{commit}",
	)
	if err != nil {
		return "", "", fmt.Errorf("plan: resolve completion revision: %w", err)
	}
	commits := strings.Fields(string(commitOutput))
	if len(commits) != 1 {
		return "", "", fmt.Errorf("plan: completion revision resolved to %d commits", len(commits))
	}
	return canonicalRepository, commits[0], nil
}

func completionNeeds(document Plan) (map[string]bool, map[string]bool, map[string]bool) {
	present := make(map[string]bool)
	presentItems := make(map[string]bool)
	for _, item := range document.Items {
		presentItems[item.ID] = true
		for _, step := range item.Steps {
			present[item.ID+"/"+step.ID] = true
		}
	}
	wanted := make(map[string]bool, len(present))
	for reference := range present {
		wanted[reference] = true
	}
	for _, item := range document.Items {
		for _, step := range item.Steps {
			for _, reference := range step.DependsOn {
				if !present[reference] {
					wanted[reference] = true
				}
			}
		}
	}
	return wanted, present, presentItems
}

type gitCompletionMessage struct {
	hash, message string
	parents       []string
}

type completionCandidate struct {
	commit   gitCompletionMessage
	trailers completionTrailers
	relevant bool
}

type completionMessageParse struct {
	commit        gitCompletionMessage
	trailers      completionTrailers
	hasCompletion bool
	err           error
}

type completionTransition struct {
	baseline Plan
	child    Plan
}

// protectedCompletionCommits starts the no-raw-removal ratchet at each
// canonical prepared-format completion and carries it through every
// descendant. Older Git history remains readable because it predates the
// self-identifying snapshot format; after activation, every row removal must
// be an independently verified completion transition.
func protectedCompletionCommits(
	commits []gitCompletionMessage,
	preparedSeeds map[string]bool,
) ([]gitCompletionMessage, map[string]map[string]bool) {
	children := make(map[string][]string, len(commits))
	commitIndex := make(map[string]gitCompletionMessage, len(commits))
	indegree := make(map[string]int, len(commits))
	for _, commit := range commits {
		commitIndex[commit.hash] = commit
	}
	for _, commit := range commits {
		for _, parent := range commit.parents {
			if _, reachable := commitIndex[parent]; reachable {
				children[parent] = append(children[parent], commit.hash)
				indegree[commit.hash]++
			}
		}
	}
	protectionSeeds := make(map[string]map[string]bool, len(commits))
	queue := make([]string, 0, len(commits))
	for _, commit := range commits {
		if indegree[commit.hash] == 0 {
			queue = append(queue, commit.hash)
		}
	}
	for len(queue) != 0 {
		hash := queue[0]
		queue = queue[1:]
		seeds := make(map[string]bool)
		for _, parent := range commitIndex[hash].parents {
			for seed := range protectionSeeds[parent] {
				seeds[seed] = true
			}
		}
		// A later prepared completion remains in its earliest ancestor's
		// protection epoch. Mint a root only where no prepared ancestor is
		// reachable, keeping the common-epoch proof linear on normal history.
		if preparedSeeds[hash] && len(seeds) == 0 {
			seeds[hash] = true
		}
		if len(seeds) != 0 {
			protectionSeeds[hash] = seeds
		}
		for _, child := range children[hash] {
			indegree[child]--
			if indegree[child] == 0 {
				queue = append(queue, child)
			}
		}
	}
	result := make([]gitCompletionMessage, 0, len(protectionSeeds))
	for _, commit := range commits {
		if len(protectionSeeds[commit.hash]) != 0 {
			result = append(result, commit)
		}
	}
	return result, protectionSeeds
}

func gitCompletionMessages(ctx context.Context, repository, revision string) ([]gitCompletionMessage, error) {
	output, err := gitCompletionCommand(ctx, repository, "rev-list", revision)
	if err != nil {
		return nil, err
	}
	hashes := strings.Fields(string(output))
	if len(hashes) == 0 || hashes[0] != revision {
		return nil, errors.New("plan: Git completion history lacks its requested revision")
	}
	var input strings.Builder
	for _, hash := range hashes {
		if !validCommit(hash) {
			return nil, fmt.Errorf("plan: Git completion history contains invalid commit %q", hash)
		}
		input.WriteString(hash)
		input.WriteByte('\n')
	}
	objects, err := gitCompletionCommandInput(
		ctx, repository, strings.NewReader(input.String()), "cat-file", "--batch",
	)
	if err != nil {
		return nil, err
	}
	reader := bufio.NewReader(bytes.NewReader(objects))
	commits := make([]gitCompletionMessage, 0, len(hashes))
	commitIndex := make(map[string]bool, len(hashes))
	for _, hash := range hashes {
		data, err := readGitCompletionObject(reader, hash, "commit")
		if err != nil {
			return nil, err
		}
		parents, message, err := parseRawCompletionCommit(hash, data)
		if err != nil {
			return nil, err
		}
		commits = append(commits, gitCompletionMessage{hash: hash, parents: parents, message: message})
		commitIndex[hash] = true
	}
	for _, commit := range commits {
		for _, parent := range commit.parents {
			if !commitIndex[parent] {
				return nil, fmt.Errorf(
					"plan: raw Git completion parent %.12s of %.12s is absent from history",
					parent, commit.hash,
				)
			}
		}
	}
	return commits, nil
}

func parseRawCompletionCommit(hash string, object []byte) ([]string, string, error) {
	if bytes.IndexByte(object, 0) >= 0 {
		return nil, "", fmt.Errorf("plan: raw Git completion commit %.12s contains NUL", hash)
	}
	header, message, found := bytes.Cut(object, []byte("\n\n"))
	if !found {
		return nil, "", fmt.Errorf("plan: raw Git completion commit %.12s lacks its message separator", hash)
	}
	var parents []string
	for _, line := range bytes.Split(header, []byte{'\n'}) {
		if !bytes.HasPrefix(line, []byte("parent ")) {
			continue
		}
		parent := string(bytes.TrimPrefix(line, []byte("parent ")))
		if !validCommit(parent) {
			return nil, "", fmt.Errorf("plan: raw Git completion commit %.12s has invalid parent", hash)
		}
		parents = append(parents, parent)
	}
	return parents, string(message), nil
}

func parseCompletionTrailers(message string) (completionTrailers, bool, error) {
	message = strings.TrimSpace(strings.ReplaceAll(message, "\r\n", "\n"))
	if message == "" {
		return completionTrailers{}, false, nil
	}
	block := message
	if split := strings.LastIndex(block, "\n\n"); split >= 0 {
		block = block[split+2:]
	}
	known := map[string][]string{}
	hasCompletion := false
	noncanonical := ""
	for _, line := range strings.Split(block, "\n") {
		key, value, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		canonical := ""
		for _, candidate := range completionTrailerKeys {
			if strings.EqualFold(strings.TrimSpace(key), candidate) {
				canonical = candidate
				break
			}
		}
		if canonical == "" {
			continue
		}
		hasCompletion = true
		if strings.TrimSpace(key) != canonical && noncanonical == "" {
			noncanonical = strings.TrimSpace(key)
		}
		known[canonical] = append(known[canonical], strings.TrimSpace(value))
	}
	if !hasCompletion {
		return completionTrailers{}, false, nil
	}
	trailers := completionTrailers{declaresPreparedFormat: len(known[completionPreparationTrailer]) != 0 ||
		len(known[completionPreparationCommitTrailer]) != 0 || len(known[completionSnapshotTrailer]) != 0 ||
		len(known[completionItemIndexTrailer]) != 0}
	if values := known[completionItemTrailer]; len(values) != 0 {
		trailers.item = values[0]
	}
	if values := known[completionStepTrailer]; len(values) != 0 {
		trailers.step = values[0]
	}
	if noncanonical != "" {
		return trailers, true, fmt.Errorf("non-canonical trailer %q", noncanonical)
	}
	for _, key := range legacyCompletionTrailerKeys {
		if len(known[key]) != 1 || known[key][0] == "" {
			return trailers, true, fmt.Errorf("trailer %s must occur exactly once", key)
		}
	}
	if len(known[completionPreparationTrailer]) > 1 {
		return trailers, true, fmt.Errorf(
			"trailer %s must occur at most once", completionPreparationTrailer,
		)
	}
	trailers.item = known[completionItemTrailer][0]
	trailers.step = known[completionStepTrailer][0]
	trailers.verify = known[completionVerifyTrailer][0]
	if !validPlanID(trailers.item) || !validPlanID(trailers.step) ||
		trailers.verify != strings.TrimSpace(trailers.verify) || !validAutomationDetail(trailers.verify) {
		return trailers, true, errors.New("plan item, step, or verify trailer is invalid")
	}
	trailers.manifest, _ = artifact.ParseID(known[completionManifestTrailer][0])
	trailers.codeManifest, _ = artifact.ParseID(known[completionCodeManifestTrailer][0])
	if trailers.manifest.Kind() != artifact.KindRecipe || trailers.codeManifest.Kind() != artifact.KindProfile {
		return trailers, true, errors.New("manifest trailers have invalid identities")
	}
	if values := known[completionPreparationTrailer]; len(values) == 1 {
		trailers.preparation, _ = artifact.ParseID(values[0])
		if trailers.preparation.Kind() != artifact.KindEvidence {
			return trailers, true, errors.New("gate preparation trailer has an invalid identity")
		}
	}
	snapshotValues := known[completionSnapshotTrailer]
	indexValues := known[completionItemIndexTrailer]
	preparationCommitValues := known[completionPreparationCommitTrailer]
	if trailers.preparation.Valid() {
		if len(preparationCommitValues) != 1 || preparationCommitValues[0] == "" {
			return trailers, true, fmt.Errorf(
				"trailer %s must occur exactly once", completionPreparationCommitTrailer,
			)
		}
		var valid bool
		trailers.preparationCommit, valid = parseCompletionCommitID(preparationCommitValues[0])
		if !valid {
			return trailers, true, errors.New("gate preparation commit trailer has an invalid identity")
		}
		if len(snapshotValues) != 1 || snapshotValues[0] == "" {
			return trailers, true, fmt.Errorf("trailer %s must occur exactly once", completionSnapshotTrailer)
		}
		if len(indexValues) != 1 || indexValues[0] == "" {
			return trailers, true, fmt.Errorf("trailer %s must occur exactly once", completionItemIndexTrailer)
		}
		if err := decodeCompletionSnapshot(&trailers, snapshotValues[0], indexValues[0]); err != nil {
			return trailers, true, err
		}
	} else if len(preparationCommitValues) != 0 || len(snapshotValues) != 0 || len(indexValues) != 0 {
		return trailers, true, errors.New("prepared completion trailers require gate preparation authority")
	}
	return trailers, true, nil
}

func parseCompletionCommitID(text string) (artifact.CommitID, bool) {
	var id artifact.CommitID
	if len(text) != hex.EncodedLen(len(id)) {
		return artifact.CommitID{}, false
	}
	if _, err := hex.Decode(id[:], []byte(text)); err != nil || !id.Valid() || id.String() != text {
		return artifact.CommitID{}, false
	}
	return id, true
}

func decodeCompletionSnapshot(trailers *completionTrailers, snapshotText, indexText string) error {
	if trailers == nil {
		return errors.New("completion snapshot trailers are absent")
	}
	snapshotJSON, err := base64.RawURLEncoding.DecodeString(snapshotText)
	if err != nil {
		return errors.New("completion item snapshot is not raw base64url")
	}
	if base64.RawURLEncoding.EncodeToString(snapshotJSON) != snapshotText {
		return errors.New("completion item snapshot encoding is not canonical")
	}
	var snapshot Item
	if err := strictjson.DecodeBytes(snapshotJSON, &snapshot); err != nil {
		return fmt.Errorf("decode completion item snapshot: %w", err)
	}
	canonical, err := json.Marshal(snapshot)
	if err != nil || !bytes.Equal(canonical, snapshotJSON) {
		return errors.New("completion item snapshot JSON is not canonical")
	}
	if err := validatePlanGraph(Plan{Items: []Item{snapshot}}); err != nil {
		return fmt.Errorf("validate completion item snapshot: %w", err)
	}
	index, err := strconv.Atoi(indexText)
	if err != nil || index < 0 || strconv.Itoa(index) != indexText {
		return errors.New("completion item index is not canonical decimal")
	}
	step, found := exactPlanStep(Plan{Items: []Item{snapshot}}, trailers.item, trailers.step)
	if !found || snapshot.ID != trailers.item || step.Status != StatusOpen || step.Verify != trailers.verify {
		return errors.New("completion item snapshot differs from the completion row trailers")
	}
	trailers.snapshot = snapshot
	trailers.itemIndex = index
	trailers.snapshotPresent = true
	return nil
}

func requireCompletionEvidence(
	ctx context.Context,
	commit string,
	baseline, child Plan,
	trailers completionTrailers,
	store *overgodb.Store,
) (completionEvidence, error) {
	var err error
	var expected Plan
	if trailers.snapshotPresent {
		preAdvance, reconstructionErr := reconstructCompletionPrePlan(child, trailers)
		if reconstructionErr != nil {
			return completionEvidence{}, reconstructionErr
		}
		if err := verifyPreparedCompletionTransition(baseline, preAdvance, child, trailers); err != nil {
			return completionEvidence{}, err
		}
	} else {
		parentStep, retained := exactPlanStep(baseline, trailers.item, trailers.step)
		if !retained {
			return completionEvidence{}, errors.New("parent plan lacks the exact completion row")
		}
		if parentStep.Status != StatusOpen {
			return completionEvidence{}, errors.New("parent plan completion row is not open")
		}
		if trailers.verify != parentStep.Verify {
			return completionEvidence{}, errors.New("verify trailer differs from the parent plan step")
		}
		expected, err = advanceHistoricalPlan(baseline, trailers.item, trailers.step)
		if err != nil {
			return completionEvidence{}, fmt.Errorf("derive exact completion transition: %w", err)
		}
		if equal, err := exactCanonicalJSONEqual("completion plan", expected, child); err != nil {
			return completionEvidence{}, err
		} else if !equal {
			return completionEvidence{}, errors.New("completion commit is not the exact plan transition")
		}
	}

	var attempts []runrecord.AttemptRecord
	if trailers.preparation.Valid() {
		introduction, found, introductionErr := store.ArtifactIntroduction(ctx, trailers.preparation)
		if introductionErr != nil {
			return completionEvidence{}, fmt.Errorf("resolve gate preparation introduction: %w", introductionErr)
		}
		if !found {
			return completionEvidence{}, errors.New("gate preparation has no durable introduction authority")
		}
		if introduction.Commit != trailers.preparationCommit {
			return completionEvidence{}, errors.New("gate preparation commit trailer differs from its introduction authority")
		}
		attempts, err = runrecord.AttemptsForPreparation(ctx, store, trailers.preparation)
	} else {
		attempts, err = runrecord.AttemptsForRecipe(ctx, store, trailers.manifest)
	}
	if err != nil {
		return completionEvidence{}, fmt.Errorf("resolve completion attempts: %w", err)
	}
	var matched []runrecord.AttemptRecord
	for _, attempt := range attempts {
		if attempt.PlanItem == trailers.item && attempt.PlanStep == trailers.step &&
			attempt.CodeCommit == commit && attempt.Outcome == runrecord.OutcomeSucceeded {
			matched = append(matched, attempt)
		}
	}
	if len(matched) != 1 {
		return completionEvidence{}, fmt.Errorf("requires exactly one successful gate attempt, found %d", len(matched))
	}
	attempt := matched[0]
	if attempt.Recipe != trailers.manifest || attempt.CandidateManifest != trailers.codeManifest {
		return completionEvidence{}, errors.New("attempt authorities differ from the manifest trailers")
	}
	verification, err := runrecord.VerifyAttemptGate(ctx, store, attempt)
	if err != nil {
		return completionEvidence{}, fmt.Errorf("verify attempt gate: %w", err)
	}
	if trailers.preparation.Valid() {
		if verification.Preparation.ID != trailers.preparation {
			return completionEvidence{}, errors.New("attempt gate differs from the preparation trailer")
		}
		if err := requireCompletionManifestBinding(ctx, store, verification, attempt); err != nil {
			return completionEvidence{}, err
		}
		if err := requireCompletionAcceptance(
			verification.Gate, trailers.item+"/"+trailers.step, trailers.verify,
		); err != nil {
			return completionEvidence{}, err
		}
	}
	return completionEvidence{
		commit: commit, verify: trailers.verify, manifest: trailers.manifest,
		codeManifest: trailers.codeManifest, attempt: attempt.ID,
		retiredItem: !planHasItem(child, trailers.item),
	}, nil
}

func requireCompletionManifestBinding(
	ctx context.Context,
	store *overgodb.Store,
	verification runrecord.AttemptGateVerification,
	attempt runrecord.AttemptRecord,
) error {
	edges, err := store.Parents(ctx, verification.Gate.ID)
	if err != nil {
		return err
	}
	var analyses []automationcheck.ManifestAnalysis
	for _, edge := range edges {
		if edge.Child != verification.Gate.ID || edge.Relation != artifact.RelationDependsOn {
			continue
		}
		content, found, readErr := artifact.ReadContent(ctx, store, edge.Parent)
		if readErr != nil {
			return readErr
		}
		if !found || content.Descriptor.MediaType != automationcheck.ManifestAnalysisMediaType ||
			content.Descriptor.Schema != automationcheck.ManifestAnalysisSchema {
			continue
		}
		analysis, parseErr := automationcheck.ParseManifestAnalysis(content.Data)
		if parseErr != nil {
			return parseErr
		}
		if analysis.ID != edge.Parent {
			return errors.New("completion manifest analysis identity differs from its lineage")
		}
		analyses = append(analyses, analysis)
	}
	if len(analyses) != 1 {
		return fmt.Errorf("completion requires exactly one manifest analysis, found %d", len(analyses))
	}
	analysis := analyses[0]
	analysisIntroduction, found, err := store.ArtifactIntroduction(ctx, analysis.ID)
	if err != nil {
		return fmt.Errorf("resolve completion manifest analysis introduction: %w", err)
	}
	if !found {
		return errors.New("completion manifest analysis has no durable introduction authority")
	}
	attemptIntroduction, found, err := store.ArtifactIntroduction(ctx, attempt.ID)
	if err != nil {
		return fmt.Errorf("resolve completion attempt introduction: %w", err)
	}
	if !found {
		return errors.New("completion attempt has no durable introduction authority")
	}
	if analysisIntroduction.Commit != attemptIntroduction.Commit ||
		analysisIntroduction.Sequence != attemptIntroduction.Sequence {
		return errors.New("completion manifest analysis was not published atomically with the attempt authority")
	}
	manifest := analysis.Plan
	if manifest.ID != attempt.Recipe || manifest.CandidateManifest != attempt.CandidateManifest ||
		manifest.CandidateTree != verification.Preparation.TreeKey {
		return errors.New("completion manifest analysis differs from the attempt or prepared candidate")
	}
	return nil
}

func reconstructCompletionPrePlan(child Plan, trailers completionTrailers) (Plan, error) {
	if !trailers.snapshotPresent || trailers.itemIndex < 0 {
		return Plan{}, errors.New("completion item snapshot is absent")
	}
	preAdvance := child
	preAdvance.Items = slices.Clone(child.Items)
	if len(trailers.snapshot.Steps) == 1 {
		if trailers.itemIndex > len(preAdvance.Items) {
			return Plan{}, errors.New("completion item index is outside the child plan")
		}
		preAdvance.Items = slices.Insert(preAdvance.Items, trailers.itemIndex, trailers.snapshot)
	} else {
		if trailers.itemIndex >= len(preAdvance.Items) {
			return Plan{}, errors.New("completion item index is outside the child plan")
		}
		preAdvance.Items[trailers.itemIndex] = trailers.snapshot
	}
	if err := Validate(preAdvance); err != nil {
		return Plan{}, fmt.Errorf("validate reconstructed pre-advance plan: %w", err)
	}
	return preAdvance, nil
}

func preservesPlanIdentities(baseline, candidate Plan) bool {
	for _, baselineItem := range baseline.Items {
		candidateIndex := slices.IndexFunc(candidate.Items, func(item Item) bool {
			return item.ID == baselineItem.ID
		})
		if candidateIndex < 0 {
			return false
		}
		candidateItem := candidate.Items[candidateIndex]
		for _, baselineStep := range baselineItem.Steps {
			if !slices.ContainsFunc(candidateItem.Steps, func(step Step) bool {
				return step.ID == baselineStep.ID
			}) {
				return false
			}
		}
	}
	return true
}

func completionTransitionPlans(
	ctx context.Context,
	repository string,
	commits []gitCompletionMessage,
) (map[string]completionTransition, error) {
	revisions := make([]string, 0, 3*len(commits))
	seenRevisions := make(map[string]bool, 3*len(commits))
	appendRevision := func(revision string) {
		if seenRevisions[revision] {
			return
		}
		seenRevisions[revision] = true
		revisions = append(revisions, revision)
	}
	mergeBases := make(map[string]string)
	for _, commit := range commits {
		appendRevision(commit.hash)
		for _, parent := range commit.parents {
			appendRevision(parent)
		}
		parentCount := len(commit.parents)
		if parentCount == 0 || parentCount > maximumCompletionParents {
			return nil, fmt.Errorf(
				"completion commit %.12s requires one or two parents, found %d",
				commit.hash, parentCount,
			)
		}
		if parentCount == maximumCompletionParents {
			output, err := gitCompletionCommand(
				ctx, repository, "merge-base", "--all", commit.parents[0], commit.parents[1],
			)
			if err != nil {
				return nil, fmt.Errorf("derive completion merge base: %w", err)
			}
			bases := strings.Fields(string(output))
			if len(bases) != 1 {
				return nil, fmt.Errorf("completion merge requires exactly one merge base, found %d", len(bases))
			}
			mergeBases[commit.hash] = bases[0]
			appendRevision(bases[0])
		}
	}
	plans, err := gitCompletionPlans(ctx, repository, revisions)
	if err != nil {
		return nil, err
	}
	transitions := make(map[string]completionTransition, len(commits))
	for _, commit := range commits {
		child := plans[commit.hash]
		var baseline Plan
		if len(commit.parents) < maximumCompletionParents {
			baseline = plans[commit.parents[0]]
		} else {
			merged, err := MergeDocuments(
				plans[mergeBases[commit.hash]],
				plans[commit.parents[0]],
				plans[commit.parents[1]],
			)
			if err != nil {
				return nil, fmt.Errorf("derive completion merge plan: %w", err)
			}
			baseline = merged
		}
		transitions[commit.hash] = completionTransition{baseline: baseline, child: child}
	}
	return transitions, nil
}

func gitCompletionPlans(ctx context.Context, repository string, revisions []string) (map[string]Plan, error) {
	plans := make(map[string]Plan, len(revisions))
	if len(revisions) == 0 {
		return plans, nil
	}
	var input strings.Builder
	for _, revision := range revisions {
		input.WriteString(revision)
		input.WriteByte(':')
		input.WriteString(Path)
		input.WriteByte('\n')
	}
	output, err := gitCompletionCommandInput(ctx, repository, strings.NewReader(input.String()), "cat-file", "--batch")
	if err != nil {
		return nil, err
	}
	reader := bufio.NewReader(bytes.NewReader(output))
	for _, revision := range revisions {
		data, err := readGitCompletionObject(reader, revision+":"+Path, "blob")
		if err != nil {
			return nil, err
		}
		document, err := ParseHistorical(data)
		if err != nil {
			return nil, fmt.Errorf("plan: validate Git plan at %.12s: %w", revision, err)
		}
		plans[revision] = document
	}
	return plans, nil
}

func readGitCompletionObject(reader *bufio.Reader, requested, objectType string) ([]byte, error) {
	header, err := reader.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("plan: read Git %s header for %.12s: %w", objectType, requested, err)
	}
	fields := strings.Fields(header)
	if len(fields) != 3 || fields[1] != objectType {
		return nil, fmt.Errorf("plan: Git %s for %.12s is absent or invalid", objectType, requested)
	}
	size, err := strconv.Atoi(fields[2])
	if err != nil || size < 0 {
		return nil, fmt.Errorf("plan: Git %s for %.12s has invalid size", objectType, requested)
	}
	data := make([]byte, size)
	if _, err := io.ReadFull(reader, data); err != nil {
		return nil, fmt.Errorf("plan: read Git %s for %.12s: %w", objectType, requested, err)
	}
	separator, err := reader.ReadByte()
	if err != nil || separator != '\n' {
		return nil, fmt.Errorf("plan: Git %s for %.12s lacks its separator", objectType, requested)
	}
	return data, nil
}

// advanceHistoricalPlan applies the same row transition as Advance without
// reinterpreting dependency syntax from an older plan schema. The exact child
// comparison below remains the authority for what the commit changed.
func advanceHistoricalPlan(document Plan, itemID, stepID string) (Plan, error) {
	document.Items = slices.Clone(document.Items)
	for itemIndex := range document.Items {
		if document.Items[itemIndex].ID != itemID {
			continue
		}
		document.Items[itemIndex].Steps = slices.Clone(document.Items[itemIndex].Steps)
		for stepIndex := range document.Items[itemIndex].Steps {
			if document.Items[itemIndex].Steps[stepIndex].ID != stepID {
				continue
			}
			if document.Items[itemIndex].Steps[stepIndex].Status != StatusOpen {
				return Plan{}, fmt.Errorf("step %q in %q is not open", stepID, itemID)
			}
			document.Items[itemIndex].Steps = slices.Delete(
				document.Items[itemIndex].Steps, stepIndex, stepIndex+1,
			)
			if len(document.Items[itemIndex].Steps) == 0 {
				document.Items = slices.Delete(document.Items, itemIndex, itemIndex+1)
			}
			return document, nil
		}
		return Plan{}, fmt.Errorf("step %q not found in %q", stepID, itemID)
	}
	return Plan{}, fmt.Errorf("item %q not found", itemID)
}

func requireCompletionAcceptance(gate runrecord.GateResult, reference, verify string) error {
	var acceptance []runrecord.GateStep
	for _, step := range gate.Steps {
		if step.Name == "acceptance" {
			acceptance = append(acceptance, step)
		}
	}
	if len(acceptance) != 1 ||
		acceptance[0].Outcome != runrecord.StepSucceeded && acceptance[0].Outcome != runrecord.StepReused {
		return errors.New("completion gate lacks one successful acceptance step")
	}
	if err := runrecord.VerifyCompletionAcceptanceEvidence(acceptance[0].Evidence, reference, verify); err != nil {
		return errors.New("completion acceptance evidence does not bind the exact plan verifier")
	}
	return nil
}

func exactPlanStep(document Plan, itemID, stepID string) (Step, bool) {
	for _, item := range document.Items {
		if item.ID != itemID {
			continue
		}
		for _, step := range item.Steps {
			if step.ID == stepID {
				return step, true
			}
		}
		return Step{}, false
	}
	return Step{}, false
}

func planHasItem(document Plan, itemID string) bool {
	for _, item := range document.Items {
		if item.ID == itemID {
			return true
		}
	}
	return false
}

func gitCompletionCommand(ctx context.Context, repository string, arguments ...string) ([]byte, error) {
	return gitCompletionCommandInput(ctx, repository, nil, arguments...)
}

func gitCompletionCommandInput(
	ctx context.Context,
	repository string,
	stdin io.Reader,
	arguments ...string,
) ([]byte, error) {
	gitArguments := make([]string, 0, len(arguments)+1)
	gitArguments = append(gitArguments, "--no-replace-objects")
	gitArguments = append(gitArguments, arguments...)
	var stdout, stderr bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "git", Args: gitArguments, Dir: filepath.Clean(repository),
		Env: gitauthority.RepositoryEnvironment(), Stdin: stdin, Stdout: &stdout, Stderr: &stderr,
	})
	if err != nil {
		return nil, fmt.Errorf("plan: git %s: %w: %s", strings.Join(arguments, " "), err, strings.TrimSpace(stderr.String()))
	}
	if receipt.ExitCode != 0 {
		return nil, fmt.Errorf("plan: git %s: exit=%d: %s", strings.Join(arguments, " "), receipt.ExitCode, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}
