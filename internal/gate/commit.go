package gate

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/atomicfile"
	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/fsatomic"
	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/runrecord"
)

func (g *gateContext) stepCommit() (bool, error) {
	if g.manifestPlan == nil {
		return false, errors.New("commit admission: manifest plan is absent")
	}
	if g.acceptedTree == "" || candidateTreeKey(g.acceptedTree) != g.manifestPlan.CandidateTree {
		return false, errors.New("commit admission: accepted immutable candidate tree is absent")
	}
	if err := g.requirePreparedCandidate(g.manifestPlan.CandidateTree); err != nil {
		return false, err
	}
	if err := g.requireCandidateTree(g.manifestPlan.CandidateTree); err != nil {
		return false, err
	}
	currentTree, err := g.plannedTree()
	if err != nil {
		return false, err
	}
	if currentTree != g.acceptedTree {
		return false, errors.New("commit admission: planned content changed after acceptance")
	}
	if err := validateManifestCommitAdmission(*g.manifestPlan, g.terminal); err != nil {
		return false, err
	}
	// Validate the immutable result shape before Git advances. A schema error
	// discovered after commit cannot be represented by the normal debt batch.
	recipeID, err := g.gateRecipeID()
	if err != nil {
		return false, err
	}
	validationDuration, err := completedGateMeasurement(g.start, "pre-commit record validation")
	if err != nil {
		return false, err
	}
	steps := append(slices.Clone(g.steps), runrecord.GateStep{
		Name: "commit", Phase: runrecord.PhasePackage, Outcome: runrecord.StepSucceeded, DurationNS: validationDuration,
	})
	if _, err := runrecord.NewGateRecord(
		recipeID, g.environment.ID, strings.Repeat("0", hex.EncodedLen(sha1.Size)), runrecord.OutcomeSucceeded, "", validationDuration, steps,
	); err != nil {
		return false, fmt.Errorf("pre-commit record validation: %w", err)
	}
	planFile := filepath.Join(g.repo, filepath.FromSlash(plan.Path))
	planBefore, err := acceptedPlanBytes(g.repo, g.acceptedTree)
	if err != nil {
		return false, err
	}
	worktreePlan, err := os.ReadFile(planFile)
	if err != nil {
		return false, err
	}
	if !bytes.Equal(worktreePlan, planBefore) {
		return false, errors.New("commit admission: worktree plan differs from the accepted candidate tree")
	}
	document, err := plan.Parse(planBefore)
	if err != nil {
		return false, err
	}
	acceptance, err := completionAcceptanceEvidenceForPlan(document, g.planRef, g.completionAuthority)
	if err != nil {
		return false, fmt.Errorf("commit admission: plan changed after acceptance: %w", err)
	}
	if acceptance != g.stepEvidence["acceptance"] {
		return false, errors.New("commit admission: plan verifier changed after acceptance")
	}
	planHead, err := command(g.repo, "git", "rev-parse", "HEAD")
	if err != nil {
		return false, err
	}
	planHead = strings.TrimSpace(planHead)
	if planHead != g.planHead {
		return false, fmt.Errorf("commit admission: plan authority moved from %.12s to %.12s", g.planHead, planHead)
	}
	completionStore, err := overgodb.Open(filepath.Join(g.repo, g.storePath))
	if err != nil {
		return false, fmt.Errorf("commit admission: lock completion store: %w", err)
	}
	if err := requireSoleCurrentGatePreparation(
		context.Background(), completionStore, g.preparation, g.preparationCommit,
	); err != nil {
		_ = completionStore.Close()
		return false, fmt.Errorf("commit admission: recheck gate preparation authority: %w", err)
	}
	completionAuthority, err := plan.ResolveCompletionAuthority(
		context.Background(), g.repo, g.planHead, document, completionStore,
	)
	if err != nil {
		_ = completionStore.Close()
		return false, fmt.Errorf("commit admission: recheck completion authority: %w", err)
	}
	role, err := plan.AutomationRole("")
	if err != nil {
		_ = completionStore.Close()
		return false, err
	}
	item, step, open := plan.Current(document, role, completionAuthority)
	if !open || item.ID+"/"+step.ID != g.planRef {
		_ = completionStore.Close()
		return false, errors.New("commit admission: completion authority no longer selects the gated row")
	}
	planHead, err = command(g.repo, "git", "rev-parse", "HEAD")
	if err != nil {
		_ = completionStore.Close()
		return false, err
	}
	if planHead = strings.TrimSpace(planHead); planHead != g.planHead {
		_ = completionStore.Close()
		return false, fmt.Errorf("commit admission: plan authority moved from %.12s to %.12s", g.planHead, planHead)
	}
	g.completionAuthority = completionAuthority
	g.completionStore = completionStore
	planAfter, err := advancedPlanBytes(planBefore, g.planRef)
	if err != nil {
		return false, err
	}
	childDocument, err := plan.Parse(planAfter)
	if err != nil {
		return false, err
	}
	mergeAuthority, err := g.deriveProjectedMergeAuthority(document, childDocument, completionStore)
	if err != nil {
		return false, fmt.Errorf("commit admission: derive projected merge authority: %w", err)
	}
	g.mergeAuthority = mergeAuthority
	// Structured completion trailers make Git the completion record:
	// the row leaves the plan in this same commit, and the trailers
	// carry what completed and how it was verified. The gate record
	// published after commit binds the evidence by commit SHA.
	messageFile, err := g.completionMessageFile(document)
	if err != nil {
		return false, err
	}
	defer os.Remove(messageFile)
	planInfo, err := os.Stat(planFile)
	if err != nil {
		return false, err
	}
	headReference, err := currentHeadReference(g.repo)
	if err != nil {
		return false, err
	}
	if headReference == "HEAD" {
		return false, errors.New("commit admission: detached HEAD cannot provide branch-reference authority")
	}
	if err := g.requireGateStartState(); err != nil {
		return false, err
	}
	indexBefore, mergeState := g.indexBefore, g.mergeBefore
	indexAfter, err := buildAcceptedCompletionIndex(g.repo, g.acceptedTree, planAfter)
	if err != nil {
		return false, err
	}
	indexRestore, err := buildGateIndexForTree(g.repo, indexBefore.Tree)
	if err != nil {
		return false, err
	}
	intent := gateCommitIntent{
		Version: artifact.InitialDocumentVersion, Preparation: g.preparation.ID,
		PreparationCommit: g.preparationCommit, Recipe: recipeID,
		CandidateManifest: g.candidateManifest.ID,
		Parent:            g.planHead, HeadReference: headReference, Merge: mergeState,
		PlanProjection: g.planProjection,
		IndexTree:      indexBefore.Tree, Tree: indexAfter.Tree,
		IndexBefore: indexBefore.Data, IndexAfter: indexAfter.Data, IndexRestore: indexRestore.Data,
		IndexMode: uint32(indexBefore.Mode.Perm()),
		PlanRef:   g.planRef, Paths: slices.Clone(g.paths),
		Plan: planBefore, AdvancedPlan: planAfter, PlanMode: uint32(planInfo.Mode().Perm()),
	}
	if mergeAuthority != nil {
		content, contentErr := mergeAuthority.Content()
		if contentErr != nil {
			return false, contentErr
		}
		intent.MergeAuthority = content.Data
	}
	completionMessage, err := os.ReadFile(messageFile)
	if err != nil {
		return false, err
	}
	messageItem, messageStep, found := strings.Cut(g.planRef, "/")
	if !found {
		return false, errors.New("commit admission: invalid completion reference")
	}
	mergeAuthorityID := artifact.ID{}
	if mergeAuthority != nil {
		mergeAuthorityID = mergeAuthority.ID
	}
	if err := plan.VerifyCompletionCommitMessageWithMergeAuthority(
		string(completionMessage), document, messageItem, messageStep, recipeID,
		g.candidateManifest.ID, g.preparation.ID, g.preparationCommit,
		g.planProjection, mergeAuthorityID,
	); err != nil {
		return false, fmt.Errorf("commit admission: verify completion message: %w", err)
	}
	if err := verifyProspectiveGateCompletion(
		g.repo, intent, document, childDocument, string(completionMessage), g.completionStore,
	); err != nil {
		return false, fmt.Errorf("commit admission: prospective completion transition: %w", err)
	}
	// Construct the exact completion commit while every tree is still private
	// or referenced by the caller's captured state. The first durable intent can
	// then bind one keepalive commit that reaches both rollback trees and this
	// otherwise-unreferenced completion commit before any shared mutation.
	intent.Commit, err = createGateCommit(g.repo, intent, messageFile)
	if err != nil {
		return false, err
	}
	if err := initializeGateIntentKeepalive(g.repo, &intent); err != nil {
		return false, err
	}
	if err := writeGateCommitIntent(g.repo, intent); err != nil {
		return false, err
	}
	referenceAdvanced, planPublished := false, false
	var rollbackPlan func() error
	defer func() {
		if referenceAdvanced || g.commitInterrupted {
			return
		}
		// A failed pre-CAS attempt may have published the write-ahead plan or
		// installed only the exact final index recorded by the intent. Restore
		// both only while this worktree still names the captured parent and each
		// shared file remains in one of its exact write-ahead states. Any other
		// state belongs to a concurrent operation and must survive for recovery.
		if !exactGateHead(g.repo, intent.HeadReference, intent.Parent) {
			g.commitInterrupted = true
			return
		}
		if planPublished {
			if err := rollbackPlan(); err != nil {
				g.commitInterrupted = true
				return
			}
			planPublished = false
		}
		if err := restoreCapturedIndex(g.repo, intent); err != nil {
			g.commitInterrupted = true
			return
		}
		matches, err := planBytesMatch(g.repo, intent.Plan)
		if err != nil || !matches {
			g.commitInterrupted = true
			return
		}
		if err := removeGateCommitIntent(g.repo); err != nil {
			g.commitInterrupted = true
		}
	}()
	// Publish the plan before the prepared branch transaction. A process death
	// can therefore leave a durable intent with the parent and either exact plan
	// state, but never the completion branch pointing at a plan that publication
	// has not yet installed.
	rollbackPlan, err = publishPlanTransition(
		g.repo, intent.Preparation, planBefore, planAfter, planInfo.Mode().Perm(),
	)
	if err != nil {
		return false, err
	}
	planPublished = true
	// The final tree was built in an isolated index and made durable in the
	// intent before this shared-index transition. Install it only from the exact
	// captured index so a concurrent staged change is never silently replaced.
	if err := installCompletionIndexAndAdvanceGateReference(g.repo, intent); err != nil {
		if !gateReferenceEquals(g.repo, intent.HeadReference, intent.Parent) {
			g.commitInterrupted = true
		}
		return false, err
	}
	referenceAdvanced = true
	planPublished = false
	g.commitInterrupted = true
	writtenPlan, err := os.ReadFile(planFile)
	if err != nil {
		return false, err
	}
	if !bytes.Equal(writtenPlan, planAfter) {
		return false, errors.New("commit admission: advanced plan bytes differ from the write-ahead transition")
	}
	if err := clearCommittedMergeState(g.repo, intent); err != nil {
		return false, err
	}
	if err := validateInterruptedCommitAuthority(g.repo, intent, g.completionStore); err != nil {
		return false, fmt.Errorf("commit authority validation: %w", err)
	}
	if !exactGateHead(g.repo, intent.HeadReference, intent.Commit) {
		return false, errors.New("commit authority validation: captured branch moved after exact ref update")
	}
	g.committedHead = intent.Commit
	g.commitInterrupted = false
	return false, nil
}

func treePathMode(repo, tree, repositoryPath string) (string, error) {
	entry, err := command(repo, "git", "ls-tree", tree, "--", repositoryPath)
	if err != nil {
		return "", err
	}
	metadata, _, found := strings.Cut(entry, "\t")
	mode, objectMetadata, foundMode := strings.Cut(metadata, " ")
	objectType, objectID, foundObject := strings.Cut(objectMetadata, " ")
	if !found || !foundMode || !foundObject || mode == "" || objectType != "blob" ||
		!validGitObjectID(strings.TrimSpace(objectID)) {
		return "", fmt.Errorf("commit admission: %s is absent from accepted tree %s", repositoryPath, tree)
	}
	return mode, nil
}

func acceptedPlanBytes(repo, tree string) ([]byte, error) {
	if !validGitObjectID(tree) {
		return nil, errors.New("commit admission: accepted tree identity is invalid")
	}
	content, err := command(repo, "git", "show", tree+":"+plan.Path)
	if err != nil {
		return nil, err
	}
	return []byte(content), nil
}

func gitHashObject(repo string, content []byte) (string, error) {
	process := newGateGitWriterCommand(repo, nil, "hash-object", "-w", "--stdin")
	process.Stdin = bytes.NewReader(content)
	output, err := process.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf(
			"git hash-object -w --stdin: %v: %s",
			err,
			clioptions.Tail(string(output), clioptions.DiagnosticTailBytes),
		)
	}
	object := strings.TrimSpace(string(output))
	if !validGitObjectID(object) {
		return "", errors.New("git hash-object returned an invalid object identity")
	}
	return object, nil
}

type gateIndexSnapshot struct {
	Tree string
	Data []byte
	Mode fs.FileMode
}

// buildAcceptedCompletionIndex constructs the exact commit index in private.
// The caller's shared index is not touched until the completed index and its
// exact original bytes are both durable in the commit intent.
func buildAcceptedCompletionIndex(repo, acceptedTree string, planAfter []byte) (gateIndexSnapshot, error) {
	if !validGitObjectID(acceptedTree) {
		return gateIndexSnapshot{}, errors.New("commit admission: accepted tree identity is invalid")
	}
	temporary, err := os.MkdirTemp("", "overgo-gate-commit-index-*")
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	defer os.RemoveAll(temporary)
	indexPath := filepath.Join(temporary, "index")
	environment := gitIndexEnvironment(indexPath)
	if _, err := gitWriterCommandEnvironment(repo, environment, "read-tree", acceptedTree); err != nil {
		return gateIndexSnapshot{}, err
	}
	planMode, err := treePathMode(repo, acceptedTree, plan.Path)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	planBlob, err := gitHashObject(repo, planAfter)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	if _, err := gitWriterCommandEnvironment(
		repo, environment, "update-index", "--add", "--cacheinfo", planMode+","+planBlob+","+plan.Path,
	); err != nil {
		return gateIndexSnapshot{}, err
	}
	tree, err := gitWriterCommandEnvironment(repo, environment, "write-tree")
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	return gateIndexSnapshot{Tree: strings.TrimSpace(tree), Data: data}, nil
}

// buildGateIndexForTree precomputes the exact recovery target. Reusing captured
// index bytes after renaming them would retain stale entry stat caches under a
// new index-file timestamp and can make racily-clean worktree changes invisible.
// The gate rejects semantic special-entry flags, so a normalized index for the
// same tree preserves staging content while deliberately resetting volatile
// stat/cache metadata.
func buildGateIndexForTree(repo, tree string) (gateIndexSnapshot, error) {
	if !validGitObjectID(tree) {
		return gateIndexSnapshot{}, errors.New("gate: exact recovery index tree is invalid")
	}
	temporary, err := os.MkdirTemp("", "overgo-gate-recovery-index-*")
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	defer os.RemoveAll(temporary)
	indexPath := filepath.Join(temporary, "index")
	environment := gitIndexEnvironment(indexPath)
	if _, err := gitWriterCommandEnvironment(repo, environment, "read-tree", tree); err != nil {
		return gateIndexSnapshot{}, err
	}
	writtenTree, err := gitWriterCommandEnvironment(repo, environment, "write-tree")
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	writtenTree = strings.TrimSpace(writtenTree)
	if writtenTree != tree {
		return gateIndexSnapshot{}, errors.New("gate: normalized recovery index differs from its tree authority")
	}
	return gateIndexSnapshot{Tree: tree, Data: data}, nil
}

func captureGateIndex(repo string) (gateIndexSnapshot, error) {
	indexPath, err := gateIndexPath(repo)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	info, err := os.Lstat(indexPath)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	if !info.Mode().IsRegular() {
		return gateIndexSnapshot{}, errors.New("gate: Git index must be one regular file")
	}
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	tree, err := indexTreeForBytes(repo, data)
	if err != nil {
		return gateIndexSnapshot{}, err
	}
	return gateIndexSnapshot{Tree: tree, Data: data, Mode: info.Mode().Perm()}, nil
}

func requireSupportedGateIndex(repo string, environment []string) error {
	shared, err := commandEnvironment(repo, environment, "git", "rev-parse", "--shared-index-path")
	if err != nil {
		return err
	}
	if strings.TrimSpace(shared) != "" {
		return errors.New("commit admission: split Git indexes are not supported by exact index recovery")
	}
	entries, err := commandEnvironment(repo, environment, "git", "ls-files", "-v", "-z")
	if err != nil {
		return err
	}
	for entry := range strings.SplitSeq(entries, "\x00") {
		if entry == "" {
			continue
		}
		if !strings.HasPrefix(entry, "H ") {
			return errors.New("commit admission: special Git index entry flags are not supported by exact completion staging")
		}
	}
	resolveUndo, err := commandEnvironment(repo, environment, "git", "ls-files", "--resolve-undo", "-z")
	if err != nil {
		return err
	}
	if resolveUndo != "" {
		return errors.New("commit admission: resolve-undo Git index metadata is not supported by exact completion staging")
	}
	return nil
}

func gateIndexPath(repo string) (string, error) {
	indexPath, err := gitMetadataPath(repo, "index")
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(indexPath) == "" {
		return "", errors.New("gate: Git index path is absent")
	}
	return indexPath, nil
}

func indexTreeForBytes(repo string, data []byte) (string, error) {
	if len(data) == 0 {
		return "", errors.New("gate: Git index bytes are absent")
	}
	temporary, err := os.MkdirTemp("", "overgo-gate-index-verify-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(temporary)
	indexPath := filepath.Join(temporary, "index")
	if err := os.WriteFile(indexPath, data, gatePrivateFileMode); err != nil {
		return "", err
	}
	environment := gitIndexEnvironment(indexPath)
	if err := requireSupportedGateIndex(repo, environment); err != nil {
		return "", err
	}
	tree, err := gitWriterCommandEnvironment(repo, environment, "write-tree")
	if err != nil {
		return "", err
	}
	tree = strings.TrimSpace(tree)
	if !validGitObjectID(tree) {
		return "", errors.New("gate: Git index produced an invalid tree identity")
	}
	indexed, err := commandEnvironment(repo, environment, "git", "ls-files", "-z")
	if err != nil {
		return "", err
	}
	treePaths, err := commandEnvironment(repo, environment, "git", "ls-tree", "-r", "--name-only", "-z", tree)
	if err != nil {
		return "", err
	}
	if !sameNULTerminatedNames(indexed, treePaths) {
		return "", errors.New("commit admission: intent-to-add or non-tree Git index entries are not supported by exact completion staging")
	}
	return tree, nil
}

func sameNULTerminatedNames(left, right string) bool {
	toSet := func(raw string) map[string]struct{} {
		values := make(map[string]struct{})
		for value := range strings.SplitSeq(raw, "\x00") {
			if value != "" {
				values[value] = struct{}{}
			}
		}
		return values
	}
	return maps.Equal(toSet(left), toSet(right))
}

func validateGateIntentIndexes(repo string, intent gateCommitIntent) error {
	beforeTree, err := indexTreeForBytes(repo, intent.IndexBefore)
	if err != nil {
		return err
	}
	afterTree, err := indexTreeForBytes(repo, intent.IndexAfter)
	if err != nil {
		return err
	}
	restoreTree, err := indexTreeForBytes(repo, intent.IndexRestore)
	if err != nil {
		return err
	}
	if beforeTree != intent.IndexTree || afterTree != intent.Tree || restoreTree != intent.IndexTree {
		return errors.New("gate: interrupted commit index bytes differ from their tree authorities")
	}
	keepaliveTree, err := buildGateIntentKeepaliveTree(repo, intent)
	if err != nil {
		return err
	}
	if keepaliveTree != intent.KeepaliveTree {
		return errors.New("gate: interrupted commit keepalive differs from its object authorities")
	}
	keepaliveCommit, err := buildGateIntentKeepaliveCommit(repo, intent, keepaliveTree)
	if err != nil {
		return err
	}
	if keepaliveCommit != intent.KeepaliveCommit {
		return errors.New("gate: interrupted commit keepalive differs from its commit authority")
	}
	return nil
}

const gateIntentKeepaliveNamespace = "refs/overgo/gate-intents/"

func gateIntentKeepaliveReference(preparation artifact.ID) string {
	return gateIntentKeepaliveNamespace + preparation.DigestHex()
}

// buildGateIntentKeepaliveTree creates a deterministic synthetic tree whose
// subtrees are every otherwise-unreferenced Git object source needed for exact
// rollback. A deterministic synthetic commit points at this tree and names the
// completion commit as its parent, making the one intent-owned ref a complete
// reachability root without walking ordinary history during validation.
func buildGateIntentKeepaliveTree(repo string, intent gateCommitIntent) (string, error) {
	if !validGitObjectID(intent.IndexTree) {
		return "", errors.New("gate: interrupted commit keepalive has no exact index tree")
	}
	type entry struct {
		name string
		tree string
	}
	if !validGitObjectID(intent.Tree) {
		return "", errors.New("gate: interrupted commit keepalive has no exact accepted tree")
	}
	entries := []entry{
		{name: "accepted", tree: intent.Tree},
		{name: "index", tree: intent.IndexTree},
	}
	if intent.Merge != nil && intent.Merge.AutoMerge != nil {
		autoMerge := strings.TrimSpace(string(intent.Merge.AutoMerge))
		if !validGitObjectID(autoMerge) {
			return "", errors.New("gate: interrupted commit keepalive has invalid AUTO_MERGE authority")
		}
		entries = append(entries, entry{name: "auto-merge", tree: autoMerge})
	}
	slices.SortFunc(entries, func(left, right entry) int { return strings.Compare(left.name, right.name) })
	var input bytes.Buffer
	for _, candidate := range entries {
		fmt.Fprintf(&input, "040000 tree %s\t%s\x00", candidate.tree, candidate.name)
	}
	process := newGateGitWriterCommand(repo, nil, "mktree", "-z")
	process.Stdin = &input
	output, err := process.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf(
			"gate: build interrupted commit keepalive tree: %w: %s",
			err, clioptions.Tail(string(output), clioptions.DiagnosticTailBytes),
		)
	}
	tree := strings.TrimSpace(string(output))
	if !validGitObjectID(tree) {
		return "", errors.New("gate: Git produced an invalid interrupted commit keepalive tree")
	}
	return tree, nil
}

func buildGateIntentKeepaliveCommit(repo string, intent gateCommitIntent, tree string) (string, error) {
	if !validGitObjectID(tree) || !validGitObjectID(intent.Commit) {
		return "", errors.New("gate: interrupted commit keepalive requires exact tree and completion commit authorities")
	}
	process := newGateGitWriterCommand(repo, nil, "commit-tree", tree, "-p", intent.Commit)
	process.Env = append(
		process.Env,
		"GIT_AUTHOR_NAME=Overgo Gate",
		"GIT_AUTHOR_EMAIL=gate@overgo.invalid",
		"GIT_AUTHOR_DATE=1970-01-01T00:00:00 +0000",
		"GIT_COMMITTER_NAME=Overgo Gate",
		"GIT_COMMITTER_EMAIL=gate@overgo.invalid",
		"GIT_COMMITTER_DATE=1970-01-01T00:00:00 +0000",
	)
	process.Stdin = strings.NewReader("overgo interrupted commit keepalive\n")
	output, err := process.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf(
			"gate: build interrupted commit keepalive commit: %w: %s",
			err, clioptions.Tail(string(output), clioptions.DiagnosticTailBytes),
		)
	}
	commit := strings.TrimSpace(string(output))
	if !validGitObjectID(commit) {
		return "", errors.New("gate: Git produced an invalid interrupted commit keepalive identity")
	}
	return commit, nil
}

func initializeGateIntentKeepalive(repo string, intent *gateCommitIntent) error {
	if intent == nil || !intent.Preparation.Valid() {
		return errors.New("gate: interrupted commit keepalive has no preparation authority")
	}
	tree, err := buildGateIntentKeepaliveTree(repo, *intent)
	if err != nil {
		return err
	}
	commit, err := buildGateIntentKeepaliveCommit(repo, *intent, tree)
	if err != nil {
		return err
	}
	intent.KeepaliveRef = gateIntentKeepaliveReference(intent.Preparation)
	intent.KeepaliveTree = tree
	intent.KeepaliveCommit = commit
	return nil
}

func installCompletionIndexUnderLock(repo, indexPath string, intent gateCommitIntent) error {
	if err := requireGateIntentKeepalive(repo, intent); err != nil {
		return err
	}
	return replaceGateIndexStatesUnderLock(
		indexPath,
		[][]byte{intent.IndexBefore},
		[][]byte{intent.IndexAfter},
		intent.IndexAfter,
		fs.FileMode(intent.IndexMode),
	)
}

func createGateCommit(repo string, intent gateCommitIntent, messageFile string) (string, error) {
	if !validGitObjectID(intent.Tree) || !validGitObjectID(intent.Parent) || strings.TrimSpace(messageFile) == "" {
		return "", errors.New("commit admission: exact tree, parent, and message file are required")
	}
	arguments := []string{"commit-tree", intent.Tree, "-p", intent.Parent}
	if intent.Merge != nil {
		for _, parent := range intent.Merge.parents() {
			arguments = append(arguments, "-p", parent)
		}
	}
	arguments = append(arguments, "-F", messageFile)
	output, err := gitAuthorityWriterOutput(repo, arguments...)
	if err != nil {
		return "", err
	}
	commit := strings.TrimSpace(string(output))
	if !validGitObjectID(commit) {
		return "", errors.New("commit admission: git commit-tree returned an invalid object identity")
	}
	return commit, nil
}

type gatePreparedReferenceTransaction struct {
	process  *exec.Cmd
	input    io.WriteCloser
	output   *bufio.Reader
	stderr   *bytes.Buffer
	finished bool
}

func prepareGateReferenceTransaction(
	repo, reference, target, observed, reason string,
) (*gatePreparedReferenceTransaction, error) {
	if !strings.HasPrefix(reference, "refs/heads/") || strings.ContainsAny(reference, "\x00\r\n \t") ||
		!validGitObjectID(target) || !validGitObjectID(observed) {
		return nil, errors.New("gate: prepared ref transaction requires an attached branch and exact objects")
	}
	stderr := new(bytes.Buffer)
	process := newGateGitWriterCommand(repo, nil, "update-ref", "-m", reason, "--stdin")
	input, err := process.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := process.StdoutPipe()
	if err != nil {
		_ = input.Close()
		return nil, err
	}
	process.Stderr = stderr
	if err := process.Start(); err != nil {
		_ = input.Close()
		return nil, err
	}
	transaction := &gatePreparedReferenceTransaction{
		process: process, input: input, output: bufio.NewReader(stdout), stderr: stderr,
	}
	if err := transaction.exchange("start", "start: ok"); err != nil {
		return nil, transaction.fail("start", err)
	}
	// Because reference is the current symbolic HEAD referent, Git includes
	// HEAD in the prepared branch update and holds HEAD.lock with the branch
	// lock. Explicitly queueing symref-verify HEAD is both redundant and rejected
	// by Git as a duplicate referent update; the post-prepare exact check below
	// binds the locked symbolic target before any shared-state mutation.
	instruction := fmt.Sprintf("update %s %s %s\n", reference, target, observed)
	if target == observed {
		instruction = fmt.Sprintf("verify %s %s\n", reference, observed)
	}
	if _, err := io.WriteString(transaction.input, instruction); err != nil {
		return nil, transaction.fail("queue", err)
	}
	if err := transaction.exchange("prepare", "prepare: ok"); err != nil {
		return nil, transaction.fail("prepare", err)
	}
	return transaction, nil
}

// installCompletionIndexAndAdvanceGateReference holds the actual index and
// captured branch locks across exact index installation and branch CAS. It
// never updates whichever branch HEAD might name later, and detached HEAD is
// refused because Git cannot atomically distinguish it from a symbolic HEAD at
// the same object identity.
func installCompletionIndexAndAdvanceGateReference(repo string, intent gateCommitIntent) error {
	err := withGateGitStateLock(
		repo, fs.FileMode(intent.IndexMode), intent,
		gateIndexCASBeforeLockHook, gateIndexCASLockedHook,
		func(indexPath string) (err error) {
			if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
				return errors.New("commit admission: HEAD reference or parent moved before prepared commit")
			}
			if err := exactGateIndexState(
				indexPath, fs.FileMode(intent.IndexMode), intent.IndexBefore,
			); err != nil {
				return err
			}
			if err := requireGateIntentKeepalive(repo, intent); err != nil {
				return err
			}
			transaction, err := prepareGateReferenceTransaction(
				repo, intent.HeadReference, intent.Commit, intent.Parent, "overgo gate completion",
			)
			if err != nil {
				return err
			}
			defer func() { err = errors.Join(err, transaction.abort()) }()
			if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
				return errors.New("commit admission: HEAD moved while preparing exact ref update")
			}
			if err := installCompletionIndexUnderLock(repo, indexPath, intent); err != nil {
				return err
			}
			if gateCommitIndexInstalledHook != nil {
				gateCommitIndexInstalledHook(repo, intent)
			}
			if !exactGateHead(repo, intent.HeadReference, intent.Parent) {
				return errors.New("commit admission: prepared branch moved before publication")
			}
			if err := exactGateIndexState(
				indexPath, fs.FileMode(intent.IndexMode), intent.IndexAfter,
			); err != nil {
				return err
			}
			if err := requireGateIntentKeepalive(repo, intent); err != nil {
				return err
			}
			if err := transaction.commit(); err != nil {
				return err
			}
			if !exactGateHead(repo, intent.HeadReference, intent.Commit) {
				return errors.New("commit admission: prepared branch publication has inexact HEAD authority")
			}
			return nil
		},
	)
	if errors.Is(err, errGateIndexMoved) {
		return errors.New("commit admission: Git index moved before exact completion publication")
	}
	return err
}

func exactGateHead(repo, reference, commit string) bool {
	current, err := currentHeadReference(repo)
	return err == nil && current == reference && gateReferenceEquals(repo, reference, commit)
}

func gateReferenceEquals(repo, reference, commit string) bool {
	if strings.TrimSpace(reference) == "" || !validGitObjectID(commit) {
		return false
	}
	value, err := gitAuthorityOutput(repo, "rev-parse", "--verify", reference)
	return err == nil && strings.TrimSpace(string(value)) == commit
}

func validateManifestCommitAdmission(manifest automationcheck.ManifestPlan, terminal map[string]automationcheck.Evidence) error {
	if err := manifest.Validate(); err != nil {
		return fmt.Errorf("commit admission: %w", err)
	}
	if len(manifest.Invocations) == 0 || manifest.Invocations[len(manifest.Invocations)-1].Check.Name != "commit" {
		return errors.New("commit admission: commit is not the final planned invocation")
	}
	for _, invocation := range manifest.Invocations[:len(manifest.Invocations)-1] {
		evidence, found := terminal[invocation.Check.Name]
		if !found || !evidence.ID.Valid() {
			return fmt.Errorf("commit admission: %s lacks terminal evidence", invocation.Check.Name)
		}
		if evidence.Outcome != runrecord.LanePassed || evidence.Authority == nil || evidence.Authority.Plan != manifest.ID ||
			evidence.Authority.Definition != invocation.ID {
			return fmt.Errorf("commit admission: %s evidence has wrong outcome or authority", invocation.Check.Name)
		}
		if evidence.Inapplicable && evidence.Reused {
			return fmt.Errorf("commit admission: %s evidence has contradictory outcomes", invocation.Check.Name)
		}
	}
	return nil
}

// completionMessageFile writes the operator's message plus the
// structured completion trailers to a temporary file: the plan item
// and step this commit completes, and the verify command that gated
// it. Git carries completion history; the plan keeps only open work.
func (g *gateContext) completionMessageFile(document plan.Plan) (string, error) {
	message, err := os.ReadFile(g.messageFile)
	if err != nil {
		return "", err
	}
	itemID, stepID, _ := strings.Cut(g.planRef, "/")
	if g.manifestPlan == nil || g.candidateManifest == nil {
		return "", errors.New("completion message: manifest authorities are absent")
	}
	mergeAuthority := artifact.ID{}
	if g.mergeAuthority != nil {
		mergeAuthority = g.mergeAuthority.ID
	}
	augmented, err := plan.CompletionCommitMessageWithMergeAuthority(
		message, document, itemID, stepID, g.manifestPlan.ID, g.candidateManifest.ID, g.preparation.ID,
		g.preparationCommit, g.planProjection, mergeAuthority,
	)
	if err != nil {
		return "", err
	}
	file, err := os.CreateTemp("", "gate-message-*.txt")
	if err != nil {
		return "", err
	}
	if _, err := file.Write(augmented); err != nil {
		file.Close()
		return "", err
	}
	if err := file.Close(); err != nil {
		return "", err
	}
	return file.Name(), nil
}

func (g *gateContext) deriveProjectedMergeAuthority(
	preAdvance, child plan.Plan,
	targetStore *overgodb.Store,
) (*plan.FirstParentTargetMergeAuthority, error) {
	if g.planProjection != plan.MergeProjectionFirstParentTarget {
		return nil, nil
	}
	if targetStore == nil || g.mergeBefore == nil || g.mergeSourceStore == "" {
		return nil, errors.New("first-parent-target merge requires target and source store authority")
	}
	parents := g.mergeBefore.parents()
	if len(parents) != 1 || !validGitObjectID(parents[0]) || parents[0] == g.planHead {
		return nil, errors.New("first-parent-target merge requires one distinct incoming parent")
	}
	incomingRevision := parents[0]
	loadPlan := func(label, revision string) (plan.Plan, error) {
		data, err := gitCompletionFile(g.repo, revision, plan.Path)
		if err != nil {
			return plan.Plan{}, err
		}
		document, err := plan.ParseHistorical(data)
		if err != nil {
			return plan.Plan{}, fmt.Errorf("parse %s plan %.12s: %w", label, revision, err)
		}
		return document, nil
	}
	local, err := loadPlan("local parent", g.planHead)
	if err != nil {
		return nil, err
	}
	incoming, err := loadPlan("incoming parent", incomingRevision)
	if err != nil {
		return nil, err
	}
	mergeBaseRevision, err := projectedMergeBase(g.repo, g.planHead, incomingRevision)
	if err != nil {
		return nil, err
	}
	mergeBase, err := loadPlan("merge-base", mergeBaseRevision)
	if err != nil {
		return nil, err
	}
	sourceStore, err := overgodb.OpenReadOnly(g.mergeSourceStore)
	if err != nil {
		return nil, fmt.Errorf("open projected merge source store: %w", err)
	}
	defer sourceStore.Close()
	localAuthority, err := plan.ResolveCompletionAuthority(
		context.Background(), g.repo, g.planHead, local, targetStore,
	)
	if err != nil {
		return nil, fmt.Errorf("resolve local projected-merge authority: %w", err)
	}
	incomingAuthority, err := plan.ResolveCompletionAuthority(
		context.Background(), g.repo, incomingRevision, incoming, sourceStore,
	)
	if err != nil {
		return nil, fmt.Errorf("resolve incoming projected-merge authority: %w", err)
	}
	item, step, found := strings.Cut(g.planRef, "/")
	if !found {
		return nil, errors.New("first-parent-target merge has an invalid completion reference")
	}
	receipt, err := plan.NewFirstParentTargetMergeAuthority(
		context.Background(), g.repo, g.planHead, incomingRevision, mergeBaseRevision,
		local, incoming, mergeBase, preAdvance, child, item, step,
		g.preparation.ID, g.preparationCommit, localAuthority, incomingAuthority,
		targetStore, sourceStore,
	)
	if err != nil {
		return nil, err
	}
	return &receipt, nil
}

func publishPlanTransition(
	repo string,
	preparation artifact.ID,
	before, after []byte,
	mode fs.FileMode,
) (func() error, error) {
	path := filepath.Join(repo, filepath.FromSlash(plan.Path))
	scratch, err := gatePlanScratchPath(repo, preparation)
	if err != nil {
		return nil, err
	}
	if err := atomicfile.CompareAndSwap(path, scratch, before, after, mode); err != nil {
		if errors.Is(err, atomicfile.ErrChanged) {
			return nil, fmt.Errorf("commit admission: plan changed before transition publication: %w", err)
		}
		return nil, err
	}
	return func() error {
		if err := atomicfile.CompareAndSwap(path, scratch, after, before, mode); err != nil {
			return fmt.Errorf("commit admission: plan changed before transition rollback: %w", err)
		}
		return nil
	}, nil
}

func advancedPlanBytes(before []byte, ref string) ([]byte, error) {
	document, err := plan.Parse(before)
	if err != nil {
		return nil, err
	}
	item, step, found := strings.Cut(ref, "/")
	if !found || item == "" || step == "" {
		return nil, fmt.Errorf("advance plan: invalid reference %q", ref)
	}
	advanced, err := plan.Advance(document, item, step)
	if err != nil {
		return nil, err
	}
	data, err := json.MarshalIndent(advanced, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}

// gateCommitIntent is the write-ahead boundary between the Git ref update and
// the final OvergoDB batch. If the process dies after git commit, the exact
// parent, original index, final tree, preparation, and pre-prune plan bytes
// make rollback bounded and deterministic instead of leaving a pruned row with
// no successful authority or overwriting staged-only caller state.
type gateCommitIntent struct {
	Version           uint16               `json:"version"`
	Preparation       artifact.ID          `json:"preparation"`
	PreparationCommit artifact.CommitID    `json:"preparation_commit"`
	Recipe            artifact.ID          `json:"recipe"`
	CandidateManifest artifact.ID          `json:"candidate_manifest"`
	Parent            string               `json:"parent"`
	HeadReference     string               `json:"head_reference"`
	Merge             *gateMergeIntent     `json:"merge,omitempty"`
	PlanProjection    plan.MergeProjection `json:"plan_projection,omitzero"`
	MergeAuthority    []byte               `json:"merge_authority,omitempty"`
	Commit            string               `json:"commit,omitzero"`
	IndexTree         string               `json:"index_tree"`
	Tree              string               `json:"tree"`
	KeepaliveRef      string               `json:"keepalive_ref"`
	KeepaliveTree     string               `json:"keepalive_tree"`
	KeepaliveCommit   string               `json:"keepalive_commit"`
	IndexBefore       []byte               `json:"index_before"`
	IndexAfter        []byte               `json:"index_after"`
	IndexRestore      []byte               `json:"index_restore"`
	IndexMode         uint32               `json:"index_mode"`
	PlanRef           string               `json:"plan_ref"`
	Paths             []string             `json:"paths"`
	Plan              []byte               `json:"plan"`
	AdvancedPlan      []byte               `json:"advanced_plan"`
	PlanMode          uint32               `json:"plan_mode"`
}

var gatePlanRecoveryBeforeSwapHook func(string)

type gateMergeIntent struct {
	IndexTree         string `json:"index_tree"`
	Head              []byte `json:"head"`
	HeadFileMode      uint32 `json:"head_file_mode"`
	Mode              []byte `json:"mode"`
	ModeFileMode      uint32 `json:"mode_file_mode"`
	Message           []byte `json:"message"`
	MessageFileMode   uint32 `json:"message_file_mode"`
	AutoMerge         []byte `json:"auto_merge,omitempty"`
	AutoMergeFileMode uint32 `json:"auto_merge_file_mode,omitzero"`
}

func validGitObjectID(value string) bool {
	return gitauthority.ValidObjectID(value)
}

func currentHeadReference(repo string) (string, error) {
	var stdout, stderr bytes.Buffer
	cmd := newGateGitReaderCommand(repo, "symbolic-ref", "--quiet", "HEAD")
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if err == nil {
		reference := strings.TrimSpace(stdout.String())
		if reference == "" {
			return "", errors.New("gate: cannot resolve the current HEAD reference")
		}
		return reference, nil
	}
	exitError, exitFailure := errors.AsType[*exec.ExitError](err)
	if !exitFailure || exitError.ExitCode() != 1 {
		return "", fmt.Errorf("gate: resolve current HEAD reference: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	if _, verifyErr := gitAuthorityOutput(repo, "rev-parse", "--verify", "HEAD"); verifyErr != nil {
		return "", verifyErr
	}
	return "HEAD", nil
}

func gitMetadataPath(repo, name string) (string, error) {
	paths, err := gitMetadataPaths(repo, []string{name})
	if err != nil {
		return "", err
	}
	return paths[name], nil
}

// Resolve each transaction's paths together. Git remains the authority for
// worktree-local metadata and shared refs; no path survives into another call.
func gitMetadataPaths(repo string, names []string) (map[string]string, error) {
	args := []string{"rev-parse", "--path-format=absolute"}
	for _, name := range names {
		if name == "" || strings.ContainsAny(name, "\x00\r\n") {
			return nil, errors.New("gate: invalid Git metadata name")
		}
		args = append(args, "--git-path", name)
	}
	output, err := command(repo, "git", args...)
	if err != nil {
		return nil, err
	}
	values := strings.Split(strings.TrimSuffix(output, "\n"), "\n")
	if len(values) != len(names) {
		return nil, errors.New("gate: Git metadata path count differs")
	}
	paths := make(map[string]string, len(names))
	for index, name := range names {
		value := strings.TrimSuffix(values[index], "\r")
		if !filepath.IsAbs(value) {
			return nil, errors.New("gate: Git metadata path is not absolute")
		}
		paths[name] = filepath.Clean(value)
	}
	return paths, nil
}

func requireNoGitOperation(repo, message string) error {
	paths, err := gitMetadataPaths(repo, gitOperationMarkers[:])
	if err != nil {
		return err
	}
	for _, name := range gitOperationMarkers {
		if _, err := os.Stat(paths[name]); err == nil {
			return fmt.Errorf(message, name)
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	return nil
}

var gateBeforeCommitStateHook func(string)

func captureGateStartState(repo string) (*gateMergeIntent, gateIndexSnapshot, error) {
	if err := requireFilesGateRefFormat(repo); err != nil {
		return nil, gateIndexSnapshot{}, err
	}
	if err := requireNoGitOperation(repo, "commit admission: Git operation %s is not supported by gate"); err != nil {
		return nil, gateIndexSnapshot{}, err
	}
	if err := requireSupportedMergeRR(repo); err != nil {
		return nil, gateIndexSnapshot{}, err
	}
	merge, err := capturePendingMerge(repo)
	if err != nil {
		return nil, gateIndexSnapshot{}, err
	}
	index, err := captureGateIndex(repo)
	if err != nil {
		return nil, gateIndexSnapshot{}, err
	}
	if merge != nil && merge.IndexTree != index.Tree {
		return nil, gateIndexSnapshot{}, errors.New(
			"commit admission: pending merge index changed while capturing the gate-start state",
		)
	}
	return merge, index, nil
}

func (g *gateContext) requireGateStartState() error {
	if g == nil || g.repo == "" || !validGitObjectID(g.indexBefore.Tree) || len(g.indexBefore.Data) == 0 ||
		g.indexBefore.Mode.Perm() == 0 {
		return errors.New("commit admission: exact gate-start Git state is absent")
	}
	if gateBeforeCommitStateHook != nil {
		gateBeforeCommitStateHook(g.repo)
	}
	merge, index, err := captureGateStartState(g.repo)
	if err != nil {
		return err
	}
	if index.Tree != g.indexBefore.Tree || index.Mode.Perm() != g.indexBefore.Mode.Perm() ||
		!bytes.Equal(index.Data, g.indexBefore.Data) {
		return errors.New("commit admission: Git index moved after the gate-start snapshot")
	}
	if !sameGateMergeState(merge, g.mergeBefore) {
		return errors.New("commit admission: pending merge moved after the gate-start snapshot")
	}
	return nil
}

func sameGateMergeState(left, right *gateMergeIntent) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	return left.IndexTree == right.IndexTree && bytes.Equal(left.Head, right.Head) &&
		bytes.Equal(left.Mode, right.Mode) && bytes.Equal(left.Message, right.Message) &&
		bytes.Equal(left.AutoMerge, right.AutoMerge) &&
		left.HeadFileMode == right.HeadFileMode && left.ModeFileMode == right.ModeFileMode &&
		left.MessageFileMode == right.MessageFileMode && left.AutoMergeFileMode == right.AutoMergeFileMode
}

func capturePendingMerge(repo string) (*gateMergeIntent, error) {
	if err := requireFilesGateRefFormat(repo); err != nil {
		return nil, err
	}
	if err := requireSupportedMergeRR(repo); err != nil {
		return nil, err
	}
	paths, err := gitMetadataPaths(repo, []string{"MERGE_HEAD", "AUTO_MERGE", "MERGE_AUTOSTASH", "MERGE_MODE", "MERGE_MSG"})
	if err != nil {
		return nil, err
	}
	headPath := paths["MERGE_HEAD"]
	headInfo, err := os.Lstat(headPath)
	if errors.Is(err, os.ErrNotExist) {
		if _, autoMergeErr := os.Lstat(paths["AUTO_MERGE"]); autoMergeErr == nil {
			// A stash pop leaves AUTO_MERGE behind with no merge in
			// progress; the gate refuses rather than deleting a marker
			// that could also belong to a concurrent merge, and names
			// the one-line remediation for the common case.
			return nil, errors.New("gate: orphan AUTO_MERGE exists without MERGE_HEAD -- a git stash pop leaves it behind; if no merge is in progress, remove .git/AUTO_MERGE and re-run")
		} else if !errors.Is(autoMergeErr, os.ErrNotExist) {
			return nil, autoMergeErr
		}
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if !headInfo.Mode().IsRegular() {
		return nil, errors.New("gate: pending MERGE_HEAD authority is not a regular file")
	}
	head, err := os.ReadFile(headPath)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(paths["MERGE_AUTOSTASH"]); err == nil {
		return nil, errors.New("gate: pending merge uses MERGE_AUTOSTASH; abort it and stage the merge without autostash")
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	read := func(name string) ([]byte, uint32, error) {
		path := paths[name]
		info, statErr := os.Lstat(path)
		if statErr != nil {
			return nil, 0, statErr
		}
		if !info.Mode().IsRegular() {
			return nil, 0, fmt.Errorf("gate: pending merge metadata %s is not a regular file", name)
		}
		data, readErr := os.ReadFile(path)
		return data, uint32(info.Mode().Perm()), readErr
	}
	mode, modeFileMode, err := read("MERGE_MODE")
	if err != nil {
		return nil, err
	}
	message, messageFileMode, err := read("MERGE_MSG")
	if err != nil {
		return nil, err
	}
	autoMergePath := paths["AUTO_MERGE"]
	autoMergeInfo, err := os.Lstat(autoMergePath)
	var autoMerge []byte
	var autoMergeFileMode uint32
	if errors.Is(err, os.ErrNotExist) {
		autoMerge = nil
	} else if err != nil {
		return nil, err
	} else if !autoMergeInfo.Mode().IsRegular() {
		return nil, errors.New("gate: pending AUTO_MERGE authority is not a regular file")
	} else if autoMerge, err = os.ReadFile(autoMergePath); err != nil {
		return nil, err
	} else {
		autoMergeFileMode = uint32(autoMergeInfo.Mode().Perm())
	}
	index, err := captureGateIndex(repo)
	if err != nil {
		return nil, err
	}
	merge := &gateMergeIntent{
		IndexTree: index.Tree,
		Head:      head, HeadFileMode: uint32(headInfo.Mode().Perm()),
		Mode: mode, ModeFileMode: modeFileMode,
		Message: message, MessageFileMode: messageFileMode,
		AutoMerge: autoMerge, AutoMergeFileMode: autoMergeFileMode,
	}
	if err := merge.validate(""); err != nil {
		return nil, err
	}
	return merge, nil
}

func requireSupportedMergeRR(repo string) error {
	mergeRRPath, err := gitMetadataPath(repo, "MERGE_RR")
	if err != nil {
		return err
	}
	info, err := os.Lstat(mergeRRPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return errors.New("gate: MERGE_RR is not a regular file")
	}
	mergeRR, err := os.ReadFile(mergeRRPath)
	if err != nil {
		return err
	}
	if len(mergeRR) != 0 {
		return errors.New(
			"gate: pending merge has nonempty MERGE_RR; finish or clear rerere state before gate",
		)
	}
	return nil
}

func requireFilesGateRefFormat(repo string) error {
	var stdout, stderr bytes.Buffer
	cmd := newGateGitReaderCommand(repo, "config", "--local", "--get", "extensions.refStorage")
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	format := strings.TrimSpace(stdout.String())
	if err != nil {
		exitError, exitFailure := errors.AsType[*exec.ExitError](err)
		if !exitFailure || exitError.ExitCode() != 1 || format != "" {
			return fmt.Errorf(
				"gate: resolve Git reference format: %w: %s", err, strings.TrimSpace(stderr.String()),
			)
		}
		// Older files-backend repositories predate extensions.refStorage and
		// report a missing config key with exit 1. Reftable repositories name
		// that extension explicitly, so absence is a positive files proof.
		format = "files"
	}
	if strings.ContainsAny(format, "\r\n") || format != "files" {
		return fmt.Errorf(
			"gate: Git reference format %q cannot provide exact AUTO_MERGE file authority; files is required",
			format,
		)
	}
	return nil
}

var errGateMergeMetadataMoved = errors.New("Git merge metadata moved beyond the write-ahead states")

type gateMergeMetadataFile struct {
	name       string
	path       string
	want       []byte
	mode       fs.FileMode
	managed    bool
	allowEmpty bool
}

type gateMergeMetadataState struct {
	present bool
	data    []byte
	mode    fs.FileMode
}

func gateMergeMetadataFiles(repo string, merge *gateMergeIntent) ([]gateMergeMetadataFile, error) {
	files := []gateMergeMetadataFile{
		{name: "MERGE_MODE"},
		{name: "MERGE_MSG"},
		{name: "AUTO_MERGE"},
		// MERGE_RR is unsupported rerere state. An absent or empty file is
		// preserved exactly as Git left it; nonempty content is never consumed,
		// removed, or restored by the gate transaction.
		{name: "MERGE_RR", allowEmpty: true},
		// MERGE_HEAD is the operation-presence marker. It is always the last
		// cleanup removal and the last recovery publication.
		{name: "MERGE_HEAD"},
	}
	if merge != nil {
		files[0].want, files[0].mode, files[0].managed = merge.Mode, fs.FileMode(merge.ModeFileMode), true
		files[1].want, files[1].mode, files[1].managed = merge.Message, fs.FileMode(merge.MessageFileMode), true
		if merge.AutoMerge != nil {
			files[2].want, files[2].mode, files[2].managed = merge.AutoMerge, fs.FileMode(merge.AutoMergeFileMode), true
		}
		files[4].want, files[4].mode, files[4].managed = merge.Head, fs.FileMode(merge.HeadFileMode), true
	}
	names := make([]string, len(files))
	for index, file := range files {
		names[index] = file.name
	}
	paths, err := gitMetadataPaths(repo, names)
	if err != nil {
		return nil, err
	}
	for index := range files {
		files[index].path = paths[files[index].name]
	}
	return files, nil
}

func readGateMergeMetadata(files []gateMergeMetadataFile) ([]gateMergeMetadataState, error) {
	states := make([]gateMergeMetadataState, len(files))
	for index, file := range files {
		info, err := os.Lstat(file.path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() {
			return nil, fmt.Errorf("%w: %s is not a regular file", errGateMergeMetadataMoved, file.name)
		}
		data, err := os.ReadFile(file.path)
		if err != nil {
			return nil, err
		}
		states[index] = gateMergeMetadataState{present: true, data: data, mode: info.Mode().Perm()}
	}
	return states, nil
}

// gateMergeMetadataProgress recognizes the only crash-recoverable states for
// the ordered MODE, MSG, optional AUTO_MERGE, HEAD transaction. A present
// prefix is restoration progress; an absent prefix is cleanup progress.
// Unmanaged metadata must remain absent, except that MERGE_RR may remain an
// untouched empty file.
func gateMergeMetadataProgress(
	files []gateMergeMetadataFile,
	states []gateMergeMetadataState,
	presentPrefix bool,
) (int, bool) {
	var progress int
	if len(files) != len(states) {
		return progress, false
	}
	var managedStates []gateMergeMetadataState
	for index, state := range states {
		file := files[index]
		if !file.managed {
			if state.present && !(file.allowEmpty && len(state.data) == 0) {
				return progress, false
			}
			continue
		}
		if state.present && (!bytes.Equal(state.data, file.want) || state.mode != file.mode.Perm()) {
			return progress, false
		}
		managedStates = append(managedStates, state)
	}
	for progress < len(managedStates) && managedStates[progress].present == presentPrefix {
		progress++
	}
	for index := progress; index < len(managedStates); index++ {
		if managedStates[index].present == presentPrefix {
			return progress, false
		}
	}
	return progress, true
}

func gateGitManualLockPaths(
	repo string,
	mode fs.FileMode,
	intent gateCommitIntent,
) (string, []gateGitLockPath, error) {
	if mode.Perm() == 0 || mode.Perm() != mode {
		return "", nil, errors.New("gate: exact Git index mode is required for merge metadata")
	}
	if !strings.HasPrefix(intent.KeepaliveRef, gateIntentKeepaliveNamespace) ||
		strings.ContainsAny(intent.KeepaliveRef, "\x00\r\n") {
		return "", nil, fmt.Errorf(
			"gate: invalid interrupted commit keepalive lock authority %q", intent.KeepaliveRef,
		)
	}
	indexPath, err := gateIndexPath(repo)
	if err != nil {
		return "", nil, err
	}
	locks := []gateGitLockPath{{name: "index", path: indexPath + ".lock", mode: mode.Perm()}}
	rootRefNames := []string{
		"AUTO_MERGE", "MERGE_HEAD", "MERGE_AUTOSTASH", "CHERRY_PICK_HEAD", "REVERT_HEAD",
		intent.KeepaliveRef,
	}
	paths, err := gitMetadataPaths(repo, rootRefNames)
	if err != nil {
		return "", nil, err
	}
	for _, name := range rootRefNames {
		path := paths[name]
		if name == intent.KeepaliveRef {
			if err := os.MkdirAll(filepath.Dir(path), gatePrivateDirectoryMode); err != nil {
				return "", nil, fmt.Errorf("gate: prepare interrupted commit keepalive lock: %w", err)
			}
		}
		locks = append(locks, gateGitLockPath{
			name: name, path: path + ".lock", mode: gatePrivateFileMode,
		})
	}
	return indexPath, locks, nil
}

func gateGitStateProcessLockPath(repo string) (string, error) {
	output, err := gitAuthorityOutput(repo, "rev-parse", "--git-common-dir")
	if err != nil {
		return "", err
	}
	commonDirectory := strings.TrimSpace(string(output))
	if commonDirectory == "" || strings.ContainsAny(commonDirectory, "\x00\r\n") {
		return "", errors.New("gate: Git common directory is unavailable for state locking")
	}
	if !filepath.IsAbs(commonDirectory) {
		commonDirectory = filepath.Join(repo, commonDirectory)
	}
	commonDirectory = filepath.Clean(commonDirectory)
	if err := os.MkdirAll(commonDirectory, gatePrivateDirectoryMode); err != nil {
		return "", err
	}
	return filepath.Join(commonDirectory, gateGitStateLockFile), nil
}

func inspectGateGitLockMarker(path string, marker []byte, mode fs.FileMode) (bool, bool, error) {
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, false, nil
	}
	if err != nil {
		return false, false, err
	}
	if !info.Mode().IsRegular() || runtime.GOOS != "windows" && info.Mode().Perm() != mode.Perm() ||
		info.Size() != int64(len(marker)) {
		return true, false, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return true, false, err
	}
	return true, bytes.Equal(data, marker), nil
}

func removeExactGateGitLockMarkers(
	locks []gateGitLockPath,
	marker []byte,
	requirePresent bool,
) error {
	toRemove := make([]gateGitLockPath, 0, len(locks))
	for _, lock := range locks {
		present, exact, err := inspectGateGitLockMarker(lock.path, marker, lock.mode)
		if err != nil {
			return err
		}
		if !present {
			if requirePresent {
				return fmt.Errorf("gate: owned Git lock %s disappeared before release", lock.name)
			}
			// fsatomic.Remove is also the idempotent cleanup for a Windows
			// write-through tombstone left after the target name disappeared.
			toRemove = append(toRemove, lock)
			continue
		}
		if !exact {
			return fmt.Errorf(
				"gate: Git lock %s has partial or foreign ownership; automatic recovery refused",
				lock.name,
			)
		}
		toRemove = append(toRemove, lock)
	}
	var err error
	for index := len(toRemove) - 1; index >= 0; index-- {
		if removeErr := fsatomic.Remove(toRemove[index].path); removeErr != nil {
			err = errors.Join(err, fmt.Errorf("remove owned Git lock %s: %w", toRemove[index].name, removeErr))
		}
	}
	return err
}

func publishGateGitLockMarker(lock gateGitLockPath, marker []byte) (err error) {
	directory := filepath.Dir(lock.path)
	prepared, err := os.CreateTemp(directory, ".overgo-gate-lock-marker-*")
	if err != nil {
		return err
	}
	preparedPath := prepared.Name()
	closed := false
	defer func() {
		if !closed {
			err = errors.Join(err, prepared.Close())
		}
		_ = fsatomic.Remove(preparedPath)
	}()
	if err := prepared.Chmod(lock.mode); err != nil {
		return err
	}
	if _, err := prepared.Write(marker); err != nil {
		return err
	}
	if err := prepared.Sync(); err != nil {
		return err
	}
	if err := prepared.Close(); err != nil {
		return err
	}
	closed = true
	if err := os.Link(preparedPath, lock.path); err != nil {
		return err
	}
	if err := fsatomic.SyncFile(lock.path); err != nil {
		return err
	}
	if err := fsatomic.SyncDirectory(directory); err != nil {
		return err
	}
	present, exact, err := inspectGateGitLockMarker(lock.path, marker, lock.mode)
	if err != nil {
		return err
	}
	if !present || !exact {
		return fmt.Errorf("gate: Git lock %s marker publication is inexact", lock.name)
	}
	return nil
}

func acquireGateGitLockMarkers(
	locks []gateGitLockPath,
	marker []byte,
) ([]gateGitLockPath, error) {
	owned := make([]gateGitLockPath, 0, len(locks))
	for _, lock := range locks {
		if err := publishGateGitLockMarker(lock, marker); err != nil {
			present, exact, inspectErr := inspectGateGitLockMarker(lock.path, marker, lock.mode)
			if inspectErr == nil && present && exact {
				owned = append(owned, lock)
			}
			cleanupErr := removeExactGateGitLockMarkers(owned, marker, true)
			return nil, errors.Join(
				fmt.Errorf("gate: acquire exact %s lock: %w", lock.name, err), inspectErr, cleanupErr,
			)
		}
		owned = append(owned, lock)
	}
	return owned, nil
}

func withGateMergeMetadataLock(
	repo string,
	mode fs.FileMode,
	intent gateCommitIntent,
	transaction func(string) error,
) error {
	return withGateGitStateLock(
		repo, mode, intent, gateMergeCASBeforeLockHook, gateMergeCASLockedHook, transaction,
	)
}

// withGateGitStateLock serializes the shared index, merge root refs, and the
// durable intent's keepalive ref. A process-lifetime guard makes exact marker
// residues reclaimable after abrupt process death; marker census refuses any
// partial or foreign contents before deleting even one pathname.
func withGateGitStateLock(
	repo string,
	mode fs.FileMode,
	intent gateCommitIntent,
	beforeLock, locked func(string),
	transaction func(string) error,
) (err error) {
	if err := requireFilesGateRefFormat(repo); err != nil {
		return err
	}
	indexPath, manualLocks, err := gateGitManualLockPaths(repo, mode, intent)
	if err != nil {
		return err
	}
	marker, err := gateGitLockMarkerBytes(repo, indexPath, intent)
	if err != nil {
		return err
	}
	processLockPath, err := gateGitStateProcessLockPath(repo)
	if err != nil {
		return err
	}
	processGuard, err := processlock.Acquire(processLockPath, gatePrivateFileMode)
	if err != nil {
		return fmt.Errorf("gate: another Git-state transaction is active: %w", err)
	}
	defer func() { err = errors.Join(err, processGuard.Close()) }()
	if err := removeExactGateGitLockMarkers(manualLocks, marker, false); err != nil {
		return err
	}
	if beforeLock != nil {
		beforeLock(repo)
	}
	ownedLocks, err := acquireGateGitLockMarkers(manualLocks, marker)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, removeExactGateGitLockMarkers(ownedLocks, marker, true))
	}()
	if locked != nil {
		locked(repo)
	}
	return transaction(indexPath)
}

func gateIndexState(indexPath string, mode fs.FileMode) ([]byte, error) {
	info, err := os.Lstat(indexPath)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Mode().Perm() != mode.Perm() {
		return nil, errGateIndexMoved
	}
	return os.ReadFile(indexPath)
}

func exactGateIndexState(indexPath string, mode fs.FileMode, accepted ...[]byte) error {
	current, err := gateIndexState(indexPath, mode)
	if err != nil {
		return err
	}
	if slices.ContainsFunc(accepted, func(state []byte) bool { return bytes.Equal(current, state) }) {
		return nil
	}
	return errGateIndexMoved
}

func replaceGateIndexStatesUnderLock(
	indexPath string,
	sources, accepted [][]byte,
	target []byte,
	mode fs.FileMode,
) error {
	if len(sources) == 0 || len(accepted) == 0 || len(target) == 0 ||
		mode.Perm() == 0 || mode.Perm() != mode {
		return errors.New("gate: exact locked Git index states and mode are required")
	}
	current, err := gateIndexState(indexPath, mode)
	if err != nil {
		return err
	}
	if slices.ContainsFunc(accepted, func(state []byte) bool { return bytes.Equal(current, state) }) {
		return nil
	}
	if !slices.ContainsFunc(sources, func(state []byte) bool { return bytes.Equal(current, state) }) {
		return errGateIndexMoved
	}
	if err := atomicfile.Write(indexPath, target, mode.Perm()); err != nil {
		return err
	}
	return exactGateIndexState(indexPath, mode, target)
}

func removeGateMergeMetadata(
	repo, operation string,
	files []gateMergeMetadataFile,
	progress int,
) error {
	managed := gateMergeManagedMetadata(files)
	for progress < len(managed) {
		states, err := readGateMergeMetadata(files)
		if err != nil {
			return err
		}
		current, exact := gateMergeMetadataProgress(files, states, false)
		if !exact || current != progress {
			return errGateMergeMetadataMoved
		}
		file := managed[progress]
		if err := fsatomic.Remove(file.path); err != nil {
			return fmt.Errorf("remove %s: %w", file.name, err)
		}
		if gateMergeCASAfterStepHook != nil {
			gateMergeCASAfterStepHook(repo, operation, file.name)
		}
		progress++
	}
	states, err := readGateMergeMetadata(files)
	if err != nil {
		return err
	}
	if current, exact := gateMergeMetadataProgress(files, states, false); !exact || current != len(managed) {
		return errGateMergeMetadataMoved
	}
	return nil
}

func restoreGateMergeMetadata(
	repo string,
	files []gateMergeMetadataFile,
	progress int,
) error {
	managed := gateMergeManagedMetadata(files)
	for progress < len(managed) {
		states, err := readGateMergeMetadata(files)
		if err != nil {
			return err
		}
		current, exact := gateMergeMetadataProgress(files, states, true)
		if !exact || current != progress {
			return errGateMergeMetadataMoved
		}
		file := managed[progress]
		if err := atomicfile.Write(file.path, file.want, file.mode.Perm()); err != nil {
			return fmt.Errorf("restore %s: %w", file.name, err)
		}
		if gateMergeCASAfterStepHook != nil {
			gateMergeCASAfterStepHook(repo, "restore", file.name)
		}
		progress++
	}
	states, err := readGateMergeMetadata(files)
	if err != nil {
		return err
	}
	if current, exact := gateMergeMetadataProgress(files, states, true); !exact || current != len(managed) {
		return errGateMergeMetadataMoved
	}
	return nil
}

func clearCommittedMergeState(repo string, intent gateCommitIntent) error {
	files, err := gateMergeMetadataFiles(repo, intent.Merge)
	if err != nil {
		return err
	}
	err = withGateMergeMetadataLock(repo, fs.FileMode(intent.IndexMode), intent, func(indexPath string) error {
		if err := requireGateIntentKeepalive(repo, intent); err != nil {
			return err
		}
		if err := exactGateIndexState(indexPath, fs.FileMode(intent.IndexMode), intent.IndexAfter); err != nil {
			return err
		}
		if err := requireNoGitOperation(repo, "commit admission: Git operation %s appeared before final state cleanup"); err != nil {
			return err
		}
		states, err := readGateMergeMetadata(files)
		if err != nil {
			return err
		}
		progress, exact := gateMergeMetadataProgress(files, states, false)
		if !exact {
			return errGateMergeMetadataMoved
		}
		return removeGateMergeMetadata(repo, "cleanup", files, progress)
	})
	if errors.Is(err, errGateIndexMoved) {
		return errors.New("commit admission: Git index moved before merge metadata cleanup")
	}
	if errors.Is(err, errGateMergeMetadataMoved) {
		return fmt.Errorf("commit admission: pending merge metadata moved before cleanup: %w", err)
	}
	return err
}

func gateIntentKeepaliveRefValue(repo, reference string) (string, bool, error) {
	var stdout, stderr bytes.Buffer
	process := newGateGitReaderCommand(repo, "rev-parse", "--verify", "--quiet", reference)
	process.Stdout, process.Stderr = &stdout, &stderr
	err := process.Run()
	if err == nil {
		object := strings.TrimSpace(stdout.String())
		if !validGitObjectID(object) {
			return "", false, errors.New("gate: keepalive reference resolved to an invalid object identity")
		}
		return object, true, nil
	}
	if exitError, exitFailure := errors.AsType[*exec.ExitError](err); exitFailure && exitError.ExitCode() == 1 && stdout.Len() == 0 {
		return "", false, nil
	}
	return "", false, fmt.Errorf(
		"gate: resolve interrupted commit keepalive reference: %w: %s",
		err, strings.TrimSpace(stderr.String()),
	)
}

func requireGateIntentKeepaliveObjects(repo string, intent gateCommitIntent) error {
	if _, err := gitAuthorityOutput(repo, "cat-file", "-e", intent.KeepaliveCommit+"^{commit}"); err != nil {
		return fmt.Errorf("gate: interrupted commit keepalive commit is absent: %w", err)
	}
	if _, err := gitAuthorityOutput(repo, "cat-file", "-e", intent.Commit+"^{commit}"); err != nil {
		return fmt.Errorf("gate: interrupted completion commit is absent: %w", err)
	}
	output, err := gitAuthorityOutput(
		repo, "rev-list", "--objects", "--missing=print", intent.KeepaliveTree,
	)
	if err != nil {
		return fmt.Errorf("gate: inspect interrupted commit keepalive objects: %w", err)
	}
	for line := range strings.SplitSeq(string(output), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "?") {
			return errors.New("gate: interrupted commit keepalive object closure is incomplete")
		}
	}
	return nil
}

func requireGateIntentKeepalive(repo string, intent gateCommitIntent) error {
	if intent.KeepaliveRef != gateIntentKeepaliveReference(intent.Preparation) ||
		!validGitObjectID(intent.KeepaliveTree) || !validGitObjectID(intent.KeepaliveCommit) {
		return errors.New("gate: interrupted commit keepalive authority is invalid")
	}
	object, found, err := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef)
	if err != nil {
		return err
	}
	if !found || object != intent.KeepaliveCommit {
		return errors.New("gate: interrupted commit keepalive reference moved from its exact intent")
	}
	return requireGateIntentKeepaliveObjects(repo, intent)
}

func bindGateIntentKeepalive(repo string, intent gateCommitIntent) (bool, error) {
	expectedTree, err := buildGateIntentKeepaliveTree(repo, intent)
	if err != nil {
		return false, err
	}
	expectedCommit, err := buildGateIntentKeepaliveCommit(repo, intent, expectedTree)
	if err != nil {
		return false, err
	}
	if intent.KeepaliveRef != gateIntentKeepaliveReference(intent.Preparation) ||
		intent.KeepaliveTree != expectedTree || intent.KeepaliveCommit != expectedCommit {
		return false, errors.New("gate: interrupted commit keepalive differs from its exact intent")
	}
	if err := requireGateIntentKeepaliveObjects(repo, intent); err != nil {
		return false, err
	}
	current, found, err := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef)
	if err != nil {
		return false, err
	}
	if found {
		if current != intent.KeepaliveCommit {
			return false, errors.New("gate: interrupted commit keepalive reference is owned by another object")
		}
		return false, requireGateIntentKeepalive(repo, intent)
	}
	zero := strings.Repeat("0", len(intent.KeepaliveCommit))
	if _, err := gitAuthorityWriterOutput(
		repo, "update-ref", intent.KeepaliveRef, intent.KeepaliveCommit, zero,
	); err != nil {
		current, found, inspectErr := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef)
		if inspectErr == nil && found && current == intent.KeepaliveCommit {
			return false, requireGateIntentKeepalive(repo, intent)
		}
		return false, errors.Join(err, inspectErr)
	}
	return true, requireGateIntentKeepalive(repo, intent)
}

func ensureGateIntentKeepalive(repo string, intent gateCommitIntent) error {
	_, err := bindGateIntentKeepalive(repo, intent)
	if err != nil {
		return err
	}
	return nil
}

func deleteGateIntentKeepalive(repo string, intent gateCommitIntent) error {
	current, found, err := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef)
	if err != nil || !found {
		return err
	}
	if current != intent.KeepaliveCommit {
		return errors.New("gate: interrupted commit keepalive reference moved before intent resolution")
	}
	if _, err := gitAuthorityWriterOutput(
		repo, "update-ref", "-d", intent.KeepaliveRef, intent.KeepaliveCommit,
	); err != nil {
		return err
	}
	if current, found, err := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef); err != nil {
		return err
	} else if found {
		return fmt.Errorf(
			"gate: interrupted commit keepalive remains at %s after intent resolution", current,
		)
	}
	return nil
}

func writeGateCommitIntent(repo string, intent gateCommitIntent) error {
	if intent.KeepaliveRef == "" && intent.KeepaliveTree == "" && intent.KeepaliveCommit == "" {
		if err := initializeGateIntentKeepalive(repo, &intent); err != nil {
			return err
		}
	}
	if err := intent.validate(); err != nil {
		return err
	}
	if err := validateGateIntentIndexes(repo, intent); err != nil {
		return err
	}
	intentPath := filepath.Join(repo, filepath.FromSlash(gateCommitIntentFile))
	if _, err := os.Stat(intentPath); err == nil {
		var previous gateCommitIntent
		if err := readJSON(repo, gateCommitIntentFile, &previous); err != nil {
			return err
		}
		if err := previous.validate(); err != nil {
			return err
		}
		if previous.KeepaliveRef != intent.KeepaliveRef ||
			previous.KeepaliveTree != intent.KeepaliveTree ||
			previous.KeepaliveCommit != intent.KeepaliveCommit {
			return errors.New("gate: interrupted commit rewrite changed keepalive authority")
		}
		if err := ensureGateIntentKeepalive(repo, previous); err != nil {
			return err
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	_, err := bindGateIntentKeepalive(repo, intent)
	if err != nil {
		return err
	}
	if gateKeepaliveBoundHook != nil {
		gateKeepaliveBoundHook(repo, intent)
	}
	if err := writeJSON(repo, gateCommitIntentFile, intent, gatePrivateFileMode); err != nil {
		// Never delete the ref on an ambiguous publication failure: a post-rename
		// error or racing direct writer may have made the intent durable. A stale
		// exact ref is a bounded leak; a durable intent without it is unrecoverable.
		return err
	}
	return ensureGateIntentKeepalive(repo, intent)
}

func removeGateCommitIntent(repo string) error {
	path := filepath.Join(repo, filepath.FromSlash(gateCommitIntentFile))
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return fsatomic.Remove(path)
	} else if err != nil {
		return err
	}
	var intent gateCommitIntent
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		return err
	}
	if err := intent.validate(); err != nil {
		return err
	}
	if current, found, err := gateIntentKeepaliveRefValue(repo, intent.KeepaliveRef); err != nil {
		return err
	} else if !found || current != intent.KeepaliveCommit {
		return errors.New("gate: interrupted commit keepalive reference moved before intent resolution")
	}
	if err := fsatomic.Remove(path); err != nil {
		return err
	}
	return deleteGateIntentKeepalive(repo, intent)
}

func planBytesMatch(repo string, expected []byte) (bool, error) {
	current, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(plan.Path)))
	if err != nil {
		return false, err
	}
	return bytes.Equal(current, expected), nil
}

func gitCompletionFile(repo, revision, path string) ([]byte, error) {
	return gitAuthorityOutput(repo, "show", revision+":"+path)
}

// projectedMergeBase names the merge base a merge completion reasons
// from. A history where each side merged the other has two bases; the
// target's plan projection reasons from the target's own history, so the
// base on the target's first-parent chain (the target state the incoming
// side merged) is the one, and a history with none or several bases on
// that chain refuses.
func projectedMergeBase(repo, target, incoming string) (string, error) {
	baseOutput, err := gitAuthorityOutput(repo, "merge-base", "--all", target, incoming)
	if err != nil {
		return "", err
	}
	bases := strings.Fields(string(baseOutput))
	if len(bases) == 1 {
		return bases[0], nil
	}
	chainOutput, err := gitAuthorityOutput(repo, "rev-list", "--first-parent", target)
	if err != nil {
		return "", err
	}
	chain := map[string]bool{}
	for revision := range strings.FieldsSeq(string(chainOutput)) {
		chain[revision] = true
	}
	var selected []string
	for _, base := range bases {
		if chain[base] {
			selected = append(selected, base)
		}
	}
	if len(selected) != 1 {
		return "", fmt.Errorf("gate: merge requires one merge base on the target's first-parent chain, found %d of %d", len(selected), len(bases))
	}
	return selected[0], nil
}

func gitAuthorityOutput(repo string, arguments ...string) ([]byte, error) {
	var stdout, stderr bytes.Buffer
	cmd := newGateGitReaderCommand(repo, arguments...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("gate: git %s: %w: %s", strings.Join(arguments, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

func gitAuthorityWriterOutput(repo string, arguments ...string) ([]byte, error) {
	var stdout, stderr bytes.Buffer
	cmd := newGateGitWriterCommand(repo, nil, arguments...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("gate: git %s: %w: %s", strings.Join(arguments, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

func projectedMergeAuthorityFromIntent(
	intent gateCommitIntent,
) (*plan.FirstParentTargetMergeAuthority, error) {
	if intent.PlanProjection != plan.MergeProjectionFirstParentTarget {
		if len(intent.MergeAuthority) != 0 {
			return nil, errors.New("gate: semantic-union intent carries projected merge authority")
		}
		return nil, nil
	}
	receipt, err := plan.ParseFirstParentTargetMergeAuthority(intent.MergeAuthority)
	if err != nil {
		return nil, fmt.Errorf("gate: parse projected merge authority: %w", err)
	}
	return &receipt, nil
}

func verifyProspectiveGateCompletion(
	repo string,
	intent gateCommitIntent,
	preAdvance, child plan.Plan,
	message string,
	store *overgodb.Store,
) error {
	mergeAuthority, err := projectedMergeAuthorityFromIntent(intent)
	if err != nil {
		return err
	}
	parentIDs := []string{intent.Parent}
	mergeParent := ""
	if intent.Merge != nil {
		for _, parentID := range intent.Merge.parents() {
			if mergeParent != "" {
				return errors.New("gate: completion merge requires one merge parent")
			}
			mergeParent = parentID
		}
		if !validGitObjectID(mergeParent) || mergeParent == intent.Parent {
			return errors.New("gate: completion merge requires one distinct merge parent")
		}
		parentIDs = append(parentIDs, mergeParent)
	}
	parents := make([]plan.Plan, 0, len(parentIDs))
	for _, parentID := range parentIDs {
		data, err := gitCompletionFile(repo, parentID, plan.Path)
		if err != nil {
			return err
		}
		document, err := plan.ParseHistorical(data)
		if err != nil {
			return fmt.Errorf("gate: parse parent plan %.12s: %w", parentID, err)
		}
		parents = append(parents, document)
	}
	var mergeBase *plan.Plan
	if intent.Merge != nil {
		mergeBaseRevision, err := projectedMergeBase(repo, intent.Parent, mergeParent)
		if err != nil {
			return err
		}
		data, err := gitCompletionFile(repo, mergeBaseRevision, plan.Path)
		if err != nil {
			return err
		}
		base, err := plan.ParseHistorical(data)
		if err != nil {
			return fmt.Errorf("gate: parse merge-base plan %.12s: %w", mergeBaseRevision, err)
		}
		mergeBase = &base
		if store == nil {
			return errors.New("gate: completion merge requires the locked authority store")
		}
		if intent.PlanProjection == plan.MergeProjectionFirstParentTarget {
			if mergeAuthority == nil {
				return errors.New("gate: first-parent-target completion lacks projected merge authority")
			}
			item, step, found := strings.Cut(intent.PlanRef, "/")
			if !found {
				return errors.New("gate: projected completion has an invalid plan reference")
			}
			if err := plan.VerifyFirstParentTargetMergeAuthorityTransition(
				intent.Parent, mergeParent, mergeBaseRevision, parents[0], parents[1], base,
				preAdvance, child, item, step, intent.Preparation, intent.PreparationCommit,
				*mergeAuthority,
			); err != nil {
				return fmt.Errorf("gate: audit projected merge receipt: %w", err)
			}
			localAuthority, err := plan.ResolveCompletionAuthority(
				context.Background(), repo, intent.Parent, parents[0], store,
			)
			if err != nil {
				return fmt.Errorf("gate: audit local completion merge parent %.12s: %w", intent.Parent, err)
			}
			if !localAuthority.ProtectsRevision() {
				return fmt.Errorf("gate: local completion merge parent %.12s is outside the protected epoch", intent.Parent)
			}
			if err := plan.VerifyFirstParentTargetLocalAuthority(
				repo, intent.Parent, parents[0], preAdvance, child, localAuthority,
			); err != nil {
				return fmt.Errorf("gate: audit first-parent target authority: %w", err)
			}
			return plan.VerifyProspectiveCompletionTransitionWithProjection(
				parents, mergeBase, preAdvance, child, message, intent.PlanProjection,
			)
		}
		parentAuthorities := make([]plan.CompletionAuthority, len(parentIDs))
		for index, parentID := range parentIDs {
			authority, err := plan.ResolveCompletionAuthority(
				context.Background(), repo, parentID, parents[index], store,
			)
			if err != nil {
				return fmt.Errorf("gate: audit completion merge parent %.12s: %w", parentID, err)
			}
			if !authority.ProtectsRevision() {
				return fmt.Errorf(
					"gate: completion merge parent %.12s is outside the protected epoch; rebase the merge source",
					parentID,
				)
			}
			parentAuthorities[index] = authority
		}
		baseAuthority, err := plan.ResolveCompletionAuthority(
			context.Background(), repo, mergeBaseRevision, base, store,
		)
		if err != nil {
			return fmt.Errorf("gate: audit completion merge base %.12s: %w", mergeBaseRevision, err)
		}
		if !baseAuthority.ProtectsRevision() {
			return errors.New("gate: completion merge parents do not share a protected epoch; rebase the merge source")
		}
		if err := plan.VerifyProspectiveMergeAuthorityWithProjection(
			repo, parentIDs[0], parentIDs[1],
			parents[0], parents[1], preAdvance,
			parentAuthorities[0], parentAuthorities[1],
			intent.PlanProjection,
		); err != nil {
			return fmt.Errorf("gate: audit prospective completion merge: %w", err)
		}
	}
	return plan.VerifyProspectiveCompletionTransitionWithProjection(
		parents, mergeBase, preAdvance, child, message, intent.PlanProjection,
	)
}

var gitOperationMarkers = [...]string{
	"MERGE_AUTOSTASH", "SQUASH_MSG", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-apply", "rebase-merge", "sequencer",
}

// normalizeGateIndexToStates restores the exact write-ahead index bytes when
// a concurrent read-only observer — a `git status` from another session —
// has opportunistically refreshed the stat cache. A refresh rewrites entry
// timestamps but never the tree the index resolves to, so when the live
// bytes differ from every captured state yet write-tree to the same tree
// authority as one of them, that state's exact bytes are restored under the
// held gate lock; an index whose tree matches no captured state is genuine
// drift and stays untouched for the byte comparison to refuse.
func normalizeGateIndexToStates(repo, indexPath string, mode fs.FileMode, states ...[]byte) error {
	current, err := gateIndexState(indexPath, mode)
	if err != nil {
		return err
	}
	for _, state := range states {
		if bytes.Equal(current, state) {
			return nil
		}
	}
	currentTree, err := indexTreeForBytes(repo, current)
	if err != nil {
		return err
	}
	for _, state := range states {
		stateTree, err := indexTreeForBytes(repo, state)
		if err != nil {
			return err
		}
		if stateTree == currentTree {
			if err := atomicfile.Write(indexPath, state, mode.Perm()); err != nil {
				return err
			}
			return exactGateIndexState(indexPath, mode, state)
		}
	}
	return nil
}

func removeGateCommitIntentForPreparation(repo string, preparation artifact.ID) error {
	_, found, err := readGateCommitIntentForPreparation(repo, preparation)
	if err != nil || !found {
		return err
	}
	return removeGateCommitIntent(repo)
}

func readGateCommitIntentForPreparation(
	repo string,
	preparation artifact.ID,
) (gateCommitIntent, bool, error) {
	path := filepath.Join(repo, filepath.FromSlash(gateCommitIntentFile))
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return gateCommitIntent{}, false, nil
	} else if err != nil {
		return gateCommitIntent{}, false, err
	}
	var intent gateCommitIntent
	if err := readJSON(repo, gateCommitIntentFile, &intent); err != nil {
		return gateCommitIntent{}, false, err
	}
	if err := intent.validate(); err != nil {
		return gateCommitIntent{}, false, err
	}
	if err := validateGateIntentIndexes(repo, intent); err != nil {
		return gateCommitIntent{}, false, err
	}
	if intent.Preparation != preparation {
		return gateCommitIntent{}, false, errors.New("gate: record debt and interrupted commit name different preparations")
	}
	return intent, true, nil
}
