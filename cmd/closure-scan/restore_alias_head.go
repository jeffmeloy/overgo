package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
	"overgo/internal/repoanalysis"
)

const (
	closureAliasRestoreSchema    = "overgo/closure-alias-restore/v1"
	unreviewedRestoreCount       = -1
	restoreCoordinateResultLimit = 2
)

var closureAliasRestoreContract = artifact.JSONContract(artifact.KindEvidence, closureAliasRestoreSchema)

type closureAliasRestoreSelector struct {
	SourceHead              artifact.CommitID
	SourceSequence          uint64
	ExpectedHead            artifact.CommitID
	ExpectedSequence        uint64
	ExpectedAliases         int
	ExpectedChanges         int
	ExpectedStale           int
	ExpectedAuthorityDigest string
	Confirm                 bool
}

type closureAliasRestoreResult struct {
	SourceHead       string      `json:"source_head"`
	SourceSequence   uint64      `json:"source_sequence"`
	ExpectedHead     string      `json:"expected_head"`
	ExpectedSequence uint64      `json:"expected_sequence"`
	DesiredAliases   int         `json:"desired_aliases"`
	CurrentAliases   int         `json:"current_aliases"`
	Changes          int         `json:"changes"`
	PredictedStale   int         `json:"predicted_stale_bindings"`
	AuthorityError   string      `json:"predicted_authority_error,omitempty"`
	AuthorityDigest  string      `json:"authority_digest"`
	Confirmed        bool        `json:"confirmed"`
	Record           artifact.ID `json:"record,omitzero"`
	Commit           string      `json:"commit,omitempty"`
}

type closureAliasRestoreRecord struct {
	Version          uint16 `json:"version"`
	SourceHead       string `json:"source_head"`
	SourceSequence   uint64 `json:"source_sequence"`
	ReplacedHead     string `json:"replaced_head"`
	ReplacedSequence uint64 `json:"replaced_sequence"`
	DesiredAliases   int    `json:"desired_aliases"`
	Changes          int    `json:"changes"`
	PredictedStale   int    `json:"predicted_stale_bindings"`
	AuthorityDigest  string `json:"authority_digest"`
}

func (selector closureAliasRestoreSelector) validate() error {
	if !selector.SourceHead.Valid() || selector.SourceSequence == 0 ||
		!selector.ExpectedHead.Valid() || selector.ExpectedSequence == 0 ||
		selector.SourceSequence > selector.ExpectedSequence {
		return errors.New("closure-scan: invalid reviewed alias-restore selector")
	}
	if selector.Confirm && (selector.ExpectedAliases < 0 || selector.ExpectedChanges < 0 || selector.ExpectedStale != 0) {
		return errors.New("closure-scan: confirmed alias restore requires reviewed nonnegative alias and change counts and zero stale bindings")
	}
	if selector.Confirm && !validClosureAuthorityDigest(selector.ExpectedAuthorityDigest) {
		return errors.New("closure-scan: confirmed alias restore requires an exact lowercase SHA-256 authority digest")
	}
	return nil
}

func validClosureAuthorityDigest(value string) bool {
	if len(value) != hex.EncodedLen(sha256.Size) {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && hex.EncodeToString(decoded) == value
}

func closureAuthorityDigest(authorityError string) string {
	digest := sha256.Sum256([]byte(authorityError))
	return hex.EncodeToString(digest[:])
}

// restoreClosureAliasesAtReviewedHead replays only the closure alias facet at
// one exact historical coordinate. Immutable documents and unrelated commits
// remain intact. Without Confirm it is a read-only prediction.
func restoreClosureAliasesAtReviewedHead(
	ctx context.Context,
	storePath string,
	snapshot repoanalysis.SourceSnapshot,
	selector closureAliasRestoreSelector,
) (result closureAliasRestoreResult, finalErr error) {
	if ctx == nil {
		return result, errors.New("closure-scan: nil reviewed alias-restore context")
	}
	if err := selector.validate(); err != nil {
		return result, err
	}
	var store *overgodb.Store
	var err error
	if selector.Confirm {
		store, err = overgodb.Open(storePath)
	} else {
		store, err = overgodb.OpenReadOnly(storePath)
	}
	if err != nil {
		return result, err
	}
	defer func() {
		closeErr := store.Close()
		if result.Commit == "" {
			finalErr = errors.Join(finalErr, closeErr)
		}
	}()

	head, sequence := store.Head()
	if head != selector.ExpectedHead || sequence != selector.ExpectedSequence {
		return result, fmt.Errorf(
			"closure-scan: alias-restore store head is stale: have %s@%d, reviewed %s@%d",
			head, sequence, selector.ExpectedHead, selector.ExpectedSequence,
		)
	}
	if err := requireClosureRestoreSource(ctx, store, selector.SourceHead, selector.SourceSequence); err != nil {
		return result, err
	}
	desired, err := closureAliasMapAt(ctx, store, selector.SourceSequence)
	if err != nil {
		return result, err
	}
	documents, err := allClosureDocuments(ctx, store)
	if err != nil {
		return result, err
	}
	if err := requireCanonicalClosureAliasTargets(desired, documents); err != nil {
		return result, err
	}
	current, err := currentClosureAliasMap(ctx, store)
	if err != nil {
		return result, err
	}
	changes := closureAliasDelta(current, desired)
	issues, err := closurescan.ValidateActiveBindings(snapshot, documents, desired)
	if err != nil {
		return result, err
	}
	authorityError := ""
	if _, err := closurescan.ValidatePermanentActiveAuthority(snapshot, documents, desired); err != nil {
		authorityError = err.Error()
	}
	authorityDigest := closureAuthorityDigest(authorityError)
	result = closureAliasRestoreResult{
		SourceHead: selector.SourceHead.String(), SourceSequence: selector.SourceSequence,
		ExpectedHead: selector.ExpectedHead.String(), ExpectedSequence: selector.ExpectedSequence,
		DesiredAliases: len(desired), CurrentAliases: len(current), Changes: len(changes),
		PredictedStale: len(issues), AuthorityError: authorityError, AuthorityDigest: authorityDigest,
		Confirmed: selector.Confirm,
	}
	if !selector.Confirm {
		return result, nil
	}
	if len(desired) != selector.ExpectedAliases || len(changes) != selector.ExpectedChanges ||
		len(issues) != selector.ExpectedStale {
		return result, fmt.Errorf(
			"closure-scan: alias-restore review changed: aliases=%d/%d changes=%d/%d stale=%d/%d",
			len(desired), selector.ExpectedAliases, len(changes), selector.ExpectedChanges,
			len(issues), selector.ExpectedStale,
		)
	}
	if authorityDigest != selector.ExpectedAuthorityDigest {
		return result, fmt.Errorf(
			"closure-scan: alias-restore authority digest changed: have %s, reviewed %s",
			authorityDigest, selector.ExpectedAuthorityDigest,
		)
	}
	if len(changes) == 0 {
		return result, nil
	}
	record := closureAliasRestoreRecord{
		Version:    artifact.InitialDocumentVersion,
		SourceHead: selector.SourceHead.String(), SourceSequence: selector.SourceSequence,
		ReplacedHead: selector.ExpectedHead.String(), ReplacedSequence: selector.ExpectedSequence,
		DesiredAliases: len(desired), Changes: len(changes), PredictedStale: len(issues),
		AuthorityDigest: authorityDigest,
	}
	content, err := artifact.JSONContent(closureAliasRestoreContract, record)
	if err != nil {
		return result, err
	}
	expected := selector.ExpectedHead
	batch := artifact.Batch{ExpectedHead: &expected, Contents: []artifact.Content{content}, Aliases: changes}
	if err := bindClosureOperationKey(closureRestoreReviewedHeadOperation, &batch); err != nil {
		return result, err
	}
	commit, err := store.Commit(ctx, batch)
	if err != nil {
		return result, err
	}
	result.Record = content.Descriptor.ID
	result.Commit = commit.String()
	return result, nil
}

func requireClosureRestoreSource(
	ctx context.Context,
	store *overgodb.Store,
	want artifact.CommitID,
	sequence uint64,
) error {
	result, err := store.Query(ctx, overgodb.Query{
		FromSequence: sequence, ToSequence: sequence, MaxResults: restoreCoordinateResultLimit,
		Projection: overgodb.ProjectCommits,
	})
	if err != nil {
		return err
	}
	if len(result.Commits) != 1 || result.Commits[0].Sequence != sequence || result.Commits[0].ID != want {
		return errors.New("closure-scan: reviewed alias-restore source coordinate does not exist")
	}
	return nil
}

func requireCanonicalClosureAliasTargets(
	aliases map[string]artifact.ID,
	documents []closureledger.Document,
) error {
	byID := make(map[artifact.ID]closureledger.Document, len(documents))
	for _, document := range documents {
		byID[document.ID] = document
	}
	names := make([]string, 0, len(aliases))
	for name := range aliases {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		target := aliases[name]
		document, found := byID[target]
		if !found {
			return fmt.Errorf(
				"closure-scan: reviewed alias %q target is not a closure document: %s",
				name, target,
			)
		}
		matches := 0
		for _, binding := range document.Bindings {
			canonical, err := closureledger.ActiveAlias(binding)
			if err != nil {
				return err
			}
			if canonical == name {
				matches++
			}
		}
		if matches != 1 {
			return fmt.Errorf(
				"closure-scan: reviewed alias %q target %s has %d canonical bindings for that alias",
				name, target, matches,
			)
		}
	}
	return nil
}

func closureAliasMapAt(ctx context.Context, store *overgodb.Store, sequence uint64) (map[string]artifact.ID, error) {
	aliases := map[string]artifact.ID{}
	err := store.VisitAliasEvents(ctx, overgodb.AliasEventRange{
		Prefix: closureledger.ActiveAliasPrefix, ToSequence: sequence,
	}, func(event overgodb.AliasEvent) error {
		if event.Binding.Remove {
			delete(aliases, event.Binding.Name)
		} else {
			aliases[event.Binding.Name] = event.Binding.Target
		}
		return nil
	})
	return aliases, err
}

func currentClosureAliasMap(ctx context.Context, store *overgodb.Store) (map[string]artifact.ID, error) {
	aliases := map[string]artifact.ID{}
	err := store.VisitAliases(ctx, closureledger.ActiveAliasPrefix, func(view overgodb.AliasView) error {
		aliases[view.Name] = view.Target
		return nil
	})
	return aliases, err
}

func closureAliasDelta(current, desired map[string]artifact.ID) []artifact.AliasBinding {
	names := make([]string, 0, len(current)+len(desired))
	for name := range current {
		names = append(names, name)
	}
	for name := range desired {
		if _, found := current[name]; !found {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	changes := make([]artifact.AliasBinding, 0)
	for _, name := range names {
		before, hadBefore := current[name]
		after, hasAfter := desired[name]
		switch {
		case hadBefore && hasAfter && before == after:
			continue
		case hadBefore && !hasAfter:
			changes = append(changes, artifact.AliasBinding{
				Name: name, Target: before, Previous: artifact.IDPointer(before), Remove: true,
			})
		case hadBefore:
			changes = append(changes, artifact.AliasBinding{
				Name: name, Target: after, Previous: artifact.IDPointer(before),
			})
		default:
			changes = append(changes, artifact.AliasBinding{Name: name, Target: after})
		}
	}
	return changes
}
