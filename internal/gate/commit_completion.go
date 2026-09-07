package gate

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/overgodb"
	"overgo/internal/plan"
)

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
		mergeBaseRevision, err := plan.CompletionMergeBase(context.Background(), repo, intent.Parent, mergeParent)
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
