package main

import (
	"context"
	"fmt"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

func admitStoredReview(repo, storePath, verdictText, targetHead string) error {
	verdictID, err := artifact.ParseID(verdictText)
	if err != nil || verdictID.Kind() != artifact.KindEvidence {
		return fmt.Errorf("gate: invalid review verdict ID %q", verdictText)
	}
	store, err := repodb.OpenReadOnly(filepath.Join(repo, storePath))
	if err != nil {
		return fmt.Errorf("gate: open review store: %w", err)
	}
	defer store.Close()
	ctx := context.Background()
	verdict, err := reviewDocument(ctx, store, verdictID, runrecord.ReviewVerdictMediaType, runrecord.ReviewVerdictSchema, runrecord.ParseReviewVerdict)
	if err != nil {
		return err
	}
	candidate, err := reviewDocument(ctx, store, verdict.Candidate, runrecord.ReviewCandidateMediaType, runrecord.ReviewCandidateSchema, runrecord.ParseReviewCandidate)
	if err != nil {
		return err
	}
	developer, err := reviewDocument(ctx, store, candidate.Developer, runrecord.ReviewActorMediaType, runrecord.ReviewActorSchema, runrecord.ParseReviewActor)
	if err != nil {
		return err
	}
	reviewer, err := reviewDocument(ctx, store, verdict.Reviewer, runrecord.ReviewActorMediaType, runrecord.ReviewActorSchema, runrecord.ParseReviewActor)
	if err != nil {
		return err
	}
	developerWorktree, err := reviewDocument(ctx, store, candidate.Worktree, runrecord.ReviewWorktreeMediaType, runrecord.ReviewWorktreeSchema, runrecord.ParseReviewWorktree)
	if err != nil {
		return err
	}
	reviewWorktree, err := reviewDocument(ctx, store, verdict.Worktree, runrecord.ReviewWorktreeMediaType, runrecord.ReviewWorktreeSchema, runrecord.ParseReviewWorktree)
	if err != nil {
		return err
	}
	evaluator, err := reviewDocument(ctx, store, verdict.Evaluator, runrecord.ReviewEvaluatorMediaType, runrecord.ReviewEvaluatorSchema, runrecord.ParseReviewEvaluator)
	if err != nil {
		return err
	}
	findings := make([]runrecord.ReviewFinding, 0, len(verdict.Findings))
	for _, id := range verdict.Findings {
		finding, err := reviewDocument(ctx, store, id, runrecord.ReviewFindingMediaType, runrecord.ReviewFindingSchema, runrecord.ParseReviewFinding)
		if err != nil {
			return err
		}
		findings = append(findings, finding)
	}
	return runrecord.AdmitReview(targetHead, runrecord.ReviewAdmission{
		Developer: developer, Reviewer: reviewer, DeveloperWorktree: developerWorktree, ReviewWorktree: reviewWorktree,
		Evaluator: evaluator, Candidate: candidate, Findings: findings, Verdict: verdict,
	})
}

func reviewDocument[T any](ctx context.Context, store *repodb.Store, id artifact.ID, mediaType, schema string, parse func([]byte) (T, error)) (T, error) {
	var zero T
	content, ok, err := store.Content(ctx, id)
	if err != nil {
		return zero, fmt.Errorf("gate: load review document %s: %w", id, err)
	}
	if !ok || content.Descriptor.MediaType != mediaType || content.Descriptor.Schema != schema {
		return zero, fmt.Errorf("gate: review document %s is absent or has the wrong type", id)
	}
	value, err := parse(content.Data)
	if err != nil {
		return zero, fmt.Errorf("gate: parse review document %s: %w", id, err)
	}
	return value, nil
}
