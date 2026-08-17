package repodb

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestPromotionReversibleAndDrillRestoresChampion pins the rollback
// invariant: champions are immutable content the store retains through any
// promotion; the champion alias moves only by compare-and-set (a stale
// expectation is refused, never last-writer-wins); every promotion and
// rollback decision carries its decider identity; and the drill demonstrably
// restores the prior champion -- same alias target, byte-identical content --
// after a promotion is reversed.
func TestPromotionReversibleAndDrillRestoresChampion(t *testing.T) {
	ctx := context.Background()
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	modelContent := func(payload string) artifact.Content {
		return artifact.Content{
			Descriptor: artifact.Descriptor{
				ID:        testutil.ArtifactID(t, artifact.KindModel, payload),
				Size:      uint64(len(payload)),
				MediaType: "application/octet-stream",
			},
			Data: []byte(payload),
		}
	}
	champion := modelContent("champion-weights-v1")
	challenger := modelContent("challenger-weights-v2")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "drill/models",
		Contents: []artifact.Content{champion, challenger},
	}); err != nil {
		t.Fatal(err)
	}
	const alias = "champion/inference-lane"
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:     "drill/enthrone",
		Aliases: []artifact.AliasBinding{{Name: alias, Target: champion.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}

	decider := recipe.Decider{
		CodeCommit: "0123456789abcdef0123456789abcdef01234567",
		Derivation: testutil.ArtifactID(t, artifact.KindEvidence, "promotion-authority"),
	}
	promotion, err := recipe.NewDecision(
		challenger.Descriptor.ID, recipe.DecisionAccepted, recipe.EvidenceProduction,
		"challenger promoted over champion", decider,
		[]artifact.ID{champion.Descriptor.ID},
	)
	if err != nil {
		t.Fatal(err)
	}
	if promotion.Decider != decider {
		t.Fatal("promotion decision lost its decider identity")
	}
	priorChampion := champion.Descriptor.ID
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:     "drill/promote",
		Aliases: []artifact.AliasBinding{{Name: alias, Target: challenger.Descriptor.ID, Previous: &priorChampion}},
	}); err != nil {
		t.Fatal(err)
	}
	current, ok, err := store.ResolveAlias(ctx, alias)
	if err != nil || !ok || current != challenger.Descriptor.ID {
		t.Fatalf("promoted alias = (%v, %v, %v)", current, ok, err)
	}

	// Immutable champion retention: promotion moved the alias, never the
	// content -- the prior champion's bytes remain committed and identical.
	retained, ok, err := store.Content(ctx, champion.Descriptor.ID)
	if err != nil || !ok || !bytes.Equal(retained.Data, champion.Data) {
		t.Fatalf("prior champion not retained: (%v, %v)", ok, err)
	}

	// Compare-and-set: a stale promotion expecting the old champion is
	// refused rather than clobbering the current one.
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:     "drill/stale-promote",
		Aliases: []artifact.AliasBinding{{Name: alias, Target: champion.Descriptor.ID, Previous: &priorChampion}},
	}); !errors.Is(err, ErrAliasConflict) {
		t.Fatalf("stale compare-and-set = %v, want ErrAliasConflict", err)
	}
	// An unbound-name claim against a bound alias is refused too.
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:     "drill/blind-promote",
		Aliases: []artifact.AliasBinding{{Name: alias, Target: champion.Descriptor.ID}},
	}); !errors.Is(err, ErrAliasConflict) {
		t.Fatalf("blind rebind = %v, want ErrAliasConflict", err)
	}

	// The drill: reverse the promotion with a decision carrying the decider
	// identity, compare-and-set the alias back, and prove restoration.
	rollback, err := recipe.NewDecision(
		champion.Descriptor.ID, recipe.DecisionAccepted, recipe.EvidenceProduction,
		"rollback drill restores the prior champion", decider,
		[]artifact.ID{promotion.ID},
	)
	if err != nil {
		t.Fatal(err)
	}
	rollbackContent, err := rollback.Content()
	if err != nil {
		t.Fatal(err)
	}
	promoted := challenger.Descriptor.ID
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "drill/rollback",
		Contents: []artifact.Content{rollbackContent},
		Aliases:  []artifact.AliasBinding{{Name: alias, Target: champion.Descriptor.ID, Previous: &promoted}},
	}); err != nil {
		t.Fatal(err)
	}
	restored, ok, err := store.ResolveAlias(ctx, alias)
	if err != nil || !ok || restored != champion.Descriptor.ID {
		t.Fatalf("rollback alias = (%v, %v, %v), want the prior champion", restored, ok, err)
	}
	content, ok, err := store.Content(ctx, restored)
	if err != nil || !ok || !bytes.Equal(content.Data, champion.Data) {
		t.Fatal("restored champion content differs from the original")
	}
	storedDecision, ok, err := store.Content(ctx, rollback.ID)
	if err != nil || !ok {
		t.Fatalf("rollback decision not durable: (%v, %v)", ok, err)
	}
	parsed, err := recipe.ParseDecision(storedDecision.Data)
	if err != nil || parsed.Decider != decider || parsed.Subject != champion.Descriptor.ID {
		t.Fatalf("rollback decision = (%+v, %v), want decider identity and champion subject", parsed, err)
	}
}
