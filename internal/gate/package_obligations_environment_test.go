package gate

import (
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestPackageObligationsPublishAnUnseenEnvironment prepares obligations in a
// store that has never seen the run's environment, the way the deferred lanes
// re-run without a preparation, then records the environment with its content
// the way the lane outcome does. The obligations once named the environment
// bare, and the outcome's commit then conflicted with it.
func TestPackageObligationsPublishAnUnseenEnvironment(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "lane", OS: "windows", Arch: "amd64", Device: "cpu", Backend: "host", Driver: "process",
	})
	if err != nil {
		t.Fatal(err)
	}
	document, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	ledger := &packageEvidenceLedger{store: store, environment: environment.ID, environmentDocument: document,
		obligations: map[string]runrecord.AgentObligation{}, listings: map[string]artifact.ID{}, previous: map[string]artifact.ID{}}
	input, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("package input"))
	if err != nil {
		t.Fatal(err)
	}
	cache := automationcheck.NewEvidenceCache(environment.ID)
	if err := ledger.prepare(ctx, []string{"example/app"}, "short", map[string]artifact.ID{"example/app": input}, &cache); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CommitAs(ctx, gateProducer, artifact.Batch{
		Key: "lane/outcome/environment", Contents: []artifact.Content{document},
	}); err != nil && !errors.Is(err, overgodb.ErrNoChange) {
		t.Fatalf("recording the environment after the obligations: %v", err)
	}
	descriptor, found, err := store.Artifact(ctx, environment.ID)
	if err != nil || !found || descriptor.Size == 0 {
		t.Fatalf("environment = %+v found=%t err=%v; want its published document", descriptor, found, err)
	}
}
