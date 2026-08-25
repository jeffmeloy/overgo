package agenttool

import (
	"context"
	"testing"

	"overgo/internal/overgodb"
)

// TestArgvPolicyGatesPublicationAndInvocation pins the durable
// allowlist: an argv manual publishes only under a committed policy
// naming its program, an absent policy refuses (fail closed), and a
// policy that later drops the program refuses the already-published
// manual at invocation admission.
func TestArgvPolicyGatesPublicationAndInvocation(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	manual, err := NewManual(Manual{
		Name: "probe.exec", Description: "Run the fixture program.",
		Effect:    EffectMutation,
		Transport: Transport{Kind: TransportArgv, Program: "git", Args: []string{"status"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishManualCatalog(ctx, store, []Manual{manual}); err == nil {
		t.Fatal("argv manual published with NO committed policy")
	}
	if _, err := PublishArgvPolicy(ctx, store, []string{"go"}); err != nil {
		t.Fatal(err)
	}
	if _, err := PublishManualCatalog(ctx, store, []Manual{manual}); err == nil {
		t.Fatal("argv manual published outside the committed policy")
	}
	if _, err := PublishArgvPolicy(ctx, store, []string{"git", "go"}); err != nil {
		t.Fatal(err)
	}
	if _, err := PublishManualCatalog(ctx, store, []Manual{manual}); err != nil {
		t.Fatalf("allowlisted argv manual refused: %v", err)
	}
	if err := CheckArgvAuthority(ctx, store, manual); err != nil {
		t.Fatalf("invocation admission refused an allowlisted program: %v", err)
	}
	// Tightening the policy refuses the already-published manual at the
	// invocation boundary -- publication history cannot outrun policy.
	if _, err := PublishArgvPolicy(ctx, store, []string{"go"}); err != nil {
		t.Fatal(err)
	}
	if err := CheckArgvAuthority(ctx, store, manual); err == nil {
		t.Fatal("invocation admitted a program the current policy dropped")
	}
	// Non-argv manuals never consult the policy.
	builtin, err := NewManual(Manual{
		Name: "probe.pure", Description: "Inspect nothing.",
		Effect: EffectInspection, Transport: Transport{Kind: TransportBuiltin},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := CheckArgvAuthority(ctx, store, builtin); err != nil {
		t.Fatalf("non-argv manual consulted the argv policy: %v", err)
	}
}
