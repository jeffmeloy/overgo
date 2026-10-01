package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/gguf"
	"overgo/internal/mediacapability"
	"overgo/internal/modelartifact"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestExactVerificationCandidateBinding(t *testing.T) {
	path := testutil.HermeticLlamaGGUF(t, 32)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	if _, err := prepareExactCandidate(ctx, store, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative); err == nil {
		t.Fatal("missing registered profile accepted")
	}
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
		t.Fatal(err)
	}
	want, err := modelintake.PrepareInferenceCandidate(ctx, store, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
	if err != nil {
		t.Fatal(err)
	}
	check := func() {
		t.Helper()
		candidate, err := prepareExactCandidate(ctx, store, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
		if err != nil {
			t.Fatal(err)
		}
		loaded, err := modelrecipe.ResolveCandidateGGUF(path, candidate.Definition, candidate.Resolved)
		if err != nil {
			t.Fatal(err)
		}
		defer loaded.Close()
		identity, err := loaded.Identity()
		if err != nil || candidate.Definition.ID != want.Definition.ID || identity.Recipe != want.Definition.ID ||
			identity.Profile != want.Resolved.Profile.ID || identity.Definition != want.Resolved.Document.ID {
			t.Fatalf("candidate binding differs: %+v, %v", identity, err)
		}
	}
	check()
	if _, active, err := modelrecipe.ActiveRecord(ctx, store, want.Inventory.Manifest.ID, recipe.TaskInference); err != nil || active {
		t.Fatalf("verification activated a candidate: %v", err)
	}
	// A prior active recipe deliberately uses a different decode session.
	prior, err := modelintake.PrepareInferenceCandidate(ctx, store, path,
		modelintake.SessionOverride{Set: true, Value: modelrecipe.DecodeSessionRequest}, recipe.ResidencyHybridNative)
	if err != nil || prior.Definition.ID == want.Definition.ID {
		t.Fatalf("distinct active fixture: %v", err)
	}
	if err := modelrecipetest.PublishActivation(ctx, store, "fixture/prior-active", prior.Definition); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	check()
	check()
	active, found, err := modelrecipe.ActiveRecord(ctx, store, prior.Inventory.Manifest.ID, recipe.TaskInference)
	if err != nil || !found || active.Definition.ID != prior.Definition.ID {
		t.Fatalf("verification replaced the active recipe: %v", err)
	}
	canceled, cancel := context.WithCancelCause(ctx)
	cancel(context.Canceled)
	if _, err := prepareExactCandidate(canceled, store, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
	if got, ordinal := store.Head(); got != head || ordinal != sequence {
		t.Fatal("repeat or canceled preparation mutated the store")
	}
}

func TestPrepareCapabilityReusesPublishedFacts(t *testing.T) {
	path := testutil.TempGGUF(t, "facts.gguf", nil, []gguf.TensorData{{
		Name: "weight", Shape: []uint64{1}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(make([]byte, binary.Size(float32(0)))),
	}})
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	if err != nil {
		t.Fatal(err)
	}
	capability := mediacapability.Capability{Resolve: func(string) (mediacapability.Source, error) {
		return mediacapability.Source{Inventory: inventory}, nil
	}}
	want, err := modelrecipe.CapabilityDefinition(recipe.TaskSeq2Seq, inventory.Manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	store, err := overgodb.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	batch, err := inventory.Batch("fixture/another-publisher")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	unchanged := func() {
		t.Helper()
		if gotHead, gotSequence := store.Head(); gotHead != head || gotSequence != sequence {
			t.Fatalf("publication changed head/sequence: %s/%d, want %s/%d", gotHead, gotSequence, head, sequence)
		}
	}
	for _, phase := range []string{"foreign publisher", "repeat", "reopen"} {
		t.Run(phase, func(t *testing.T) {
			if phase == "reopen" {
				if err := store.Close(); err != nil {
					t.Fatal(err)
				}
				store, err = overgodb.Open(directory)
				if err != nil {
					t.Fatal(err)
				}
			}
			model, definition, err := prepareCapability(t.Context(), store, path, recipe.TaskSeq2Seq, capability)
			if err != nil || model != inventory.Manifest.ID || definition.ID != want.ID {
				t.Fatalf("prepare = %s/%s, %v; want %s/%s", model, definition.ID, err, inventory.Manifest.ID, want.ID)
			}
			unchanged()
		})
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, _, err := prepareCapability(ctx, store, path, recipe.TaskSeq2Seq, capability); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled preparation = %v", err)
	}
	unchanged()
	inventory.Components[0].MediaType = "application/octet-stream"
	if _, _, err := prepareCapability(t.Context(), store, path, recipe.TaskSeq2Seq, capability); !errors.Is(err, overgodb.ErrArtifactConflict) {
		t.Fatalf("conflicting facts = %v", err)
	}
	unchanged()
}

func TestCommandsRegisterExecutableCapabilities(t *testing.T) {
	for _, task := range []recipe.Task{
		recipe.TaskGeneration, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq,
		recipe.TaskSpeech, recipe.TaskImageGen, recipe.TaskVideoGen,
	} {
		t.Run(string(task), func(t *testing.T) {
			capability, ok := mediacapability.Catalog[task]
			if !ok || capability.Resolve == nil || capability.Execute == nil {
				t.Fatalf("capability = %+v", capability)
			}
		})
	}
	projection := mediacapability.Projection(t.Context(), nil, "")
	if projection.Resolve == nil || projection.Execute != nil {
		t.Fatalf("projection capability = %+v", projection)
	}
}

func TestCapabilityVerificationActivatesCandidate(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "recipe-verifier-model")
	testutil.PublishArtifact(t, store, modelID)
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSeq2Seq, modelID)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/recipe-verifier/candidate", definition); err != nil {
		t.Fatal(err)
	}
	verification, err := modelintake.PublishVerification(
		ctx, store, definition, "0123456789abcdef0123456789abcdef01234567", time.Millisecond,
		"host", "go", "fixture output validated",
	)
	if err != nil {
		t.Fatal(err)
	}
	verified, err := runrecord.VerifyGateRun(ctx, store, definition.ID, verification.Gate, verification.Run)
	if err != nil {
		t.Fatal(err)
	}
	if verified.Gate.Recipe != definition.ID || verified.Run.Recipe != definition.ID {
		t.Fatalf("verification subject = %s/%s, want %s", verified.Gate.Recipe, verified.Run.Recipe, definition.ID)
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, definition, verification, recipe.EvidenceVerified, "fixture candidate execution",
	); err != nil {
		t.Fatal(err)
	}
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, modelID, recipe.TaskSeq2Seq)
	if err != nil {
		t.Fatal(err)
	}
	if !active || activation.Definition.ID != definition.ID {
		t.Fatalf("active = %t/%s, want true/%s", active, activation.Definition.ID, definition.ID)
	}
}

// TestMeasurementNeedsExecutableInput: measurement runs the request its
// activation's accepted verification retained whole, once. A verification
// that kept only the request's digest, or content that is no request, is a
// gap reported before the executor runs or any serving attempt is recorded,
// and a request another run of the recipe retained never stands in for it.
func TestMeasurementNeedsExecutableInput(t *testing.T) {
	request := func(schema, body string) artifact.Content {
		t.Helper()
		content, err := artifact.JSONContract(artifact.KindFile, schema).ContentBytes([]byte(body))
		if err != nil {
			t.Fatal(err)
		}
		return content
	}
	whole := request("overgo.fixture-input.v1", `{"prompt":"verified"}`)
	unrelated := request("overgo.fixture-input.v1", `{"prompt":"another run's"}`)
	path := testutil.TempGGUF(t, "measured.gguf", nil, []gguf.TensorData{{
		Name: "weight", Shape: []uint64{1}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(make([]byte, binary.Size(float32(0)))),
	}})
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	modelID := inventory.Manifest.ID
	for name, test := range map[string]struct {
		accepted artifact.Content
		want     string
	}{
		"whole request":  {accepted: whole, want: string(whole.Data)},
		"digest only":    {accepted: request("overgo.fixture-input-digest.v1", `{"request_sha256":"x","request_bytes":2000000}`)},
		"not a request":  {accepted: request("overgo.fixture-output.v1", `{"prompt":"output"}`)},
		"nothing at all": {},
	} {
		t.Run(name, func(t *testing.T) {
			ctx, directory := t.Context(), t.TempDir()
			store, err := overgodb.Open(directory)
			if err != nil {
				t.Fatal(err)
			}
			batch, err := inventory.Batch("fixture/measured/model")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Commit(ctx, batch); err != nil {
				t.Fatal(err)
			}
			definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSeq2Seq, modelID)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/measured/candidate", definition); err != nil {
				t.Fatal(err)
			}
			verify := func(input artifact.Content) modelrecipe.Verification {
				t.Helper()
				verification, err := modelintake.PublishMeasuredVerification(ctx, store, definition,
					"0123456789abcdef0123456789abcdef01234567", time.Millisecond, "host", "go", "fixture output validated",
					capabilityruntime.Measured{Input: input})
				if err != nil {
					t.Fatal(err)
				}
				return verification
			}
			// Another run of the recipe retained a whole request; it is not
			// the accepted verification's.
			verify(unrelated)
			if err := modelrecipe.ActivateCapability(ctx, store, definition, verify(test.accepted),
				recipe.EvidenceVerified, "fixture candidate execution"); err != nil {
				t.Fatal(err)
			}
			head, sequence := store.Head()
			if err := store.Close(); err != nil {
				t.Fatal(err)
			}
			var ran []string
			capability := mediacapability.Capability{
				Resolve: func(string) (mediacapability.Source, error) {
					return mediacapability.Source{Inventory: inventory}, nil
				},
				Execute: func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, input string) (any, error) {
					ran = append(ran, input)
					return "measured", nil
				},
			}
			err = measureCapability(directory, "fixture-path", recipe.TaskSeq2Seq, capability)
			reopened, openErr := overgodb.Open(directory)
			if openErr != nil {
				t.Fatal(openErr)
			}
			defer reopened.Close()
			after, afterSequence := reopened.Head()
			if test.want != "" {
				if err != nil || !slices.Equal(ran, []string{test.want}) || afterSequence == sequence {
					t.Fatalf("measurement ran %q (%v), recorded attempt %v; want the verified request once, recorded", ran, err, afterSequence != sequence)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), "retained no runnable request") || len(ran) != 0 || after != head || afterSequence != sequence {
				t.Fatalf("measurement ran %q (%v), store moved %v; want a gap before execution", ran, err, after != head)
			}
		})
	}
}
