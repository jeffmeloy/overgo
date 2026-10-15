package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

type restoreFaults struct {
	artifact.Repository
	open   func(artifact.ID) error
	commit func(artifact.Batch) error
}

func (f restoreFaults) OpenContent(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	if f.open != nil {
		if err := f.open(id); err != nil {
			return artifact.Descriptor{}, nil, false, err
		}
	}
	return f.Repository.OpenContent(ctx, id)
}
func (f restoreFaults) Commit(ctx context.Context, batch artifact.Batch) (artifact.CommitID, error) {
	if f.commit != nil {
		if err := f.commit(batch); err != nil {
			return artifact.CommitID{}, err
		}
	}
	return f.Repository.Commit(ctx, batch)
}

func TestAgentMutationCheckpointLifecycle(t *testing.T) {
	ctx := t.Context()
	root := t.TempDir()
	target := filepath.Join(root, "a-state.txt")
	second := filepath.Join(root, "b-state.txt")
	empty := filepath.Join(root, "empty.txt")
	absent := filepath.Join(root, "new.txt")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	manual, err := agenttool.NewManual(agenttool.Manual{Name: "file.write", Description: "Write workspace files.", Effect: agenttool.EffectMutation,
		Arguments: []agenttool.Field{{Name: "path", Kind: agenttool.FieldString, Required: true}},
		Ceiling:   agenttool.EffectCeiling{Targets: []agenttool.EffectTargetBinding{{Scope: agenttool.EffectScopeWorkspace, Argument: "path"}, {Scope: agenttool.EffectScopeWorkspace, Value: second}, {Scope: agenttool.EffectScopeWorkspace, Value: empty}, {Scope: agenttool.EffectScopeWorkspace, Value: absent}}}, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin}})
	if err != nil {
		t.Fatal(err)
	}
	arguments, _ := json.Marshal(map[string]string{"path": target})
	effect, err := agenttool.DeriveInvocationEffect(manual, arguments, nil)
	if err != nil {
		t.Fatal(err)
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "recipe")
	for _, failure := range []string{"later preimage", "completion receipt"} {
		t.Run(failure, func(t *testing.T) {
			before := map[string]string{target: "before", second: "second-before", empty: ""}
			modes := map[string]os.FileMode{}
			for path, data := range before {
				if err := os.WriteFile(path, []byte(data), 0o640); err != nil {
					t.Fatal(err)
				}
				info, err := os.Stat(path)
				if err != nil {
					t.Fatal(err)
				}
				modes[path] = info.Mode().Perm()
			}
			if err := os.Remove(absent); err != nil && !errors.Is(err, os.ErrNotExist) {
				t.Fatal(err)
			}
			runtime, err := NewMutationCheckpointRuntime(root, store)
			if err != nil {
				t.Fatal(err)
			}
			checkpoint, err := runtime.BeginCheckpoint(ctx, testutil.ArtifactID(t, artifact.KindEvidence, failure), effect, 1)
			if err != nil {
				t.Fatal(err)
			}
			for _, path := range []string{target, second, empty, absent} {
				if err := os.WriteFile(path, []byte("after"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			checkpoint, err = runtime.SealCheckpoint(ctx, checkpoint)
			if err != nil {
				t.Fatal(err)
			}
			request, err := RestoreApprovalRequest(checkpoint, recipeID)
			if err != nil {
				t.Fatal(err)
			}
			decision, err := runrecord.NewHumanDecision(request, operatoraction.AnswerGrant)
			if err != nil {
				t.Fatal(err)
			}
			if err := runrecord.PublishHumanDecision(ctx, store, request, decision); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(target, []byte("drift"), 0o640); err != nil {
				t.Fatal(err)
			}
			if _, err := runtime.RestoreCheckpoint(ctx, checkpoint, effect, recipeID); err == nil {
				t.Fatal("postimage drift was accepted")
			}
			if err := os.WriteFile(target, []byte("after"), 0o640); err != nil {
				t.Fatal(err)
			}
			broken := errors.New(failure)
			faults := restoreFaults{Repository: store}
			if failure == "later preimage" {
				faults.open = func(id artifact.ID) error {
					data, _ := os.ReadFile(target)
					if id == testutil.ArtifactID(t, artifact.KindFile, "second-before") && string(data) == "before" {
						return broken
					}
					return nil
				}
			} else {
				faults.commit = func(batch artifact.Batch) error {
					for _, content := range batch.Contents {
						if content.Descriptor.Schema == runrecord.StageReceiptSchema {
							receipt, err := runrecord.ParseStageReceipt(content.Data)
							if err != nil {
								return err
							}
							if receipt.State == runrecord.StageCompleted {
								return broken
							}
						}
					}
					return nil
				}
			}
			runtime.store = faults
			if _, err := runtime.RestoreCheckpoint(ctx, checkpoint, effect, recipeID); !errors.Is(err, broken) {
				t.Fatalf("restore failure=%v, want %v", err, broken)
			}
			if data, err := os.ReadFile(target); err != nil || string(data) != "before" {
				t.Fatalf("first restored file=%q, %v", data, err)
			}
			runtime, err = NewMutationCheckpointRuntime(root, store)
			if err != nil {
				t.Fatal(err)
			}
			receipt, err := runtime.RestoreCheckpoint(ctx, checkpoint, effect, recipeID)
			if err != nil {
				t.Fatalf("exact interrupted restore retry: %v", err)
			}
			for path, want := range before {
				data, err := os.ReadFile(path)
				if err != nil || string(data) != want {
					t.Fatalf("restored %s=%q, %v, want %q", path, data, err, want)
				}
				info, err := os.Stat(path)
				if err != nil || info.Mode().Perm() != modes[path] {
					t.Fatalf("restored mode for %s=%v, %v", path, info, err)
				}
			}
			if _, err := os.Stat(absent); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("new file remained: %v", err)
			}
			head, _ := store.Head()
			repeated, err := runtime.RestoreCheckpoint(ctx, checkpoint, effect, recipeID)
			after, _ := store.Head()
			if err != nil || !receipt.Valid() || repeated != receipt || after != head {
				t.Fatalf("completion replay=%s, %v, head %s, want %s and %s", repeated, err, after, receipt, head)
			}
			if err := os.WriteFile(target, []byte("after"), 0o640); err != nil {
				t.Fatal(err)
			}
			if _, err := runtime.RestoreCheckpoint(ctx, checkpoint, effect, recipeID); err == nil {
				t.Fatal("completed restore admitted subsequent mutation")
			}
		})
	}
}
