package main

import (
	"os"
	"testing"

	"overgo/internal/discovery"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestACompositeIsPublishedUnderTheModelIdentityItsBytesHold holds a composite to its text model's bytes (SQA-331):
// textModel took the catalog memo's claim, which a remembered digest answers while the file's size and modification
// time match, so a GGUF whose payload changed with both kept was published under its old identity. With the model's
// digest remembered and its payload changed, size and time kept, textModel names the identity of the bytes now.
func TestACompositeIsPublishedUnderTheModelIdentityItsBytesHold(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	text := testutil.HermeticLlamaGGUF(t, 64)
	before, _, _, err := textModel(ctx, store, text)
	testcheck.NoError(t, err)
	memo := discovery.NewMemo()
	testcheck.Must(discovery.WeightsIdentity(text, memo))(t)
	testcheck.NoError(t, discovery.PublishMemo(ctx, store, memo))

	info := testcheck.Must(os.Stat(text))(t)
	data := testcheck.Must(os.ReadFile(text))(t)
	data[len(data)-1] ^= 0xff
	testcheck.NoError(t, os.WriteFile(text, data, info.Mode()))
	testcheck.NoError(t, os.Chtimes(text, info.ModTime(), info.ModTime()))
	if _, _, remembered := discovery.LoadMemo(ctx, store).Claim(text); !remembered {
		t.Fatal("the fixture's memo does not claim the changed file, so it shows nothing")
	}
	after, _, _, err := textModel(ctx, store, text)
	testcheck.NoError(t, err)
	if after.Model == before.Model {
		t.Fatalf("a model whose payload changed, size and time kept, is published under its old identity %s", before.Model)
	}
}
