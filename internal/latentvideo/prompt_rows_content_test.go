package latentvideo

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

// syntheticEncoder is a umT5 encoder checkpoint as bytes: one block, four
// values wide, over the nine pieces of the synthetic tokenizer, every tensor
// the encoder's plan asks for in a storage of its own. The seed sets the
// weights and nothing else, so two seeds give archives of one length.
func syntheticEncoder(t *testing.T, seed int) []byte {
	t.Helper()
	const vocabulary, dim, feed, buckets, heads = 9, 4, 8, 8, 2
	tensors := []struct {
		name  string
		shape []int64
	}{
		{"token_embedding.weight", []int64{vocabulary, dim}},
		{"norm.weight", []int64{dim}},
		{"blocks.0.norm1.weight", []int64{dim}},
		{"blocks.0.attn.q.weight", []int64{dim, dim}},
		{"blocks.0.attn.k.weight", []int64{dim, dim}},
		{"blocks.0.attn.v.weight", []int64{dim, dim}},
		{"blocks.0.attn.o.weight", []int64{dim, dim}},
		{"blocks.0.norm2.weight", []int64{dim}},
		{"blocks.0.ffn.gate.0.weight", []int64{feed, dim}},
		{"blocks.0.ffn.fc1.weight", []int64{feed, dim}},
		{"blocks.0.ffn.fc2.weight", []int64{dim, feed}},
		{"blocks.0.pos_embedding.embedding.weight", []int64{buckets, heads}},
	}
	// A protocol-2 torch state_dict, written as torch writes one: each tensor rebuilt from a storage named by its place.
	var pickle bytes.Buffer
	text := func(value string) {
		pickle.WriteByte('X')
		binary.Write(&pickle, binary.LittleEndian, uint32(len(value)))
		pickle.WriteString(value)
	}
	number := func(value int64) {
		pickle.WriteByte('J')
		binary.Write(&pickle, binary.LittleEndian, int32(value))
	}
	pickle.WriteString("\x80\x02}q\x00(")
	storages := make([][]byte, len(tensors))
	for index, tensor := range tensors {
		elements := int64(1)
		for _, extent := range tensor.shape {
			elements *= extent
		}
		text(tensor.name)
		pickle.WriteString("ctorch._utils\n_rebuild_tensor_v2\n((")
		text("storage")
		pickle.WriteString("ctorch\nBFloat16Storage\n")
		text(strconv.Itoa(index))
		text("cpu")
		number(elements)
		pickle.WriteString("tQ")
		number(0)
		pickle.WriteByte('(')
		for _, extent := range tensor.shape {
			number(extent)
		}
		pickle.WriteString("t(")
		stride := elements
		for _, extent := range tensor.shape {
			stride /= extent
			number(stride)
		}
		pickle.WriteString("t\x89)tR")
		storages[index] = make([]byte, elements*2)
		for at := range int(elements) {
			value := float32((at*7+index*5+seed*3)%13-6) / 8
			binary.LittleEndian.PutUint16(storages[index][at*2:], dtype.Float32ToBF16(value))
		}
	}
	pickle.WriteString("u.")
	var archive bytes.Buffer
	written := zip.NewWriter(&archive)
	entry := func(name string, body []byte) {
		writer, err := written.CreateHeader(&zip.FileHeader{Name: name, Method: zip.Store})
		if err == nil {
			_, err = writer.Write(body)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	entry("archive/data.pkl", pickle.Bytes())
	for index, storage := range storages {
		entry("archive/data/"+strconv.Itoa(index), storage)
	}
	if err := written.Close(); err != nil {
		t.Fatal(err)
	}
	return archive.Bytes()
}

// TestPromptRowsFollowTheBytesTheyAreMadeFrom holds every keeper of a
// prompt's rows to the bytes the rows are made from, on a private text
// pipeline laid out as a Wan model's: a tokenizer's table and special tokens,
// an encoder checkpoint, and a projection.
//
// Other bytes are put in one file at a time under the place, the size and the
// time it had: the checkpoint written again with other weights, the
// tokenizer's table with two scores exchanged, and the special tokens alone,
// with another end token. After each, the pipeline has another name, the
// store holds no rows under the name the files now give, and four askers each
// get what a process that has just started, with no store, makes of the files
// as they stand: the resident session whose memo holds the same prompts, a
// new session of the same process, and the next process, which reads the
// common prompt's rows from the store. The resident session encodes once for
// it. With nothing changed nothing is encoded again: the memo answers with
// the contexts it holds, and the next process reads the common prompt's rows
// from the store and encodes the request's own prompt alone.
//
// Special tokens that name no piece refuse the resident session as they
// refuse a process that has just started. A pipeline with a file missing has
// a name where a model is served without one and is refused where rows are to
// be made, and a checkpoint that is no archive has no name.
func TestPromptRowsFollowTheBytesTheyAreMadeFrom(t *testing.T) {
	// Not parallel: the encoders this process keeps are this test's to forget.
	ctx := t.Context()
	directory := t.TempDir()
	const length = 6
	spec := WanTextConditioningSpec(directory, length)
	table, roles := filepath.Join(spec.TokenizerDir, tokenizerTable), filepath.Join(spec.TokenizerDir, tokenizerRoles)
	if err := os.MkdirAll(spec.TokenizerDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeSyntheticTokenizer(t, spec.TokenizerDir)
	if err := os.WriteFile(spec.EncoderCheckpoint, syntheticEncoder(t, 1), 0o644); err != nil {
		t.Fatal(err)
	}
	square, row := arithmeticFloat32Sequence(-0.5, 0.0625, 16), arithmeticFloat32Sequence(0.25, -0.125, 4)
	if err := safetensors.Save(filepath.Join(directory, "diffusion_pytorch_model.safetensors"),
		map[string][]float32{projectionTensorNames[0]: square, projectionTensorNames[1]: row, projectionTensorNames[2]: square, projectionTensorNames[3]: row},
		map[string][]int{projectionTensorNames[0]: {4, 4}, projectionTensorNames[1]: {4}, projectionTensorNames[2]: {4, 4}, projectionTensorNames[3]: {4}}, nil); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	spec.Common, spec.Kept, spec.KeptUnder = "world", store, "identity"
	request := WanRequest{Prompt: "hello world", NegativePrompt: spec.Common}

	// forget drops every row and tokenizer this process's encoders keep, as a process that has just started holds none.
	forget := func() {
		promptEncoders.Lock()
		clear(promptEncoders.byKey)
		promptEncoders.Unlock()
	}
	forget()
	t.Cleanup(forget)
	passes := func() (made int) {
		promptEncoders.Lock()
		defer promptEncoders.Unlock()
		for _, encoder := range promptEncoders.byKey {
			encoder.mu.Lock()
			made += encoder.passes
			encoder.mu.Unlock()
		}
		return made
	}
	named := func() string {
		t.Helper()
		name, err := TextPipelineIdentity(directory)
		if err != nil {
			t.Fatal(err)
		}
		return name
	}
	// kept reports whether the store holds the common prompt's rows under the name the files as they stand give.
	kept := func() bool {
		t.Helper()
		encoder, err := promptEncoderFor(spec)
		if err != nil {
			t.Fatal(err)
		}
		_, bound, err := artifact.ResolveAlias(ctx, store, encoder.commonRowsName(spec))
		if err != nil {
			t.Fatal(err)
		}
		return bound
	}
	// cold is what a process that has just started, with no store to read rows from, makes of the files as they stand.
	cold := func() (cond, uncond []float32, err error) {
		forget()
		alone := spec
		alone.Kept, alone.KeptUnder = nil, ""
		return new(wanConditioning).contexts(ctx, alone, request)
	}
	// replaced puts other bytes in a file under the place, the size and the time it had.
	replaced := func(path string, other func(was []byte) []byte) {
		t.Helper()
		before, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		was, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		now := other(was)
		if len(now) != len(was) || bytes.Equal(now, was) {
			t.Fatalf("%s: %d other bytes for %d, the same bytes: %t; want other bytes of the same length", filepath.Base(path), len(now), len(was), bytes.Equal(now, was))
		}
		if err := os.WriteFile(path, now, 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, before.ModTime(), before.ModTime()); err != nil {
			t.Fatal(err)
		}
		if after, err := os.Stat(path); err != nil || after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) {
			t.Fatalf("%s: its size or time moved with its bytes: %v", filepath.Base(path), err)
		}
	}
	// exchanged gives the bytes with each of the pairs' first replaced by its second, each found once.
	exchanged := func(pairs ...string) func([]byte) []byte {
		return func(was []byte) []byte {
			t.Helper()
			for at := 0; at < len(pairs); at += 2 {
				if bytes.Count(was, []byte(pairs[at])) != 1 {
					t.Fatalf("the fixture holds %q %d times, want once", pairs[at], bytes.Count(was, []byte(pairs[at])))
				}
				was = bytes.Replace(was, []byte(pairs[at]), []byte(pairs[at+1]), 1)
			}
			return was
		}
	}

	// The resident session, and nothing changed.
	var session wanConditioning
	cond, uncond, err := session.contexts(ctx, spec, request)
	if err != nil {
		t.Fatal(err)
	}
	if passes() != 1 || !kept() {
		t.Fatalf("the first request made %d passes and kept the common prompt's rows in the store: %t; want one pass, and kept", passes(), kept())
	}
	if again, _, err := session.contexts(ctx, spec, request); err != nil || passes() != 1 || &again[0] != &cond[0] {
		t.Fatalf("the same request again (%v) made %d passes and has the contexts the memo holds: %t; want no second pass, and the memo's", err, passes(), err == nil && &again[0] == &cond[0])
	}
	forget()
	if _, _, stats, err := encodePrompts(ctx, spec, request.Prompt, spec.Common); err != nil || stats.TokenRows != 3 {
		t.Fatalf("the next process, nothing changed, encoded %d rows (%v); want the 3 of the request's own prompt, the common prompt's read from the store", stats.TokenRows, err)
	}

	for _, change := range []struct {
		what, path string
		other      func(was []byte) []byte
	}{
		{"the checkpoint written again with other weights", spec.EncoderCheckpoint, func([]byte) []byte { return syntheticEncoder(t, 2) }},
		{"the tokenizer's table with two scores exchanged", table, exchanged(`["▁hello",1]`, `["▁hello",-1]`, `["▁",-1]`, `["▁",1]`)},
		{"the special tokens alone, with another end token", roles, exchanged(`"eos_token":"</s>",`, `"eos_token":"<s>", `)},
	} {
		was := named()
		replaced(change.path, change.other)
		if named() == was {
			t.Fatalf("%s: the pipeline has the name it had", change.what)
		}
		if kept() {
			t.Fatalf("%s: the store holds rows under the name of bytes nothing has encoded", change.what)
		}
		made := passes()
		warm, warmUncond, err := session.contexts(ctx, spec, request)
		if err != nil {
			t.Fatal(err)
		}
		if passes() != made+1 || !kept() {
			t.Fatalf("%s: the resident session made %d passes and kept the common prompt's rows under the new name: %t; want one pass, and kept", change.what, passes()-made, kept())
		}
		renewed, renewedUncond, err := new(wanConditioning).contexts(ctx, spec, request)
		if err != nil {
			t.Fatal(err)
		}
		forget()
		next, nextUncond, err := new(wanConditioning).contexts(ctx, spec, request)
		if err != nil {
			t.Fatal(err)
		}
		want, wantUncond, err := cold()
		if err != nil {
			t.Fatal(err)
		}
		if slices.Equal(want, cond) && slices.Equal(wantUncond, uncond) {
			t.Fatalf("%s: a process that has just started makes the contexts made before the change, so the change shows nothing", change.what)
		}
		for asker, got := range map[string][2][]float32{
			"the resident session": {warm, warmUncond}, "a new session": {renewed, renewedUncond}, "the next process": {next, nextUncond},
		} {
			if !slices.Equal(got[0], want) || !slices.Equal(got[1], wantUncond) {
				t.Errorf("%s: %s has other contexts than a process that has just started makes of the files as they stand", change.what, asker)
			}
		}
		cond, uncond = want, wantUncond
	}

	// Special tokens that name no piece.
	replaced(roles, exchanged(`"eos_token":"<s>", `, `"eos_token":"<x>", `))
	_, _, warmErr := session.contexts(ctx, spec, request)
	if _, _, coldErr := cold(); warmErr == nil || coldErr == nil || warmErr.Error() != coldErr.Error() {
		t.Fatalf("special tokens that name no piece: the resident session = %v and a process that has just started = %v; want the one refusal from both", warmErr, coldErr)
	}

	// A pipeline with a file missing, and a checkpoint that is no archive.
	absent := WanTextConditioningSpec(t.TempDir(), length)
	if name, err := TextPipelineIdentity(filepath.Dir(absent.EncoderCheckpoint)); err != nil || name == "" || name == named() {
		t.Fatalf("a model with no text pipeline is named %q (%v); want a name of its own, for a request that brings its contexts", name, err)
	}
	if _, err := absent.content(false); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("rows asked of a pipeline that is not there = %v, want it refused as not there", err)
	}
	if err := os.WriteFile(spec.EncoderCheckpoint, []byte("no archive"), 0o644); err != nil {
		t.Fatal(err)
	}
	if name, err := TextPipelineIdentity(directory); err == nil {
		t.Fatalf("a checkpoint that is no archive is named %q; want no name, and so no answer kept under one", name)
	}
}
