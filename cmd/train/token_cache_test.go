package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/safetensors"
)

// writeTinyEncoder writes directory as an encoder checkpoint of three files:
// one weight, a configuration and an image processor. Nothing loads it; the
// name tokens are kept under reads its bytes.
func writeTinyEncoder(t *testing.T, directory string, weight float32, config, processor string) string {
	t.Helper()
	if err := os.MkdirAll(directory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(directory, "model.safetensors"), map[string][]float32{"weight": {weight}}, map[string][]int{"weight": {1}}, nil); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{"config.json": config, "preprocessor_config.json": processor} {
		if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return directory
}

// TestTokensAreKeptUnderTheBytesOfTheEncodersThatMadeThem holds the directory
// a run keeps tokens in to the encoders that made them. Tokens are an
// encoder's answer for a picture, so two encoders of one name under two roots,
// whose weights differ in value alone, must not share a directory, or the
// second run would train on the first one's tokens; the same encoder under
// another root must find what it kept; and a change to any file the encoder
// loads by, its weights, its configuration or its image processor, must leave
// the old tokens behind. A second encoder's change leaves the first one's
// directory alone.
func TestTokensAreKeptUnderTheBytesOfTheEncodersThatMadeThem(t *testing.T) {
	t.Parallel()
	root, cache := t.TempDir(), filepath.Join(t.TempDir(), "cache")
	producer := "known-producing-implementation"
	const config, processor = `{"hidden_size":3}`, `{"size":8}`
	first := writeTinyEncoder(t, filepath.Join(root, "a", "enc"), 1, config, processor)
	kept := func(encoders ...string) string {
		t.Helper()
		directory, err := tokenCache(cache, producer, encoders...)
		if err != nil {
			t.Fatalf("no token directory for %v: %v", encoders, err)
		}
		inside, err := filepath.Rel(cache, directory)
		if info, statErr := os.Stat(directory); err != nil || statErr != nil || !info.IsDir() || inside == "." || !filepath.IsLocal(inside) {
			t.Fatalf("the token directory %s for %v was not made under the cache %s (%v, %v)", directory, encoders, cache, err, statErr)
		}
		return directory
	}
	own := kept(first)

	// The finding's case: the same name under another root, another value in
	// a weight of the same shape.
	if other := kept(writeTinyEncoder(t, filepath.Join(root, "b", "enc"), 2, config, processor)); other == own {
		t.Errorf("two encoders named enc whose weights differ keep their tokens in the one directory %s", own)
	}

	// The same bytes under another root are the same encoder: what one run
	// kept the next reads back exactly, and a file of another count is not it.
	same := kept(writeTinyEncoder(t, filepath.Join(root, "c", "other-name"), 1, config, processor))
	if same != own {
		t.Errorf("the same encoder under another root keeps its tokens in %s, not in %s where it kept them before", same, own)
	}
	tokens := []float32{0.25, -1.5, 3}
	if err := writeTokens(own, "picture.source", tokens); err != nil {
		t.Fatal(err)
	}
	if read, found := readTokens(same, "picture.source", len(tokens)); !found || !slices.Equal(read, tokens) {
		t.Errorf("tokens kept as %v were read back as %v (found %t)", tokens, read, found)
	}
	if _, found := readTokens(same, "picture.source", len(tokens)+1); found {
		t.Error("tokens of another count were read back")
	}

	for _, changed := range []string{"changed-producing-implementation", ""} {
		directory, err := tokenCache(cache, changed, first)
		if err != nil {
			t.Fatal(err)
		}
		if changed == "" {
			if err := writeTokens(directory, filepath.Join(own, "picture.source"), []float32{9}); err != nil {
				t.Fatal(err)
			}
			if _, found := readTokens(directory, filepath.Join(own, "picture.source"), len(tokens)); found {
				t.Error("unknown producer read persisted tokens")
			}
			if read, found := readTokens(own, "picture.source", len(tokens)); !found || !slices.Equal(read, tokens) {
				t.Error("unknown producer overwrote persisted tokens")
			}
			if directory != "" {
				t.Errorf("unknown producer admitted persistent tokens under %q", directory)
			}
		} else if directory == own {
			t.Error("changed producing implementation reused the old tokens")
		}
	}

	// Each file the encoder loads by names the directory.
	seen := map[string]string{own: "the first encoder"}
	for name, encoder := range map[string]string{
		"another configuration":   writeTinyEncoder(t, filepath.Join(root, "d", "enc"), 1, `{"hidden_size":4}`, processor),
		"another image processor": writeTinyEncoder(t, filepath.Join(root, "e", "enc"), 1, config, `{"size":9}`),
	} {
		directory := kept(encoder)
		if before, shared := seen[directory]; shared {
			t.Errorf("an encoder with %s keeps its tokens in %s, as %s does", name, directory, before)
		}
		seen[directory] = "the encoder with " + name
	}
	// An encoder with no image processor is another encoder still; the loader
	// is what refuses it.
	bare := writeTinyEncoder(t, filepath.Join(root, "f", "enc"), 1, config, processor)
	if err := os.Remove(filepath.Join(bare, "preprocessor_config.json")); err != nil {
		t.Fatal(err)
	}
	if directory := kept(bare); seen[directory] != "" {
		t.Errorf("an encoder with no image processor keeps its tokens in %s, as %s does", directory, seen[directory])
	}

	// Tokens of two encoders are kept under both, and a change to the second
	// leaves the first one's own directory where it was.
	second := writeTinyEncoder(t, filepath.Join(root, "g", "target"), 5, config, processor)
	both := kept(first, second)
	changed := kept(first, writeTinyEncoder(t, filepath.Join(root, "h", "target"), 6, config, processor))
	if both == own || both == changed || kept(first) != own {
		t.Errorf("tokens of two encoders are kept in %s, after the second changed in %s, and the first alone in %s then %s", both, changed, own, kept(first))
	}

	// An encoder with no configuration or no weights is refused, where a name
	// alone would have given it a directory.
	for name, file := range map[string]string{"configuration": "config.json", "weights": "model.safetensors"} {
		broken := writeTinyEncoder(t, filepath.Join(root, name, "enc"), 1, config, processor)
		if err := os.Remove(filepath.Join(broken, file)); err != nil {
			t.Fatal(err)
		}
		if directory, err := tokenCache(cache, producer, broken); err == nil {
			t.Errorf("an encoder with no %s was given the token directory %s", name, directory)
		}
	}
}

// TestAClipsTokensAreKeptUnderItsCut holds the name a clip's tokens are kept
// under to the picture they were made from. The tokens answer for the frame as
// it was cut to the pair's size, so two cuts of one clip must not share
// tokens, and the same picture cut from two files is one picture: the name
// reads the picture and nothing of the file it came from.
func TestAClipsTokensAreKeptUnderItsCut(t *testing.T) {
	t.Parallel()
	name := func(picture []float32) string {
		t.Helper()
		name := pictureDigest(picture)
		if name == "" || name != filepath.Base(name) {
			t.Fatalf("a picture's tokens are named %q, want one file name", name)
		}
		return name
	}
	small, large := []float32{0, 0.5, 1}, []float32{0, 0.5, 1, 0.25}
	if one, other := name(small), name(large); one == other {
		t.Errorf("two cuts of one clip keep their tokens under the one name %s", one)
	}
	if one, other := name(small), name(slices.Clone(small)); one != other {
		t.Errorf("one picture cut from two files keeps its tokens under %s and under %s", one, other)
	}
	if one, other := name(small), name([]float32{0, 0.5, 0.75}); one == other {
		t.Errorf("two pictures that differ in one value keep their tokens under the one name %s", one)
	}
}
