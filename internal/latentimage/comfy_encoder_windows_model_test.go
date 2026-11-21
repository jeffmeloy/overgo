//go:build modeltest && windows

package latentimage

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/graphruntime"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// ComfyUI conditions with the Ultra encoder in bfloat16: each weight decoded to bfloat16 before its matmul (its text
// encoders run full_precision_mm), and every operation's output rounded to bfloat16, a step of 2^-8 of the value, over
// 32 layers. overgo decodes the same bfloat16 weights and carries its activations in F32, so the two differ by
// ComfyUI's rounding, carried through the layers. Declared before the first run: each token's state lies within
// cosine 0.99 of ComfyUI's, and each prompt's states within 5% of ComfyUI's in relative L2. A weight decoded without
// its rotation, a layer read under the wrong name or states taken before the final norm lie far outside both.
const (
	ultraEncoderTokenCosine = 0.99
	ultraEncoderRelativeL2  = 0.05
)

// TestTheUltraEncoderConditionsAsComfyUIDoes holds the MOSS Ultra workflow's text encoder (moss-ultra-import/encoder)
// to ComfyUI's, recorded by scripts/python_golden_references/comfy_moss_encoder_golden.py into
// fixtures/comfy/moss-ultra/encoder.json: the abliterated Qwen3.5-9B in int8 ConvRot, read as the Hugging Face language
// model it holds (comfyLanguageView) beside MOSS-Image-SFT's encoder config and tokenizer, renders each golden prompt
// under MOSS's template to ComfyUI's token IDs exactly, and encodes them a block at a time, as a request does, to
// ComfyUI's states within the declared tolerance. The fixture names the file the import recipe pins.
func TestTheUltraEncoderConditionsAsComfyUIDoes(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	repo := testutil.RepoRoot(t)
	var golden struct {
		Encoder string `json:"encoder"`
		SHA256  string `json:"encoder_sha256"`
		Commit  string `json:"comfyui_commit"`
		Type    string `json:"clip_type"`
		Cases   []struct {
			Prompt    string `json:"prompt"`
			Tokenizer string `json:"tokenizer"`
			IDs       []int  `json:"ids"`
			Shape     []int  `json:"shape"`
			States    string `json:"states_sha256"`
		} `json:"cases"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repo, "fixtures", "comfy", "moss-ultra", "encoder.json")))(t), &golden))
	if len(golden.Cases) == 0 || golden.Type != "moss_image" {
		t.Fatalf("the golden records %d cases of %q; want moss_image's", len(golden.Cases), golden.Type)
	}
	type pinned struct {
		Path   string `json:"path"`
		SHA256 string `json:"sha256"`
	}
	var recipe struct {
		Files []pinned `json:"files"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repo, "internal", "modelrecipe", "moss_ultra_import.json")))(t), &recipe))
	encoderPath := "text_encoder/" + golden.Encoder
	if !slices.Contains(recipe.Files, pinned{Path: encoderPath, SHA256: golden.SHA256}) || ultraFiles["text_encoder"] != encoderPath {
		t.Fatalf("the golden read %s at %s; the import recipe pins no such file, or the Ultra root reads another", golden.Encoder, golden.SHA256)
	}
	roots := testcheck.Must(dataroot.Resolve(repo))(t)
	root := ultraModelRoot(t)
	conditioner := testcheck.Must(loadModelConditioner(root, textTemplate{MaxTokens: mossConditioningTokens, EnableThinking: true}))(t)
	t.Cleanup(func() { testcheck.NoError(t, conditioner.close()) })
	texts := make([]textInput, len(golden.Cases))
	embeddings := make([][]float32, len(golden.Cases))
	for index, recorded := range golden.Cases {
		text, embedding, err := conditioner.input(recorded.Prompt)
		testcheck.NoError(t, err)
		if !slices.Equal(text.IDs, recorded.IDs) {
			t.Fatalf("%q renders to %v; want ComfyUI's %v", recorded.Prompt, text.IDs, recorded.IDs)
		}
		texts[index], embeddings[index] = text, embedding
	}
	session := testcheck.Must(graphruntime.NewResidentSession(0))(t)
	t.Cleanup(func() { testcheck.NoError(t, session.Close(context.WithoutCancel(ctx))) })
	hidden := testcheck.Must(conditioner.encodeEach(ctx, session, texts, embeddings))(t)

	width := int(conditioner.catalog.Spec.EmbeddingLength)
	references := filepath.Join(roots.References(), "comfy", "moss-ultra", "encoder")
	for index, recorded := range golden.Cases {
		data := testcheck.Must(os.ReadFile(filepath.Join(references, recorded.States+".f32")))(t)
		if digest := sha256.Sum256(data); hex.EncodeToString(digest[:]) != recorded.States {
			t.Fatalf("%q: the reference states hash %x; want %s", recorded.Prompt, digest, recorded.States)
		}
		want := make([]float32, len(data)/4)
		for at := range want {
			want[at] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*at:]))
		}
		tokens := len(recorded.IDs)
		if !slices.Equal(recorded.Shape, []int{1, tokens, width}) || len(want) != tokens*width || len(hidden[index]) != tokens*width {
			t.Fatalf("%q: ComfyUI's states hold %d values of shape %v and overgo's %d; want %d tokens of %d", recorded.Prompt, len(want), recorded.Shape, len(hidden[index]), tokens, width)
		}
		var differenceSquares, wantSquares float64
		lowest, lowestToken := 1.0, 0
		for token := range tokens {
			got, reference := hidden[index][token*width:(token+1)*width], want[token*width:(token+1)*width]
			var dot, gotSquares, referenceSquares float64
			for column, value := range got {
				g, r := float64(value), float64(reference[column])
				dot += g * r
				gotSquares += g * g
				referenceSquares += r * r
				differenceSquares += (g - r) * (g - r)
			}
			wantSquares += referenceSquares
			if cosine := dot / math.Sqrt(gotSquares*referenceSquares); !(cosine >= lowest) {
				lowest, lowestToken = cosine, token
			}
		}
		relative := math.Sqrt(differenceSquares / wantSquares)
		t.Logf("%q: %d tokens, lowest token cosine %.6f (token %d), relative L2 %.5f", recorded.Prompt, tokens, lowest, lowestToken, relative)
		if !(lowest >= ultraEncoderTokenCosine) || !(relative <= ultraEncoderRelativeL2) {
			t.Errorf("%q: token %d's state lies at cosine %.6f of ComfyUI's and the prompt's at relative L2 %.5f; want at least %g and at most %g",
				recorded.Prompt, lowestToken, lowest, relative, ultraEncoderTokenCosine, ultraEncoderRelativeL2)
		}
	}
}
