//go:build modeltest && windows

package latentimage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	// The converter opens a Hugging Face directory as the model it serves (gguf.RegisterDirectoryOpener).
	_ "overgo/internal/hfconvert"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/servingtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	tokens "overgo/internal/tokenizer"
)

// enhancerNearTie is the gap between a step's two highest logits within which ComfyUI's arithmetic and overgo's may
// choose either: ComfyUI runs the int8 head and projections on activations quantized per row after the rotation and
// gives bfloat16 logits (a step of 1/8 at the logits' size), where overgo decodes the weights to bfloat16 and carries
// its activations in F32. Declared before the first run; there the chosen token's logit lay at most 0.33 from ComfyUI's.
const enhancerNearTie = 0.5

// enhancerHeld is the fewest tokens a case's greedy text holds before the two part, or its whole where it is shorter:
// the first run held 14 of the positive's (parting where ComfyUI's two highest bfloat16 logits were equal and its argmax
// took the lower ID) and all 50 of the negative's.
const enhancerHeld = 12

// TestTheEnhancersGreedyTextHoldsComfyUIs holds prompt mode 2's enhancer chat (moss-ultra-sections/enhancer-chat) to
// ComfyUI's TextGenerate on the same file, recorded by scripts/python_golden_references/comfy_enhancer_greedy_golden.py
// into fixtures/comfy/moss-ultra/enhancer_greedy.json: the Ultra encoder file with its head, beside Qwen3.5-9B's own
// config for generation (fixtures/comfy/moss-ultra/enhancer/config.json, Qwen/Qwen3.5-9B's config.json, sha256
// d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05) and MOSS-Image-SFT's tokenizer and chat template,
// is served as a chat model through overgo's own view of a Hugging Face directory, its int8 ConvRot weights decoded as
// the repository reads them. Each recorded request, set in the chat template with thinking off, tokenizes to ComfyUI's
// prompt IDs exactly, and overgo's greedy decode from them emits ComfyUI's tokens one for one to the end or to where the
// two part, which is a step where either side's two highest logits lie within enhancerNearTie; the parting is reported.
func TestTheEnhancersGreedyTextHoldsComfyUIs(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	repo := testutil.RepoRoot(t)
	var golden struct {
		Encoder string `json:"encoder"`
		Type    string `json:"clip_type"`
		Cases   []struct {
			Name      string `json:"name"`
			System    string `json:"system"`
			Request   string `json:"request"`
			Prompt    []int  `json:"prompt_ids"`
			Generated []int  `json:"generated_ids"`
			Steps     []struct {
				IDs    [2]int     `json:"ids"`
				Logits [2]float64 `json:"logits"`
				Chosen int        `json:"chosen"`
			} `json:"steps"`
		} `json:"cases"`
	}
	fixtures := filepath.Join(repo, "fixtures", "comfy", "moss-ultra")
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(fixtures, "enhancer_greedy.json")))(t), &golden))
	if len(golden.Cases) == 0 || golden.Type != "moss_image" || "text_encoder/"+golden.Encoder != ultraFiles["text_encoder"] {
		t.Fatalf("the golden records %d cases of %q from %s; want moss_image's, from the file the Ultra root reads", len(golden.Cases), golden.Type, golden.Encoder)
	}
	enhancer := testcheck.Must(mossEnhancer())(t)
	systems := map[string]string{"positive_system": enhancer.PositiveSystem, "negative_system_photo": enhancer.NegativeSystemPhoto,
		"negative_system_protected": enhancer.NegativeSystemProtected}

	root, directory := ultraModelRoot(t), t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "config.json"), testcheck.Must(os.ReadFile(filepath.Join(fixtures, "enhancer", "config.json")))(t), 0o600))
	for _, name := range []string{"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, name), testcheck.Must(os.ReadFile(filepath.Join(root, "tokenizer", name)))(t), 0o600))
	}
	testcheck.NoError(t, os.Link(filepath.Join(root, filepath.FromSlash(ultraFiles["text_encoder"])), filepath.Join(directory, golden.Encoder)))
	loaded := testcheck.Must(servingtest.ResolveActiveGGUFWithPolicy(directory, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyDeviceNative))(t)
	runner := testcheck.Must(inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{}))(t)
	t.Cleanup(func() { testcheck.NoError(t, runner.Close()) })
	greedy := testcheck.Must(sampling.New(sampling.Config{Temperature: 0}))(t)

	for _, recorded := range golden.Cases {
		system, found := systems[recorded.System]
		if !found || len(recorded.Steps) != len(recorded.Generated) {
			t.Fatalf("%s: system %q, %d steps for %d tokens", recorded.Name, recorded.System, len(recorded.Steps), len(recorded.Generated))
		}
		text := testcheck.Must(runner.FormatChatWithOptions([]inference.ChatMessage{
			{Role: inference.ChatRoleSystem, Content: system}, {Role: inference.ChatRoleUser, Content: recorded.Request},
		}, inference.ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: false}))(t)
		prompt := testcheck.Must(runner.TokenizeText(text, false, true))(t)
		want := make([]tokens.TokenID, len(recorded.Prompt))
		for index, id := range recorded.Prompt {
			want[index] = tokens.TokenID(id)
		}
		if !slices.Equal(prompt, want) {
			t.Errorf("%s: the chat tokenizes to %d IDs, ComfyUI's to %d, first differing at %d", recorded.Name, len(prompt), len(want), firstDiffering(prompt, want))
			continue
		}
		type step struct {
			id         tokens.TokenID
			first, gap float64
		}
		var steps []step
		options := inference.GenerateOptions{
			PromptTokenIDs: want, MaxNewTokens: len(recorded.Steps), Sampler: greedy,
			OnToken: func(event inference.TokenEvent) error {
				best, second := topTwo(event.Logits)
				steps = append(steps, step{id: event.ID, first: float64(event.Logits[best]), gap: float64(event.Logits[best] - event.Logits[second])})
				return nil
			},
		}
		testcheck.Must(func() ([]tokens.TokenID, error) {
			ids, _, err := runner.Generate(ctx, "", options)
			return ids, err
		}())(t)
		held, ties, largest := 0, 0, 0.0
		for index, comfy := range recorded.Steps {
			if index >= len(steps) {
				t.Errorf("%s: overgo stopped after %d tokens, ComfyUI went on to %d", recorded.Name, len(steps), len(recorded.Steps))
				break
			}
			ours, comfyGap := steps[index], comfy.Logits[0]-comfy.Logits[1]
			nearTie := comfyGap <= enhancerNearTie || ours.gap <= enhancerNearTie
			if int(ours.id) != comfy.Chosen {
				if !nearTie {
					t.Errorf("%s: parts at step %d, beyond a near tie: ComfyUI chose %d (gap %g), overgo %d (gap %g)", recorded.Name, index, comfy.Chosen, comfyGap, ours.id, ours.gap)
				}
				t.Logf("%s: parts at step %d, a near tie: ComfyUI chose %d of %v (gap %g), overgo %d (gap %g)", recorded.Name, index, comfy.Chosen, comfy.IDs, comfyGap, ours.id, ours.gap)
				break
			}
			if nearTie {
				ties++
			}
			largest = max(largest, abs64(ours.first-comfy.Logits[0]))
			held++
		}
		if held < enhancerHeld && held < len(recorded.Steps) {
			t.Errorf("%s: held %d tokens before parting, want at least %d", recorded.Name, held, enhancerHeld)
		}
		t.Logf("%s: %d of %d tokens held, %d of them through near ties; the chosen token's logit lies at most %g from ComfyUI's",
			recorded.Name, held, len(recorded.Steps), ties, largest)
	}
}

// topTwo is the indices of the highest and second-highest logits.
func topTwo(logits []float32) (int, int) {
	best, second := 0, 1
	if logits[second] > logits[best] {
		best, second = second, best
	}
	for index := 2; index < len(logits); index++ {
		switch value := logits[index]; {
		case value > logits[best]:
			best, second = index, best
		case value > logits[second]:
			second = index
		}
	}
	return best, second
}

// firstDiffering is the first index at which two ID sequences differ, or the shorter's length.
func firstDiffering(left, right []tokens.TokenID) int {
	for index := range min(len(left), len(right)) {
		if left[index] != right[index] {
			return index
		}
	}
	return min(len(left), len(right))
}

func abs64(value float64) float64 {
	if value < 0 {
		return -value
	}
	return value
}
