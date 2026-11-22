//go:build modeltest && windows

package latentimage

import (
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	// The converter opens a Hugging Face directory as the model it serves (gguf.RegisterDirectoryOpener).
	_ "overgo/internal/hfconvert"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

// TestPromptModesTwoAndThreeEqualTheRecordedStrings holds prompt modes 2 and 3 (moss-ultra-sections/prompt-modes) to
// the MOSS Ultra workflow's recordings of them (fixtures/comfy/moss-ultra/recordings/mode-2 and mode-3, the positive
// and negative its preview nodes 436 and 592 showed). Mode 3's request, the builder's fields as the recording's request
// gives them, makes the recorded prompt. Mode 2's text is sampled, so ComfyUI's is taken as the chat's answer: the chain
// keeps the recorded positive as the first answer it is, and cleans the recorded negative to itself. Then the request
// the recording asked, its source and exclusions, is enhanced through the Ultra root's enhancer chat, sampled as
// TextGenerate samples, to a request drawable by the picture session: a prompt of what the chat wrote and a negative
// the chain's cleaning leaves as it is.
func TestPromptModesTwoAndThreeEqualTheRecordedStrings(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	data := testcheck.Must(mossEnhancer())(t)
	text := func(section comfySection, node string) string {
		t.Helper()
		if len(section.Texts[node]) != 1 {
			t.Fatalf("the recording shows %d texts at %s, want one", len(section.Texts[node]), node)
		}
		return section.Texts[node][0]
	}
	field := func(inputs map[string]any, name string) string {
		value, _ := inputs[name].(string)
		return value
	}

	three := readComfySection(t, "mode-3", "text-to-image")
	builder := three.inputs("450")
	if three.Set["483.value"] != float64(3) || field(builder, "style_palette_data") != "" || field(builder, "elements_data") != "" {
		t.Fatalf("the recording set %v over a builder of %v; want mode 3 with neither a palette nor a region", three.Set, builder)
	}
	draft := LayoutDraft{Description: field(builder, "high_level_description"), Background: field(builder, "background"),
		Style: field(builder, "style"), Photo: field(builder, "style.photo"), ArtStyle: field(builder, "style.art_style"),
		Aesthetics: field(builder, "aesthetics"), Lighting: field(builder, "lighting"), Medium: field(builder, "medium")}
	laidOut := Request{Width: 1024, Height: 576, Layout: &draft}
	testcheck.NoError(t, ValidateRequest(laidOut))
	if got, want := laidOut.prompt(), text(three, "436"); got != want {
		t.Errorf("mode 3 makes the prompt %q, the recording %q", got, want)
	}

	two := readComfySection(t, "mode-2", "text-to-image")
	source, exclusions := field(two.inputs("459"), "value"), field(two.inputs("460"), "value")
	positive, negative := text(two, "436"), text(two, "592")
	if two.Set["483.value"] != float64(2) || !hasEnhancerWords(positive) {
		t.Fatalf("the recording set %v and wrote a positive of %d words; want mode 2's first answer kept", two.Set, len(strings.Fields(positive)))
	}
	if got := enhancedPositive(source, positive, ""); got != positive {
		t.Errorf("the chain keeps %q of the recorded first answer, want it whole", got)
	}
	if got := enhancedNegative(negative, source, exclusions, data.NegativeBackup); got != negative {
		t.Errorf("the chain cleans the recorded negative to %q, want %q", got, negative)
	}

	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	t.Cleanup(func() { testcheck.NoError(t, store.Close()) })
	testcheck.Must(modelrecipe.PublishArchitectureProfileCatalog(ctx, store))(t)
	root := ultraModelRoot(t)
	ultraEnhancer(t, root)
	enhanced := testcheck.Must(EnhanceRequest(ctx, enhancerOpener(store), root, Request{PromptMode: promptEnhanced, Prompt: source, NegativePrompt: exclusions, Width: 1024, Height: 576}))(t)
	testcheck.NoError(t, ValidateRequest(enhanced))
	if enhanced.Prompt == "" || enhanced.NegativePrompt == "" || enhancedNegative(enhanced.NegativePrompt, source, exclusions, data.NegativeBackup) != enhanced.NegativePrompt {
		t.Errorf("the enhancer wrote %q and %q; want a prompt and a negative the cleaning leaves as it is", enhanced.Prompt, enhanced.NegativePrompt)
	}
	t.Logf("the enhancer wrote a prompt of %d words (%t past the first answer's %d) and the negative %q",
		len(strings.Fields(enhanced.Prompt)), hasEnhancerWords(enhanced.Prompt), enhancerWords, enhanced.NegativePrompt)
}
