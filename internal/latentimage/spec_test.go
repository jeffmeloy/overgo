package latentimage

import (
	"os"
	"testing"
)

// kreaModelDir: the real Krea-2-Turbo artifact (READ-ONLY). Absent -> skip
// (worktrees / non-model hosts have no models/); this is a structural oracle,
// not a unit test of pure logic.
const kreaModelDir = `C:\Users\jeffm\adaptive_new\models\Krea-2-Turbo`

func kreaDirOrSkip(t *testing.T) string {
	t.Helper()
	if _, err := os.Stat(kreaModelDir + `\model_index.json`); err != nil {
		t.Skipf("UNAVAILABLE: Krea-2-Turbo not present: %v", err)
	}
	return kreaModelDir
}

// TestRecognizePipeline: the recognizer classifies the real artifact by
// model_index.json class and returns a fully config-cross-checked spec.
func TestRecognizePipeline(t *testing.T) {
	dir := kreaDirOrSkip(t)
	spec, ok, err := RecognizePipeline(dir)
	if err != nil {
		t.Fatalf("RecognizePipeline: %v", err)
	}
	if !ok {
		t.Fatal("RecognizePipeline: not recognized")
	}
	if spec.Pipeline != PipelineClass || spec.Family != FamilyTag || !spec.ServingOnly {
		t.Fatalf("tags: pipeline=%q family=%q servingOnly=%v", spec.Pipeline, spec.Family, spec.ServingOnly)
	}
	if spec.Tokenizer != TokenizerQwen2 {
		t.Fatalf("tokenizer=%q", spec.Tokenizer)
	}
	t.Logf("recognized: pipeline=%s family=%s scheduler=%s distilled=%v patch=%d serving_only=%v tokenizer=%s",
		spec.Pipeline, spec.Family, spec.Scheduler, spec.Distilled, spec.PatchSize, spec.ServingOnly, spec.Tokenizer)
	t.Logf("transformer: layers=%d heads=%d kv=%d head_dim=%d hidden=%d kv_dim=%d in_ch=%d ffn=%d rope=%v theta=%g tstep=%d",
		spec.Transformer.Layers, spec.Transformer.Heads, spec.Transformer.KVHeads, spec.Transformer.HeadDim,
		spec.Transformer.Hidden, spec.Transformer.KVDim, spec.Transformer.InChannels, spec.Transformer.Intermediate,
		spec.Transformer.RopeAxes, spec.Transformer.RopeTheta, spec.Transformer.TimestepEmbed)
	t.Logf("text-stream: text_layers=%d text_hidden=%d text_ffn=%d text_heads=%d layerwise=%d refiner=%d",
		spec.Transformer.TextLayers, spec.Transformer.TextHidden, spec.Transformer.TextIntermediate,
		spec.Transformer.TextHeads, spec.Transformer.LayerwiseTextBlocks, spec.Transformer.RefinerTextBlocks)
	t.Logf("vae: z_dim=%d base_dim=%d dim_mult=%v deepest=%d quant_ch=%d spatial=%dx res_blocks=%d in_ch=%d mean/std_len=%d/%d",
		spec.VAE.ZDim, spec.VAE.BaseDim, spec.VAE.DimMult, spec.VAE.DeepestDim, spec.VAE.QuantChannels,
		spec.VAE.SpatialScale, spec.VAE.ResBlocks, spec.VAE.InputChannels, len(spec.VAE.LatentsMean), len(spec.VAE.LatentsStd))
	t.Logf("text-encoder: type=%s layers=%d hidden=%d heads=%d kv=%d head_dim=%d vocab=%d theta=%g select=%v",
		spec.TextEncoder.ModelType, spec.TextEncoder.HiddenLayers, spec.TextEncoder.Hidden, spec.TextEncoder.Heads,
		spec.TextEncoder.KVHeads, spec.TextEncoder.HeadDim, spec.TextEncoder.VocabSize, spec.TextEncoder.RopeTheta,
		spec.TextEncoder.SelectLayers)
}

// TestVerifyCheckpoint: the CPU structural oracle -- every derived dim asserted
// against the real tensor shapes (safetensors headers only, no payload, no
// forward), and the cross-checks agree. Fails on any single disagreement.
func TestVerifyCheckpoint(t *testing.T) {
	if testing.Short() {
		t.Skip("opens 3 sub-model safetensors headers; skipped in -short")
	}
	dir := kreaDirOrSkip(t)
	spec, err := Derive(dir)
	if err != nil {
		t.Fatalf("Derive: %v", err)
	}
	witness, err := spec.VerifyCheckpoint(dir)
	if witness != nil {
		for _, c := range witness.SortedChecks() {
			t.Log(c.String())
		}
	}
	if err != nil {
		t.Fatalf("VerifyCheckpoint: %v", err)
	}
	t.Logf("STRUCTURAL ORACLE PASS: %d dims cross-checked vs real checkpoint (mod_fields derived=%d)",
		len(witness.Checks), witness.ModField)
}

// TestEnumerate: scanning the models root surfaces the artifact as a
// serving-in-progress image-diffusion media model (serving path not yet built).
func TestEnumerate(t *testing.T) {
	kreaDirOrSkip(t)
	root := `C:\Users\jeffm\adaptive_new\models`
	models, err := Enumerate(root)
	if err != nil {
		t.Fatalf("Enumerate: %v", err)
	}
	var found *MediaModel
	for i := range models {
		if models[i].Spec.Pipeline == PipelineClass {
			found = &models[i]
		}
		t.Logf("enumerated: dir=%s family=%s kind=%s status=%q serving=%v",
			models[i].Dir, models[i].Family, models[i].Kind, models[i].Status, models[i].Serving)
	}
	if found == nil {
		t.Fatal("Krea-2-Turbo not enumerated from models root")
	}
	if found.Serving || found.Status != StatusRecognized || found.Kind != mediaKind || found.Family != FamilyTag {
		t.Fatalf("bad descriptor: %+v", *found)
	}
}

// TestRecognizeNonPipeline: a directory without model_index.json is not
// recognized and is not an error (discovery may probe any directory).
func TestRecognizeNonPipeline(t *testing.T) {
	tmp := t.TempDir()
	spec, ok, err := RecognizePipeline(tmp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok || spec != nil {
		t.Fatalf("empty dir recognized: ok=%v spec=%v", ok, spec)
	}
}
