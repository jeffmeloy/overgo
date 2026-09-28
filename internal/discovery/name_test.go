package discovery

import "testing"

// TestLocationNameNamesTheModelNotItsParts names a single-file GGUF by
// itself and a directory model by the deepest segment that is neither a
// weights file nor a generic component directory, dotted names included.
func TestLocationNameNamesTheModelNotItsParts(t *testing.T) {
	t.Parallel()
	for location, want := range map[string]string{
		`C:\models\Qwen2.5-0.5B-f16.gguf`:                                "Qwen2.5-0.5B-f16",
		`C:\models\Qwen2.5-0.5B\model.safetensors`:                       "Qwen2.5-0.5B",
		`C:\models\Qwen2.5-0.5B`:                                         "Qwen2.5-0.5B",
		`C:\models\Krea-2-Turbo\text_encoder\model.safetensors`:          "Krea-2-Turbo",
		`C:\models\tabfm-1.0.0-pytorch\classification\model.safetensors`: "tabfm-1.0.0-pytorch",
		`C:\models\LiveEdit\ar-forcing_002000.pt`:                        "LiveEdit",
		`C:\models\Wan\model-00001-of-00008.safetensors`:                 "Wan",
	} {
		if got := LocationName(location); got != want {
			t.Errorf("LocationName(%q) = %q, want %q", location, got, want)
		}
	}
}
