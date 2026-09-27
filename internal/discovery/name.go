package discovery

import (
	"path/filepath"
	"strings"
)

// genericSegments are the weights files and repository component
// directories the HF and diffusers layouts name the same in every model,
// so none of them names a model.
var genericSegments = map[string]bool{
	"text_encoder": true, "transformer": true, "vae": true,
	"scheduler": true, "tokenizer": true,
	"classification": true, "regression": true,
}

// LocationName derives a distinctive name from a recorded weights location:
// a single-file GGUF names itself; otherwise the deepest path segment that
// is not a weights file or a generic component directory. The rules are
// format facts of the GGUF, HF and diffusers layouts, not model knowledge.
func LocationName(location string) string {
	segments := strings.Split(filepath.ToSlash(location), "/")
	if last := segments[len(segments)-1]; strings.HasSuffix(strings.ToLower(last), ".gguf") {
		return strings.TrimSuffix(last, filepath.Ext(last))
	}
	for index := len(segments) - 1; index >= 0; index-- {
		segment := segments[index]
		lower := strings.ToLower(segment)
		if strings.HasSuffix(lower, ".safetensors") || strings.HasSuffix(lower, ".pt") || genericSegments[lower] {
			continue
		}
		return segment
	}
	return filepath.Base(location)
}
