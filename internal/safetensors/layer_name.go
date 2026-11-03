package safetensors

import (
	"strings"

	"overgo/internal/textcheck"
)

// LayerName reads a repeated layer's tensor name: prefix, the layer's number in decimal, a dot and the rest of the name.
func LayerName(name, prefix string) (layer, rest string, ok bool) {
	tail, found := strings.CutPrefix(name, prefix)
	if !found {
		return "", "", false
	}
	layer, rest, found = strings.Cut(tail, ".")
	if !found || rest == "" || !textcheck.Only(layer, textcheck.Digits) {
		return "", "", false
	}
	return layer, rest, true
}
