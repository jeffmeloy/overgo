package composition

import (
	"fmt"
	"sort"
	"strings"

	"overgo/internal/organ"
)

// SelectDonorMLP picks a donor MLP component by ORGAN CLASSIFICATION, not by
// hardcoded family tensor names: every weight name is classified, gate/up/down
// roles are grouped by their shared name prefix (the layer), prefixes are
// ordered, and the requested index selects one complete triple. Any naming
// convention whose tokens classify -- HF model.layers.N.*, GGUF blk.N.* --
// works unchanged; an incomplete triple is refused by name.
func SelectDonorMLP(weights map[string][]float32, layer int) (gate, up, down string, contract organ.Contract, err error) {
	triples := map[string]*triple{}
	for name := range weights {
		classified := organ.Classify(name, "f32", "", "text", "")
		var slot *string
		switch classified.Role {
		case organ.RoleMLPGate:
			slot = &ensure(triples, mlpPrefix(name)).gate
		case organ.RoleMLPUp:
			slot = &ensure(triples, mlpPrefix(name)).up
		case organ.RoleMLPDown:
			slot = &ensure(triples, mlpPrefix(name)).down
		default:
			continue
		}
		if *slot != "" {
			return "", "", "", organ.Contract{}, fmt.Errorf("composition: prefix %q classifies two %s tensors (%s, %s)", mlpPrefix(name), classified.Role, *slot, name)
		}
		*slot = name
	}
	prefixes := make([]string, 0, len(triples))
	for prefix, candidate := range triples {
		if candidate.gate != "" && candidate.up != "" && candidate.down != "" {
			prefixes = append(prefixes, prefix)
		}
	}
	if len(prefixes) == 0 {
		return "", "", "", organ.Contract{}, fmt.Errorf("composition: donor has no complete organ-classified MLP triple among %d tensors", len(weights))
	}
	sort.Strings(prefixes)
	if layer < 0 || layer >= len(prefixes) {
		return "", "", "", organ.Contract{}, fmt.Errorf("composition: donor MLP index %d outside %d classified triples", layer, len(prefixes))
	}
	selected := triples[prefixes[layer]]
	return selected.gate, selected.up, selected.down, organ.Classify(selected.gate, "f32", "", "text", ""), nil
}

// mlpPrefix groups a tensor with its layer siblings: everything before the
// final two name segments (the projection and the parameter kind).
func mlpPrefix(name string) string {
	parts := strings.Split(name, ".")
	if len(parts) <= 2 {
		return name
	}
	return strings.Join(parts[:len(parts)-2], ".")
}

func ensure(triples map[string]*triple, prefix string) *triple {
	if triples[prefix] == nil {
		triples[prefix] = &triple{}
	}
	return triples[prefix]
}

type triple struct{ gate, up, down string }
