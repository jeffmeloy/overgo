package organ

import (
	"fmt"
	"sort"
	"strings"
)

// MLPComponent is one indexed, complete gated-MLP organ: the three tensor
// bindings plus the classified contract of its gate. Ordinal is the position
// in the catalog's deterministic layer order.
type MLPComponent struct {
	Ordinal  int
	Gate     string
	Up       string
	Down     string
	Contract Contract
}

// Catalog is the compiled component index for one model's weight inventory:
// every tensor name is classified ONCE at compile time and grouped into typed,
// indexed components. Consumers look components up; they never rescan names.
// It is the in-memory authority the store-level component catalog row will
// persist cross-model.
type Catalog struct {
	mlps []MLPComponent
}

// CompileCatalog classifies an inventory into the indexed catalog. Incomplete
// component groups are recorded as absent rather than half-bound; a prefix
// that classifies two tensors into the same role is refused by name because a
// catalog with ambiguous bindings is not an authority.
func CompileCatalog(names []string) (Catalog, error) {
	type group struct{ gate, up, down string }
	groups := map[string]*group{}
	bind := func(slot *string, prefix, name string, role Role) error {
		if *slot != "" {
			return fmt.Errorf("organ: prefix %q classifies two %s tensors (%s, %s)", prefix, role, *slot, name)
		}
		*slot = name
		return nil
	}
	for _, name := range names {
		contract := Classify(name, "f32", "", "text", "")
		prefix := componentPrefix(name)
		if groups[prefix] == nil && (contract.Role == RoleMLPGate || contract.Role == RoleMLPUp || contract.Role == RoleMLPDown) {
			groups[prefix] = &group{}
		}
		var err error
		switch contract.Role {
		case RoleMLPGate:
			err = bind(&groups[prefix].gate, prefix, name, contract.Role)
		case RoleMLPUp:
			err = bind(&groups[prefix].up, prefix, name, contract.Role)
		case RoleMLPDown:
			err = bind(&groups[prefix].down, prefix, name, contract.Role)
		}
		if err != nil {
			return Catalog{}, err
		}
	}
	prefixes := make([]string, 0, len(groups))
	for prefix, candidate := range groups {
		if candidate.gate != "" && candidate.up != "" && candidate.down != "" {
			prefixes = append(prefixes, prefix)
		}
	}
	sort.Strings(prefixes)
	catalog := Catalog{}
	for ordinal, prefix := range prefixes {
		bound := groups[prefix]
		catalog.mlps = append(catalog.mlps, MLPComponent{
			Ordinal: ordinal, Gate: bound.gate, Up: bound.up, Down: bound.down,
			Contract: Classify(bound.gate, "f32", "", "text", ""),
		})
	}
	return catalog, nil
}

// MLP returns the indexed component; ok=false when the index is outside the
// compiled catalog.
func (c Catalog) MLP(index int) (MLPComponent, bool) {
	if index < 0 || index >= len(c.mlps) {
		return MLPComponent{}, false
	}
	return c.mlps[index], true
}

// MLPCount reports how many complete MLP components compiled.
func (c Catalog) MLPCount() int { return len(c.mlps) }

// componentPrefix groups a tensor with its layer siblings: everything before
// the final two name segments (projection and parameter kind).
func componentPrefix(name string) string {
	parts := strings.Split(name, ".")
	if len(parts) <= 2 {
		return name
	}
	return strings.Join(parts[:len(parts)-2], ".")
}
