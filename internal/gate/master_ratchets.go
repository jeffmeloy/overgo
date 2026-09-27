package gate

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
	"strings"
)

// masterRatchets are the reviewed ceilings and floors master owns. A merge
// may tighten them and loosens one only with a declared paydown row: true
// marks a ceiling, which a rise or a new entry loosens; false a floor, which a
// fall or a removed entry loosens.
var masterRatchets = map[string]bool{
	"docs/clone_baseline.json": true, "docs/harness_surface_baseline.json": true, "docs/structure_budgets.json": true,
	"docs/modern_go_baseline.json": true, "docs/family_branch_baseline.json": true,
	"docs/staged_surface.json":          true,
	"docs/coverage_lanes_baseline.json": false,
}

// refuseLooserRatchets compares each of master's ratchets in the merged tree
// with the first parent's and refuses a loosening the merge row does not pay
// down, so a lane cannot bring its looser checks home.
func (g *gateContext) refuseLooserRatchets() error {
	candidate, err := g.plannedTree()
	if err != nil {
		return err
	}
	var loosened []string
	for _, path := range slices.Sorted(maps.Keys(masterRatchets)) {
		before, beforeErr := gitAuthorityOutput(g.repo, "show", g.planHead+":"+path)
		after, afterErr := gitAuthorityOutput(g.repo, "show", candidate+":"+path)
		var parent, merged any
		switch {
		case beforeErr != nil:
			continue // master holds no such ratchet yet
		case afterErr != nil:
			loosened = append(loosened, path+" removed")
		case json.Unmarshal(before, &parent) != nil || json.Unmarshal(after, &merged) != nil:
			return fmt.Errorf("master ratchet %s is not JSON", path)
		default:
			loosened = append(loosened, looserRatchet(path, parent, merged, masterRatchets[path])...)
		}
	}
	row, _, err := g.row()
	if len(loosened) == 0 || err != nil || row.Budget.Paydown != "" {
		return err
	}
	return fmt.Errorf("merge loosens master's ratchets: %s; restore master's values or declare the paydown row", strings.Join(loosened, "; "))
}

// looserRatchet names every place after loosens before.
func looserRatchet(path string, before, after any, ceiling bool) []string {
	var found []string
	switch after := after.(type) {
	case float64:
		if before, ok := before.(float64); ok && (ceiling && after > before || !ceiling && after < before) {
			found = append(found, fmt.Sprintf("%s %v -> %v", path, before, after))
		}
	case map[string]any:
		before, _ := before.(map[string]any)
		for key := range mergedKeys(before, after) {
			previous, held := before[key]
			value, kept := after[key]
			switch {
			case held && kept:
				found = append(found, looserRatchet(path+"."+key, previous, value, ceiling)...)
			case ceiling && kept:
				found = append(found, path+"."+key+" added")
			case !ceiling && held:
				found = append(found, path+"."+key+" removed")
			}
		}
	case []any:
		before, _ := before.([]any)
		entries := func(values []any) map[string]bool {
			set := map[string]bool{}
			for _, value := range values {
				encoded, _ := json.Marshal(value)
				set[string(encoded)] = true
			}
			return set
		}
		previous, current := entries(before), entries(after)
		for entry := range mergedKeys(previous, current) {
			if ceiling && current[entry] && !previous[entry] || !ceiling && previous[entry] && !current[entry] {
				found = append(found, path+" entry "+entry)
			}
		}
	}
	slices.Sort(found)
	return found
}

// mergedKeys is the union of two maps' keys.
func mergedKeys[V any](left, right map[string]V) map[string]bool {
	keys := map[string]bool{}
	for key := range left {
		keys[key] = true
	}
	for key := range right {
		keys[key] = true
	}
	return keys
}
