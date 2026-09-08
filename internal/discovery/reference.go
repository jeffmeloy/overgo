package discovery

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
)

// MatchReference resolves a present catalog entry by exact identity or location,
// then by an unambiguous basename or stem. It refuses incomplete catalogs rather
// than choosing an alias whose collision may have been omitted.
func MatchReference(entries []CatalogEntry, truncated bool, reference string) (CatalogEntry, bool, error) {
	if truncated {
		return CatalogEntry{}, false, fmt.Errorf("model catalog: cannot resolve %q from a truncated catalog", reference)
	}
	var selected CatalogEntry
	found := false
	exact := slices.ContainsFunc(entries, func(entry CatalogEntry) bool {
		return entry.Present && entry.Location != "" && (strings.EqualFold(reference, entry.Model.String()) || reference == entry.Location)
	})
	for _, entry := range entries {
		if !entry.Present || entry.Location == "" {
			continue
		}
		identity := strings.EqualFold(reference, entry.Model.String()) || reference == entry.Location
		base := filepath.Base(entry.Location)
		alias := strings.EqualFold(reference, base) || strings.EqualFold(reference, strings.TrimSuffix(base, filepath.Ext(base)))
		if exact && !identity || !exact && !alias {
			continue
		}
		if found && (selected.Model != entry.Model || selected.Location != entry.Location) {
			return CatalogEntry{}, false, fmt.Errorf("model catalog: ambiguous reference %q; use an exact model identity or location", reference)
		}
		selected, found = entry, true
	}
	return selected, found, nil
}
