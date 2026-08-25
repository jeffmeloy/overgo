package modelswap

import (
	"context"
	"path/filepath"
	"strings"

	"overgo/internal/discovery"
	"overgo/internal/overgodb"
)

// CatalogResolver resolves model names against the store's servable
// capability catalog: the store is the configuration, never a config
// file. Names match the artifact's on-disk basename with or without
// its extension, or the model identity itself.
type CatalogResolver struct {
	// Store is the OvergoDB root, opened read-only per resolve so the
	// serving child keeps the writer lock.
	Store string
	// Limit bounds the catalog listing.
	Limit int
}

// Resolve maps one requested name to a launchable servable.
func (r CatalogResolver) Resolve(ctx context.Context, name string) (Servable, bool, error) {
	store, err := overgodb.OpenReadOnly(r.Store)
	if err != nil {
		return Servable{}, false, err
	}
	defer store.Close()
	entries, _, err := discovery.CapabilityCatalog(ctx, store, r.Limit, nil)
	if err != nil {
		return Servable{}, false, err
	}
	wanted := strings.ToLower(name)
	for _, entry := range entries {
		if !entry.Present || entry.Location == "" {
			continue
		}
		base := filepath.Base(entry.Location)
		stem := strings.TrimSuffix(base, filepath.Ext(base))
		if strings.EqualFold(base, name) || strings.EqualFold(stem, name) ||
			strings.ToLower(entry.Model.String()) == wanted {
			return Servable{
				Name: base, Location: entry.Location, Model: entry.Model.String(),
			}, true, nil
		}
	}
	return Servable{}, false, nil
}
