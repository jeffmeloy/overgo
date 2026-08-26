package modelswap

import (
	"context"
	"path/filepath"
	"strings"
	"sync"

	"overgo/internal/discovery"
	"overgo/internal/overgodb"
)

// CatalogResolver resolves model names against the store's servable
// capability catalog: the store is the configuration, never a config
// file. Names match the artifact's on-disk basename with or without
// its extension, or the model identity itself.
type CatalogResolver struct {
	// Store is the OvergoDB root, opened read-only so the serving child
	// keeps the writer lock.
	Store string
	// Limit bounds the catalog listing.
	Limit int

	// The replayed handle and catalog memo persist across resolves --
	// replaying the log and scanning the catalog cost more than the
	// model load they route. A miss reopens once to see new artifacts.
	mu     sync.Mutex
	opened *overgodb.Store
	memo   *discovery.Memo
}

// Resolve maps one requested name to a launchable servable. A name the
// cached catalog does not know triggers one reopen, so artifacts
// committed after the cache was built stay servable.
func (r *CatalogResolver) Resolve(ctx context.Context, name string) (Servable, bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, reopen := range []bool{false, true} {
		if reopen {
			if r.opened != nil {
				_ = r.opened.Close()
				r.opened = nil
			}
		}
		if r.opened == nil {
			store, err := overgodb.OpenReadOnly(r.Store)
			if err != nil {
				return Servable{}, false, err
			}
			r.opened = store
			// Persisted identity evidence spares the resolver re-hashing
			// the model bytes the serving child already identified.
			r.memo = discovery.LoadMemo(ctx, store)
		}
		entries, _, err := discovery.CapabilityCatalog(ctx, r.opened, r.Limit, r.memo)
		if err != nil {
			return Servable{}, false, err
		}
		if servable, found := matchServable(entries, name); found {
			return servable, true, nil
		}
	}
	return Servable{}, false, nil
}

func matchServable(entries []discovery.CatalogEntry, name string) (Servable, bool) {
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
			}, true
		}
	}
	return Servable{}, false
}
