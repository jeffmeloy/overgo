package modelswap

import (
	"context"
	"path/filepath"
	"strings"
	"sync"

	"overgo/internal/discovery"
	"overgo/internal/overgodb"
	"overgo/internal/remoteprovider"
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
	// model load they route. A miss refreshes once to see new artifacts.
	mu     sync.Mutex
	opened *overgodb.Store
	memo   *discovery.Memo
}

// open replays the store once and refreshes its committed tail on request.
// File identities survive refresh and still validate their live size and time.
func (r *CatalogResolver) open(ctx context.Context, refresh bool) (*overgodb.Store, error) {
	if refresh && r.opened != nil {
		if err := r.opened.Refresh(ctx); err != nil {
			return nil, err
		}
	}
	if r.opened == nil {
		store, err := overgodb.OpenReadOnly(r.Store)
		if err != nil {
			return nil, err
		}
		r.opened = store
		// Persisted identity evidence spares the resolver re-hashing
		// the model bytes the serving child already identified.
		r.memo = discovery.LoadMemo(ctx, store)
	}
	return r.opened, nil
}

// SetKey places a hosted provider's key in this process for the model the
// reference names; a reference the cached catalog does not know refreshes
// once, as Resolve does.
func (r *CatalogResolver) SetKey(ctx context.Context, reference, key string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	var err error
	for _, reopen := range []bool{false, true} {
		store, openErr := r.open(ctx, reopen)
		if openErr != nil {
			return openErr
		}
		if _, err = remoteprovider.SetKey(ctx, store, r.Limit, reference, key); err == nil {
			return nil
		}
	}
	return err
}

// Catalog lists the store's activations for the idle shell's picker. The
// store refreshes each time: with no child running, the CLI is what changes
// it, and a declaration made since must list.
func (r *CatalogResolver) Catalog(ctx context.Context) ([]discovery.CatalogEntry, bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	store, err := r.open(ctx, true)
	if err != nil {
		return nil, false, err
	}
	return discovery.CapabilityCatalog(ctx, store, r.Limit, r.memo)
}

// Resolve maps one requested name to a launchable servable. A name the
// cached catalog does not know triggers one refresh, so artifacts
// committed after the cache was built stay servable.
func (r *CatalogResolver) Resolve(ctx context.Context, name string) (Servable, bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, reopen := range []bool{false, true} {
		store, err := r.open(ctx, reopen)
		if err != nil {
			return Servable{}, false, err
		}
		entries, _, err := discovery.CapabilityCatalog(ctx, store, r.Limit, r.memo)
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
