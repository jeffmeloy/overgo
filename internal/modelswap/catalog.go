package modelswap

import (
	"context"
	"errors"
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
	// Store is borrowed from the proxy and refreshed before catalog access.
	Store *overgodb.Store
	// Limit bounds the catalog listing.
	Limit int

	mu   sync.Mutex
	memo *discovery.Memo
}

func (r *CatalogResolver) refresh(ctx context.Context) error {
	if r.Store == nil {
		return errors.New("model catalog: repository is required")
	}
	if err := r.Store.Refresh(ctx); err != nil {
		return err
	}
	if r.memo == nil {
		r.memo = discovery.LoadMemo(ctx, r.Store)
	}
	return nil
}

// SetKey places a hosted provider's key in this process for the model the
// reference names in the refreshed catalog.
func (r *CatalogResolver) SetKey(ctx context.Context, reference, key string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.refresh(ctx); err != nil {
		return err
	}
	_, err := remoteprovider.SetKey(ctx, r.Store, r.Limit, reference, key)
	return err
}

// Catalog lists the store's activations for the idle shell's picker. The
// retained view refreshes so declarations made by other processes appear.
func (r *CatalogResolver) Catalog(ctx context.Context) ([]discovery.CatalogEntry, bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.refresh(ctx); err != nil {
		return nil, false, err
	}
	return discovery.CapabilityCatalog(ctx, r.Store, r.Limit, r.memo)
}

// Resolve maps one requested name to a launchable servable. A name the
// refreshed catalog resolves committed activations and retirements.
func (r *CatalogResolver) Resolve(ctx context.Context, name string) (Servable, bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.refresh(ctx); err != nil {
		return Servable{}, false, err
	}
	entries, _, err := discovery.CapabilityCatalog(ctx, r.Store, r.Limit, r.memo)
	if err != nil {
		return Servable{}, false, err
	}
	servable, found := matchServable(entries, name)
	return servable, found, nil
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
