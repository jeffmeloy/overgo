//overgo:runtime-inputs caller

package modelswap

import (
	"context"
	"errors"
	"maps"
	"path/filepath"
	"sync"
	"time"

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

	// Stage timing, reported when a picker stalls behind the resolver.
	timing  sync.Mutex
	stage   string
	since   time.Time
	waiting int
	slowest map[string]time.Duration
}

// Progress is the resolver's stage timing: the stage a caller is inside
// with its elapsed time, the callers waiting for the resolver, and the
// slowest completed run of each stage.
type Progress struct {
	Stage   string
	Elapsed time.Duration
	Waiting int
	Slowest map[string]time.Duration
}

// Progress reports the resolver's stage timing.
func (r *CatalogResolver) Progress() Progress {
	r.timing.Lock()
	defer r.timing.Unlock()
	progress := Progress{Stage: r.stage, Waiting: r.waiting, Slowest: maps.Clone(r.slowest)}
	if r.stage != "" {
		progress.Elapsed = time.Since(r.since)
	}
	return progress
}

// takes the resolver's lock, counting the caller as waiting until it holds it
func (r *CatalogResolver) acquire() func() {
	r.timing.Lock()
	r.waiting++
	r.timing.Unlock()
	r.mu.Lock()
	r.timing.Lock()
	r.waiting--
	r.timing.Unlock()
	return r.mu.Unlock
}

// enters a stage under the resolver's lock and returns its end, which keeps
// the stage's slowest run
func (r *CatalogResolver) begin(stage string) func() {
	began := time.Now()
	r.timing.Lock()
	r.stage, r.since = stage, began
	r.timing.Unlock()
	return func() {
		took := time.Since(began)
		r.timing.Lock()
		defer r.timing.Unlock()
		r.stage = ""
		if r.slowest == nil {
			r.slowest = map[string]time.Duration{}
		}
		r.slowest[stage] = max(r.slowest[stage], took)
	}
}

func (r *CatalogResolver) refresh(ctx context.Context) error {
	if r.Store == nil {
		return errors.New("model catalog: repository is required")
	}
	end := r.begin("refresh")
	err := r.Store.Refresh(ctx)
	end()
	if err != nil {
		return err
	}
	if r.memo == nil {
		defer r.begin("memo")()
		r.memo = discovery.LoadMemo(ctx, r.Store)
	}
	return nil
}

// SetKey places a hosted provider's key in this process for the model the
// reference names in the refreshed catalog.
func (r *CatalogResolver) SetKey(ctx context.Context, reference, key string) error {
	defer r.acquire()()
	if err := r.refresh(ctx); err != nil {
		return err
	}
	defer r.begin("key")()
	_, err := remoteprovider.SetKey(ctx, r.Store, r.Limit, reference, key)
	return err
}

// Catalog lists the store's activations for the idle shell's picker. The
// retained view refreshes so declarations made by other processes appear.
func (r *CatalogResolver) Catalog(ctx context.Context) ([]discovery.CatalogEntry, bool, error) {
	defer r.acquire()()
	if err := r.refresh(ctx); err != nil {
		return nil, false, err
	}
	defer r.begin("catalog")()
	return discovery.CapabilityCatalog(ctx, r.Store, r.Limit, r.memo)
}

// Resolve maps one requested name to a launchable servable. A name the
// refreshed catalog resolves committed activations and retirements.
func (r *CatalogResolver) Resolve(ctx context.Context, name string) (Servable, bool, error) {
	defer r.acquire()()
	if err := r.refresh(ctx); err != nil {
		return Servable{}, false, err
	}
	end := r.begin("catalog")
	entries, truncated, err := discovery.CapabilityCatalog(ctx, r.Store, r.Limit, r.memo)
	end()
	if err != nil {
		return Servable{}, false, err
	}
	return matchServable(entries, truncated, name)
}

func matchServable(entries []discovery.CatalogEntry, truncated bool, name string) (Servable, bool, error) {
	entry, found, err := discovery.MatchReference(entries, truncated, name)
	if err != nil || !found {
		return Servable{}, found, err
	}
	return Servable{Name: filepath.Base(entry.Location), Location: entry.Location, Model: entry.Model.String()}, true, nil
}
