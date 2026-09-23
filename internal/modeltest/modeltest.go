// Package modeltest finds a registered model's files for a test through the
// registration the store holds, not by joining a models root with a
// directory name: the root a checkout configures need not hold every model
// the store has registered, and a name joined to it guesses where a model is
// instead of reading where it was registered. It lives apart from testutil
// so the many tests that need no model do not take on the store.
package modeltest

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// registeredBound bounds the registered model listing far above the number of
// models a store registers, as the media census bounds its own read.
const registeredBound = 4096

// Directory returns the directory named name that holds a registered model's
// recorded files. A test whose model this checkout's store has not registered
// cannot apply here and says so.
func Directory(t testing.TB, name string) string {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(roots.Store); errors.Is(err, fs.ErrNotExist) {
		t.Skip(testskip.Inapplicable + ": no store holds model registrations here")
	}
	directory, err := registeredDirectory(context.Background(), roots.Store, name)
	if err != nil {
		t.Fatal(err)
	}
	if directory == "" {
		t.Skip(testskip.Inapplicable + ": no model registered here sits in a directory named " + name)
	}
	return directory
}

// registeredDirectory finds the directory named name above any registered
// model's recorded location, or reports none.
func registeredDirectory(ctx context.Context, storePath, name string) (string, error) {
	locations, err := registeredLocations(ctx, storePath)
	if err != nil {
		return "", err
	}
	for _, location := range locations {
		for directory := location; directory != filepath.Dir(directory); directory = filepath.Dir(directory) {
			if filepath.Base(directory) == name {
				return directory, nil
			}
		}
	}
	return "", nil
}

var (
	locationsMutex sync.Mutex
	locationsRead  = map[string][]string{}
)

// registeredLocations reads every registered model's recorded locations once
// per process: the registrations do not change under a running test binary,
// and opening the store for each test would cost seconds apiece.
func registeredLocations(ctx context.Context, storePath string) ([]string, error) {
	locationsMutex.Lock()
	defer locationsMutex.Unlock()
	if locations, read := locationsRead[storePath]; read {
		return locations, nil
	}
	store, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		return nil, err
	}
	defer store.Close()
	models, err := store.Query(ctx, overgodb.Query{Kind: artifact.KindModel, MaxResults: registeredBound, Projection: overgodb.ProjectArtifacts})
	if err != nil {
		return nil, err
	}
	if models.Truncated {
		return nil, errors.New("modeltest: the registered model listing is truncated")
	}
	var locations []string
	for _, model := range models.Artifacts {
		recorded, err := store.Locations(ctx, model.ID)
		if err != nil {
			return nil, err
		}
		for _, location := range recorded {
			locations = append(locations, location.Value)
		}
	}
	locationsRead[storePath] = locations
	return locations, nil
}
