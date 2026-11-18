//go:build windows

package mediacapability

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/modelartifact"
)

// pictureBeside is the identity of the files a picture request reads beside its model's components, in its order, each
// its content (modelartifact.IdentifyWeights), by which its run is keyed beside the request (capabilityruntime
// BindContent): the adapters it or its preset names, by their files' names in the model's folder, and the depth
// estimator a control picture is read by (latentimage Request.Beside). A file replaced under its name gave the earlier
// file's picture again (SQA-406). None where it reads none.
func pictureBeside(ctx context.Context, path string, request latentimage.Request) (string, error) {
	files, err := request.Beside(path)
	if err != nil {
		return "", err
	}
	identities := make([]string, 0, len(files))
	for _, file := range files {
		identity, err := besideIdentity(ctx, file)
		if err != nil {
			return "", fmt.Errorf("image-gen: the request reads %s beside its model: %w", file, err)
		}
		identities = append(identities, identity.String())
	}
	return strings.Join(identities, ","), nil
}

// besideRead is a file read beside a model as it stood when it was identified.
type besideRead struct {
	size     int64
	modified time.Time
	identity artifact.ID
}

var (
	besideMu    sync.Mutex
	besideFiles = map[string]besideRead{}
)

// besideIdentity is the identity of the file at path, read again only where its size or modification time moved since
// it was read: a Turbo request's three adapters are 1.8 GB, which every request read anew.
func besideIdentity(ctx context.Context, path string) (artifact.ID, error) {
	info, err := os.Stat(path)
	if err != nil {
		return artifact.ID{}, err
	}
	besideMu.Lock()
	read, held := besideFiles[path]
	besideMu.Unlock()
	if held && read.size == info.Size() && read.modified.Equal(info.ModTime()) {
		return read.identity, nil
	}
	identity, err := modelartifact.IdentifyWeights(ctx, path)
	if err != nil {
		return artifact.ID{}, err
	}
	besideMu.Lock()
	besideFiles[path] = besideRead{size: info.Size(), modified: info.ModTime(), identity: identity}
	besideMu.Unlock()
	return identity, nil
}
