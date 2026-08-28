// Package objectstore owns streamed, content-addressed object bytes while
// OvergoDB remains authoritative for identity and availability.
package objectstore

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/fsatomic"
)

const stagingDirectory = ".staging"

type Store struct {
	root       string
	repository artifact.Repository
}

type PublishRequest struct {
	Kind      artifact.Kind
	Expected  *artifact.ID
	MediaType string
	Schema    string
	Reader    io.Reader
}

type Publication struct {
	Descriptor artifact.Descriptor
	Path       string
	Commit     artifact.CommitID
}

func New(root string, repository artifact.Repository) (*Store, error) {
	if root == "" || repository == nil {
		return nil, errors.New("object store: root and repository are required")
	}
	absolute, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("object store: resolve root: %w", err)
	}
	if err := os.MkdirAll(filepath.Join(absolute, stagingDirectory), os.ModePerm); err != nil {
		return nil, fmt.Errorf("object store: create root: %w", err)
	}
	return &Store{root: absolute, repository: repository}, nil
}

// Publish streams one object to durable content-addressed storage, then
// records its descriptor and live file location in OvergoDB.
func (store *Store) Publish(ctx context.Context, request PublishRequest) (Publication, error) {
	if store == nil || store.repository == nil || ctx == nil || request.Reader == nil || request.Kind == artifact.KindInvalid {
		return Publication{}, errors.New("object store: invalid publication")
	}
	if request.Expected != nil && (!request.Expected.Valid() || request.Expected.Kind() != request.Kind) {
		return Publication{}, errors.New("object store: invalid expected identity")
	}
	if err := ctx.Err(); err != nil {
		return Publication{}, err
	}
	temporary, err := os.CreateTemp(filepath.Join(store.root, stagingDirectory), "object-*.partial")
	if err != nil {
		return Publication{}, fmt.Errorf("object store: create staging object: %w", err)
	}
	temporaryPath := temporary.Name()
	keepTemporary := true
	defer func() {
		_ = temporary.Close()
		if keepTemporary {
			_ = os.Remove(temporaryPath)
		}
	}()
	id, size, writeErr := artifact.Identify(request.Kind, io.TeeReader(contextReader{ctx: ctx, reader: request.Reader}, temporary))
	if writeErr != nil {
		return Publication{}, writeErr
	}
	if request.Expected != nil && id != *request.Expected {
		return Publication{}, fmt.Errorf("object store: identity mismatch: have %s want %s", id, *request.Expected)
	}
	descriptor := artifact.Descriptor{ID: id, Size: size, MediaType: request.MediaType, Schema: request.Schema}
	if err := descriptor.Validate(); err != nil {
		return Publication{}, err
	}
	if err := temporary.Sync(); err != nil {
		return Publication{}, fmt.Errorf("object store: sync staged object: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return Publication{}, fmt.Errorf("object store: close staged object: %w", err)
	}
	target, err := store.install(temporaryPath, descriptor)
	if err != nil {
		return Publication{}, err
	}
	keepTemporary = false
	location, err := artifact.CanonicalLocalLocation(id, artifact.LocationFile, target)
	if err != nil {
		return Publication{}, err
	}
	batch := artifact.Batch{
		Key:       "objectstore/publish/" + id.String(),
		Artifacts: []artifact.Descriptor{descriptor},
		Locations: []artifact.LocationEvent{{Location: location, Action: artifact.LocationAdd}},
	}
	commit, err := artifact.CommitBatch(ctx, store.repository, batch)
	if err != nil {
		return Publication{}, err
	}
	return Publication{Descriptor: descriptor, Path: target, Commit: commit}, nil
}

// Open returns a streaming reader only for an object whose descriptor and
// availability have already been published to OvergoDB.
func (store *Store) Open(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.ReadCloser, error) {
	if store == nil || store.repository == nil || ctx == nil || !id.Valid() {
		return artifact.Descriptor{}, nil, errors.New("object store: invalid open")
	}
	descriptor, found, err := store.repository.Artifact(ctx, id)
	if err != nil || !found {
		if err == nil {
			err = errors.New("object store: descriptor is absent")
		}
		return artifact.Descriptor{}, nil, err
	}
	path := store.objectPath(id)
	locations, err := store.repository.Locations(ctx, id)
	if err != nil {
		return artifact.Descriptor{}, nil, err
	}
	found = false
	for _, location := range locations {
		if location.Kind == artifact.LocationFile && location.Value == path {
			found = true
			break
		}
	}
	if !found {
		return artifact.Descriptor{}, nil, errors.New("object store: object location is not published")
	}
	file, err := os.Open(path)
	if err != nil {
		return artifact.Descriptor{}, nil, err
	}
	info, err := file.Stat()
	if err != nil || uint64(info.Size()) != descriptor.Size {
		_ = file.Close()
		if err == nil {
			err = errors.New("object store: object size differs from descriptor")
		}
		return artifact.Descriptor{}, nil, err
	}
	return descriptor, file, nil
}

func (store *Store) install(temporaryPath string, descriptor artifact.Descriptor) (string, error) {
	directory := filepath.Join(store.root, descriptor.ID.Kind().String())
	if err := os.MkdirAll(directory, os.ModePerm); err != nil {
		return "", fmt.Errorf("object store: create kind directory: %w", err)
	}
	target := store.objectPath(descriptor.ID)
	if err := os.Link(temporaryPath, target); err == nil {
		if err := os.Remove(temporaryPath); err != nil {
			_ = os.Remove(target)
			return "", fmt.Errorf("object store: retire staged object: %w", err)
		}
		if err := fsatomic.SyncDirectory(directory); err != nil {
			_ = os.Remove(target)
			return "", err
		}
		return target, nil
	} else if !errors.Is(err, os.ErrExist) {
		return "", fmt.Errorf("object store: install object: %w", err)
	}
	file, err := os.Open(target)
	if err != nil {
		return "", fmt.Errorf("object store: open existing object: %w", err)
	}
	existing, size, identifyErr := artifact.Identify(descriptor.ID.Kind(), file)
	closeErr := file.Close()
	if identifyErr != nil || closeErr != nil {
		return "", errors.Join(identifyErr, closeErr)
	}
	if existing != descriptor.ID || size != descriptor.Size {
		return "", errors.New("object store: existing object conflicts with identity")
	}
	if err := os.Remove(temporaryPath); err != nil {
		return "", fmt.Errorf("object store: remove duplicate staging object: %w", err)
	}
	return target, nil
}

func (store *Store) objectPath(id artifact.ID) string {
	return filepath.Join(store.root, id.Kind().String(), id.DigestHex())
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (reader contextReader) Read(buffer []byte) (int, error) {
	if err := reader.ctx.Err(); err != nil {
		var unread int
		return unread, err
	}
	read, err := reader.reader.Read(buffer)
	if err == nil {
		err = reader.ctx.Err()
	}
	return read, err
}
