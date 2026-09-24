package overgodb

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/fsatomic"
)

// A pack holds the blobs too small to deserve a file of their own. It is one
// file, so one rename publishes it whole: a header, an index sorted by
// identity, a digest over the index, then the payloads the index points at.
// Where a blob's bytes sit is a storage fact the journal never states, so
// packing changes no commit. A loose file wins over a packed copy; the pack
// is read only when the loose file is absent, and its index is loaded on the
// first such miss.

const (
	blobPackFilename = "small.pack"
	blobPackMagic    = "OVGPACK1"
	// A pack entry is the identity -- kind, digest -- then the payload's
	// offset in the file and its size.
	packKindBytes   = 1
	packKeyBytes    = packKindBytes + sha256.Size
	packEntryBytes  = packKeyBytes + 8 + 4
	packHeaderBytes = len(blobPackMagic) + 8
)

// blobPack is the loaded index of one store's pack, shared by the copies of
// its blobStore.
type blobPack struct {
	mu     sync.Mutex
	loaded bool
	index  []byte
}

// entries returns the verified index, loading it once; a store with no pack,
// or one whose index does not match its digest, has none.
func (p *blobPack) entries(root string) []byte {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.loaded {
		p.index, p.loaded = readPackIndex(filepath.Join(root, blobPackFilename)), true
	}
	return p.index
}

// forget drops the loaded index after the pack file has been replaced.
func (p *blobPack) forget() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.index, p.loaded = nil, false
}

func readPackIndex(path string) []byte {
	file, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer file.Close()
	header := make([]byte, packHeaderBytes)
	if _, err := io.ReadFull(file, header); err != nil || string(header[:len(blobPackMagic)]) != blobPackMagic {
		return nil
	}
	info, err := file.Stat()
	if err != nil {
		return nil
	}
	// The index and its digest must fit in the file after the header, so a
	// damaged count is refused before it sizes an allocation; the division
	// also keeps the product from overflowing.
	count := binary.BigEndian.Uint64(header[len(blobPackMagic):])
	room := info.Size() - int64(packHeaderBytes) - sha256.Size
	if room < 0 || count > uint64(room)/uint64(packEntryBytes) {
		return nil
	}
	index := make([]byte, count*uint64(packEntryBytes)+sha256.Size)
	if _, err := io.ReadFull(file, index); err != nil {
		return nil
	}
	index, digest := index[:len(index)-sha256.Size], index[len(index)-sha256.Size:]
	if sum := sha256.Sum256(index); !bytes.Equal(sum[:], digest) {
		return nil
	}
	return index
}

func packKey(id artifact.ID) ([]byte, error) {
	digest, err := hex.DecodeString(id.DigestHex())
	if err != nil || !id.Valid() || len(digest) != sha256.Size {
		return nil, errors.New("overgodb: invalid blob identity")
	}
	return append([]byte{byte(id.Kind())}, digest...), nil
}

// packed finds where the pack holds id's payload.
func (b blobStore) packed(id artifact.ID) (offset int64, size int64, found bool) {
	key, err := packKey(id)
	if err != nil || b.pack == nil {
		return offset, size, found
	}
	index := b.pack.entries(b.root)
	at, found := sort.Find(len(index)/packEntryBytes, func(entry int) int {
		return bytes.Compare(key, index[entry*packEntryBytes:][:packKeyBytes])
	})
	if !found {
		return offset, size, found
	}
	place := index[at*packEntryBytes+packKeyBytes:]
	return int64(binary.BigEndian.Uint64(place)), int64(binary.BigEndian.Uint32(place[8:])), true
}

// VerifyBlobs holds every live blob to its identity, loose or packed, and
// says how many it read: the check a candidate passes after its blobs have
// been moved.
func (s *Store) VerifyBlobs(ctx context.Context) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// The pack is read once and every member held to its identity; a blob
	// with no loose file is then answered by its place in the verified pack
	// rather than by opening the pack again for each of its members.
	if err := s.blobs.verifyPack(); err != nil {
		return 0, err
	}
	checked := 0
	for id, locator := range s.state.contents.locators {
		if !locator.blob || locator.released != 0 {
			continue
		}
		if err := ctx.Err(); err != nil {
			return checked, err
		}
		path, err := s.blobs.path(id)
		if err != nil {
			return checked, err
		}
		if _, statErr := os.Stat(path); statErr == nil {
			err = s.blobs.verify(id)
		} else if _, _, packed := s.blobs.packed(id); !packed {
			err = fmt.Errorf("overgodb: blob %s is neither loose nor packed", id)
		}
		if err != nil {
			return checked, err
		}
		checked++
	}
	return checked, nil
}

// verifyPack streams every pack member through one buffer and holds its bytes
// to the digest its index entry names, so verification holds one member's
// buffer rather than the pack; a store with no pack has nothing to verify.
func (b blobStore) verifyPack() error {
	file, err := os.Open(filepath.Join(b.root, blobPackFilename))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	index := b.pack.entries(b.root)
	if index == nil {
		return errors.New("overgodb: blob pack index does not match its digest")
	}
	buffer := make([]byte, inlineContentLimit)
	hash := sha256.New()
	for entry := range slices.Chunk(index, packEntryBytes) {
		place := entry[packKeyBytes:]
		offset, size := binary.BigEndian.Uint64(place), uint64(binary.BigEndian.Uint32(place[8:]))
		if offset+size > uint64(info.Size()) {
			return fmt.Errorf("overgodb: blob pack member %x lies outside the pack", entry[packKindBytes:packKeyBytes])
		}
		hash.Reset()
		if _, err := io.CopyBuffer(hash, io.NewSectionReader(file, int64(offset), int64(size)), buffer); err != nil {
			return err
		}
		if !bytes.Equal(hash.Sum(nil), entry[packKindBytes:packKeyBytes]) {
			return fmt.Errorf("overgodb: blob pack member %x bytes do not hash to their identity", entry[packKindBytes:packKeyBytes])
		}
	}
	return nil
}

// writePack writes a pack -- header, index, index digest -- then streams each
// member from its loose file or the pack it replaces through one buffer,
// holding each to its identity and to the size its index entry names. It
// returns the payload bytes written.
func writePack(ctx context.Context, destination io.Writer, blobs blobStore, members []artifact.ID, index, digest []byte) (int64, error) {
	header := binary.BigEndian.AppendUint64([]byte(blobPackMagic), uint64(len(members)))
	if err := writeAll(destination, slices.Concat(header, index, digest)); err != nil {
		return 0, err
	}
	// Members are the blobs under the inline limit, so a buffer of that
	// size reads a member in one call.
	buffer := make([]byte, inlineContentLimit)
	var packed int64
	for position, id := range members {
		if err := ctx.Err(); err != nil {
			return packed, err
		}
		size := int64(binary.BigEndian.Uint32(index[position*packEntryBytes+packKeyBytes+8:]))
		written, err := blobs.copyVerified(destination, id, buffer)
		if err != nil {
			return packed, err
		}
		if written != size {
			return packed, fmt.Errorf("overgodb: blob %s size differs from its committed size", id)
		}
		packed += written
	}
	return packed, nil
}

// PackReport says what one pack of a store's small blobs did.
type PackReport struct {
	Packed       int   `json:"packed"`
	PackedBytes  int64 `json:"packed_bytes"`
	LooseRemoved int   `json:"loose_removed"`
}

// PackSmallBlobs moves every live blob under the inline limit into one pack.
// Each payload is held to its identity as it is read, from its loose file or
// from the pack it replaces; the new pack is published by one rename, and
// only then are the loose files it holds removed, so an interruption leaves
// loose files, a pack, or both, and never neither.
func PackSmallBlobs(ctx context.Context, store *Store) (PackReport, error) {
	return packBlobsUnder(ctx, store, inlineContentLimit)
}

func packBlobsUnder(ctx context.Context, store *Store, limit int64) (PackReport, error) {
	var report PackReport
	err := store.writeTransaction(ctx, func() error {
		blobs := store.blobs
		var small []artifact.ID
		for id, locator := range store.state.contents.locators {
			if locator.blob && locator.released == 0 && locator.size < limit {
				small = append(small, id)
			}
		}
		slices.SortFunc(small, func(left, right artifact.ID) int {
			leftKey, _ := packKey(left)
			rightKey, _ := packKey(right)
			return bytes.Compare(leftKey, rightKey)
		})
		// The index is laid out from the committed sizes, so the payloads
		// stream into the pack one at a time; a member whose bytes differ
		// from its size or identity refuses the pack.
		index := make([]byte, 0, len(small)*packEntryBytes)
		next := int64(packHeaderBytes + len(small)*packEntryBytes + sha256.Size)
		for _, id := range small {
			key, _ := packKey(id)
			size := store.state.contents.locators[id].size
			index = binary.BigEndian.AppendUint64(append(index, key...), uint64(next))
			index = binary.BigEndian.AppendUint32(index, uint32(size))
			next += size
		}
		digest := sha256.Sum256(index)
		if err := os.MkdirAll(blobs.root, storeDirectoryMode); err != nil {
			return err
		}
		staging, err := os.CreateTemp(blobs.root, ".staging-*")
		if err != nil {
			return fmt.Errorf("overgodb: stage blob pack: %w", err)
		}
		defer os.Remove(staging.Name())
		packed, err := writePack(ctx, staging, blobs, small, index, digest[:])
		if err := errors.Join(err, staging.Sync(), staging.Close()); err != nil {
			return fmt.Errorf("overgodb: write blob pack: %w", err)
		}
		if err := fsatomic.Replace(staging.Name(), filepath.Join(blobs.root, blobPackFilename)); err != nil {
			return fmt.Errorf("overgodb: publish blob pack: %w", err)
		}
		if err := fsatomic.SyncDirectory(blobs.root); err != nil {
			return err
		}
		blobs.pack.forget()
		report.Packed, report.PackedBytes = len(small), packed
		for _, id := range small {
			removed, err := blobs.remove(id)
			if err != nil {
				return err
			}
			if removed {
				report.LooseRemoved++
			}
		}
		return nil
	})
	return report, err
}
