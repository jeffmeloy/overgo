package modeltest

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// ggufExtension names a registered location a GGUF header can be read from.
const ggufExtension = ".gguf"

// GGUF returns the smallest registered GGUF file whose header declares the
// architecture: a test names the architecture it exercises, and the store's
// registrations say where such a model is. A checkout whose store registers
// none cannot apply the test and says so.
func GGUF(t testing.TB, architecture string) string {
	t.Helper()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(roots.Store); errors.Is(err, fs.ErrNotExist) {
		testskip.NotApplicable(t, "no store holds model registrations here")
	}
	locations, err := registeredLocations(context.Background(), roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	var smallest string
	var smallestSize int64
	for _, location := range locations {
		if !strings.EqualFold(filepath.Ext(location), ggufExtension) {
			continue
		}
		info, err := os.Stat(location)
		if err != nil || smallest != "" && info.Size() >= smallestSize {
			continue
		}
		if declaredArchitecture(location) == architecture {
			smallest, smallestSize = location, info.Size()
		}
	}
	if smallest == "" {
		testskip.NotApplicable(t, "no GGUF registered here declares the architecture "+architecture)
	}
	return smallest
}

var (
	architecturesMutex sync.Mutex
	architecturesRead  = map[string]string{}
)

// architectureKey names the metadata entry a GGUF declares its architecture
// in; maxKeyBytes bounds the header strings the fast read accepts.
const (
	architectureKey = "general.architecture"
	maxKeyBytes     = 256
)

// declaredArchitecture reads a GGUF header's architecture once per process;
// a file that cannot be read declares none.
func declaredArchitecture(path string) string {
	architecturesMutex.Lock()
	defer architecturesMutex.Unlock()
	if architecture, read := architecturesRead[path]; read {
		return architecture
	}
	architecture, found := leadingArchitecture(path)
	if !found {
		if file, err := gguf.Open(path); err == nil {
			if value, found := file.MetadataValue(architectureKey); found {
				architecture, _ = value.Data.(string)
			}
			_ = file.Close()
		}
	}
	architecturesRead[path] = architecture
	return architecture
}

// leadingArchitecture reads the architecture when it is the header's first
// metadata entry, as converters write it: a few bytes of the file rather
// than a whole header whose vocabulary arrays take seconds to parse.
func leadingArchitecture(path string) (string, bool) {
	file, err := os.Open(path)
	if err != nil {
		return "", false
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	var header struct {
		Magic             [len(gguf.Magic)]byte
		Version           uint32
		Tensors, Metadata uint64
	}
	if binary.Read(reader, binary.LittleEndian, &header) != nil || string(header.Magic[:]) != gguf.Magic || header.Metadata == 0 {
		return "", false
	}
	key, ok := headerString(reader)
	if !ok || key != architectureKey {
		return "", false
	}
	var valueType gguf.ValueType
	if binary.Read(reader, binary.LittleEndian, &valueType) != nil || valueType != gguf.ValueTypeString {
		return "", false
	}
	return headerString(reader)
}

// headerString reads one length-prefixed GGUF string.
func headerString(reader io.Reader) (string, bool) {
	var length uint64
	if binary.Read(reader, binary.LittleEndian, &length) != nil || length > maxKeyBytes {
		return "", false
	}
	data := make([]byte, length)
	if _, err := io.ReadFull(reader, data); err != nil {
		return "", false
	}
	return string(data), true
}
