//go:build windows

package mediacapability

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/heldfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/testcheck"
)

// TestAServedPictureReusesItsHeldFiles holds a MOSS picture session to identifying each file it reads beside its model
// once (adapter-hold): served as the image capability serves a latent picture request (capabilityruntime.Reading over
// preparedFirst, its files named by pictureFiles through the session's held files, heldfile.Session), a session that
// reads the adapter and the depth estimator as the picture session reads them (latentimage's fileContent), owning the
// model's directory as a picture session does from its load to its close (heldfile.Set.Own), opens and hashes each
// once at the first draw, its naming and its reading together; the same request again is named as before,
// so the store gives its answer again, and opens and hashes nothing; another seed draws again and opens and hashes
// nothing. While the session holds them no one writes either file or renames another onto it. Once the session lets
// them go (its close), other bytes under the adapter's name, size and time are read again and named apart, so the
// earlier answer answers nothing.
func TestAServedPictureReusesItsHeldFiles(t *testing.T) {
	model := t.TempDir()
	adapter, estimator := filepath.Join(model, "adapters", "moss-control.safetensors"), filepath.Join(model, "depth", "depth_anything_v2_vits.pth")
	installed := time.Now().Add(-time.Hour)
	install := func(file, bytes string) {
		t.Helper()
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(file), 0o755))
		testcheck.NoError(t, os.WriteFile(file, []byte(bytes), 0o600))
		testcheck.NoError(t, os.Chtimes(file, installed, installed))
	}
	install(adapter, "the control adapter's weights")
	install(estimator, "the depth estimator's weights")
	// The session's held files, as heldfile.Session holds them, each open counted with the bytes it hashes, a file's whole.
	var opens int64
	var hashed uint64
	held := &heldfile.Set{Open: func(path string) (*os.File, error) {
		file, err := holdForCheck(path)
		if err == nil {
			info, statErr := file.Stat()
			if statErr != nil {
				return nil, errors.Join(statErr, file.Close())
			}
			opens, hashed = opens+1, hashed+uint64(info.Size())
		}
		return file, err
	}}
	// The session owns its model's directory from its first draw to its close, as a picture session does from its load.
	owned := func() error { return nil }
	defer func() { testcheck.NoError(t, owned()) }()
	drawn := 0
	session := func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
		if drawn == 0 {
			release, err := held.Own(model)
			if err != nil {
				return nil, err
			}
			owned = release
		}
		drawn++
		for _, file := range []string{adapter, estimator} {
			if _, err := held.Identity(file); err != nil {
				return nil, err
			}
		}
		return "drawn", nil
	}
	naming := capabilityruntime.Contents(held, pictureFiles)
	serve := capabilityruntime.Reading(preparedFirst(session), naming)
	request := func(seed int) string {
		return `{"prompt":"a fox","width":1024,"height":576,"seed":` + string(rune('0'+seed)) +
			`,"adapter":"moss-control","control":{"picture":"a picture"}}`
	}
	name := func(raw string) string {
		t.Helper()
		named, release, err := naming(t.Context(), nil, model, modelrecipe.CapabilityEvidenceSelection{}, raw)
		testcheck.NoError(t, err)
		testcheck.NoError(t, release())
		return named
	}
	read := func() (int64, uint64) { return opens, hashed }
	baseOpens, baseHashed := read()
	testcheck.Must(serve(t.Context(), nil, model, modelrecipe.CapabilityEvidenceSelection{}, request(7)))(t)
	firstOpens, firstHashed := read()
	size := uint64(len("the control adapter's weights") + len("the depth estimator's weights"))
	if drawn != 1 || firstOpens-baseOpens != 2 || firstHashed-baseHashed != size {
		t.Fatalf("the first draw drew %d times, opened %d files and hashed %d bytes; want 1 draw, its adapter and estimator opened and their %d bytes hashed once",
			drawn, firstOpens-baseOpens, firstHashed-baseHashed, size)
	}
	first := name(request(7))
	if again, _ := read(); name(request(7)) != first || again != firstOpens {
		t.Fatalf("the same request again is named apart or opened its files again; want the name its stored answer is kept under, reading nothing")
	}
	testcheck.Must(serve(t.Context(), nil, model, modelrecipe.CapabilityEvidenceSelection{}, request(8)))(t)
	if after, afterHashed := read(); drawn != 2 || after != firstOpens || afterHashed != firstHashed {
		t.Fatalf("another seed drew %d times in all and opened %d files and hashed %d bytes more; want a second draw reading nothing", drawn, after-firstOpens, afterHashed-firstHashed)
	}
	if os.WriteFile(adapter, []byte("weights switched while held"), 0o600) == nil {
		t.Error("the adapter was written while its session held it")
	}
	other := filepath.Join(model, "adapters", "other.safetensors")
	install(other, "the other control weights!!!")
	if os.Rename(other, adapter) == nil {
		t.Error("another file was renamed onto the adapter while its session held it")
	}
	if os.Rename(other, estimator) == nil {
		t.Error("another file was renamed onto the depth estimator while its session held it")
	}
	testcheck.NoError(t, owned())
	install(adapter, "the other control weights!!!")
	if replaced := name(request(7)); replaced == first || replaced == "" {
		t.Fatal("other bytes under the adapter's name, size and time are named as the earlier adapter's after the session let it go; want a run of their own")
	}
	if reopened, _ := read(); reopened != firstOpens+2 {
		t.Fatalf("after the release the request opened %d files more; want the replaced adapter and the estimator opened again", reopened-firstOpens)
	}
}

// holdForCheck opens path sharing reading alone, as heldfile.Session opens a file it holds.
func holdForCheck(path string) (*os.File, error) {
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	handle, err := syscall.CreateFile(name, syscall.GENERIC_READ, syscall.FILE_SHARE_READ, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, &fs.PathError{Op: "open", Path: path, Err: err}
	}
	return os.NewFile(uintptr(handle), path), nil
}
