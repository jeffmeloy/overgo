//go:build windows

package mediacapability

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/heldfile"
	"overgo/internal/latentimage"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// countedStore is a store that counts the reads of each content's payload it gives, by its identity.
type countedStore struct {
	*overgodb.Store
	reads map[artifact.ID]int
}

// OpenContent counts the read of id's payload.
func (s countedStore) OpenContent(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	s.reads[id]++
	return s.Store.OpenContent(ctx, id)
}

// VisitContents counts the read of each of ids' payloads.
func (s countedStore) VisitContents(ctx context.Context, ids []artifact.ID, visit func(artifact.Descriptor, io.Reader) error) error {
	for _, id := range ids {
		s.reads[id]++
	}
	return s.Store.VisitContents(ctx, ids, visit)
}

// repeatWork is what a picture session stood in for MOSS's did: the weights it placed (its loads), the requests it
// was given beside them (its resets) and the pictures it drew.
type repeatWork struct {
	loads, resets, draws int
}

// repeatSession is a picture session standing in for MOSS's (latentimage.Generator): it owns its model's directory from
// its load to its close, as LoadGenerator does, so the files it reads beside its model stay held.
type repeatSession struct {
	release func() error
}

// Close lets its model's directory go.
func (s *repeatSession) Close(context.Context) error { return s.release() }

// TestServedMOSSPictureReusesItsCompletedDraw holds moss-served-repeat: the same admitted, fixed-seed MOSS request sent
// twice through the picture capability's serving (pictureExecutor), on a store holding the model's profile and a
// source PNG the request names by its stored identity, the ultra-image-to-image-turbo preset naming its canvas and its
// two adapters, held by a live session. The first call reads the source picture once and draws once; the second
// gives the same recorded run and pixel bytes, reads no byte of the source picture, opens and hashes no adapter again,
// and places no weights and draws nothing. The store's reads of the returned picture are counted apart. Another seed
// draws again through the held adapters, a fresh run draws again, and the same request with another source picture
// reads that picture and is a run of its own. The model's directory holds no model_index.json once its profile is
// stored, so a request served reads none: its preset is read from the profile. A repeat had read and encoded its
// source picture and read model_index.json before its stored answer was sought.
func TestServedMOSSPictureReusesItsCompletedDraw(t *testing.T) {
	root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	source := testcheck.Must(Catalog[recipe.TaskImageGen].Resolve(t.Context(), root))(t)
	definition, facts, err := source.Define(source.Inventory.Manifest.ID)
	testcheck.NoError(t, err)
	program := testcheck.Must(modelrecipe.CompileCapability(definition))(t)
	store := countedStore{Store: overgodbtest.Open(t, t.TempDir()), reads: map[artifact.ID]int{}}
	batch := testcheck.Must(source.Inventory.Batch(t.Name()))(t)
	batch.Contents = append(batch.Contents, facts...)
	batch.Manifests = append(batch.Manifests, source.Manifests...)
	testcheck.NoError(t, modelrecipe.PublishFacts(t.Context(), store, batch))
	execution := testcheck.Must(modelrecipe.CompileCandidateExecution(t.Context(), store, program))(t)
	testcheck.NoError(t, os.Remove(filepath.Join(root, "model_index.json")))

	// The preset's adapters, installed beside the model.
	var adapters []string
	for _, name := range []string{"8-steps-dmd2-lora", "moss_image_sft"} {
		adapter := filepath.Join(root, "adapters", name+".safetensors")
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(adapter), 0o755))
		testcheck.NoError(t, os.WriteFile(adapter, []byte(name+"'s weights"), 0o600))
		adapters = append(adapters, adapter)
	}
	// Source pictures, stored as a page attaches them.
	storedPNG := func(shade uint8) artifact.ID {
		picture := image.NewNRGBA(image.Rect(0, 0, 64, 36))
		for index := range picture.Pix {
			picture.Pix[index] = shade
		}
		var encoded bytes.Buffer
		testcheck.NoError(t, png.Encode(&encoded, picture))
		data := encoded.Bytes()
		content := artifact.Content{Descriptor: artifact.Descriptor{ID: testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, data))(t), Size: uint64(len(data)), MediaType: "image/png"}, Data: data}
		testcheck.Must(artifact.Publish(t.Context(), store, artifact.Batch{Key: fmt.Sprintf("fixture/source-%d", shade), Artifacts: []artifact.Descriptor{content.Descriptor}, Contents: []artifact.Content{content}}))(t)
		return content.Descriptor.ID
	}
	sourceA, sourceB := storedPNG(90), storedPNG(200)
	sources := map[string]artifact.ID{}
	for _, id := range []artifact.ID{sourceA, sourceB} {
		content := testcheck.Must(func() (artifact.Content, error) {
			content, _, err := artifact.ReadContent(t.Context(), store.Store, id)
			return content, err
		}())(t)
		sources[base64.StdEncoding.EncodeToString(content.Data)] = id
	}

	// The session's held files, each open counted with the bytes it hashes.
	var opens int
	var hashed int64
	held := &heldfile.Set{Open: func(path string) (*os.File, error) {
		file, err := holdForCheck(path)
		if err == nil {
			info, statErr := file.Stat()
			if statErr != nil {
				return nil, errors.Join(statErr, file.Close())
			}
			opens, hashed = opens+1, hashed+info.Size()
		}
		return file, err
	}}
	var work repeatWork
	director := testcheck.Must(capabilityruntime.NewModelSessionDirector[latentimage.Request, *repeatSession, latentimage.EncodedImage](
		"image-gen", imageDevice, imageSessionCapacity, latentimage.ValidateRequest,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, _ latentimage.Request) (*repeatSession, error) {
			release, err := held.Own(path)
			if err != nil {
				return nil, err
			}
			work.loads++
			return &repeatSession{release: release}, nil
		},
		func(context.Context, *repeatSession, latentimage.Request) error {
			work.resets++
			return nil
		},
		func(runtime *workflowruntime.Runtime, modelID artifact.ID, _ *repeatSession) error {
			return workflowruntime.RegisterContextPipeline(runtime, modelID,
				modelrecipe.ModuleLatentImagePrepare, func(_ context.Context, request latentimage.Request) (latentimage.Request, error) {
					// The session takes its source picture's bytes, read where no answer was stored, and reads its
					// adapters as the picture session reads them.
					if _, read := sources[request.Image]; !read {
						return latentimage.Request{}, fmt.Errorf("the session took the source picture as %.40q, not its stored bytes", request.Image)
					}
					for _, adapter := range adapters {
						if _, err := held.Identity(adapter); err != nil {
							return latentimage.Request{}, err
						}
					}
					return request, nil
				},
				modelrecipe.ModuleLatentImageIntegrate, func(_ context.Context, request latentimage.Request) (latentimage.Request, error) {
					return request, nil
				},
				modelrecipe.ModuleLatentImageDecode, func(_ context.Context, request latentimage.Request) (latentimage.EncodedImage, error) {
					work.draws++
					shade := float32(request.Seed%7) / 7
					if sources[request.Image] == sourceB {
						shade += 0.1
					}
					pixels := make([]float32, 3*request.Width*request.Height)
					for index := range pixels {
						pixels[index] = shade
					}
					return latentimage.EncodePlanarPNG(pixels, request.Height, request.Width)
				},
				latentimage.PNGContent)
		}))(t)
	t.Cleanup(testutil.Closer(t, context.Background(), director))
	serve := pictureExecutor(director, nil, held)

	request := func(seed int, picture artifact.ID) string {
		return fmt.Sprintf(`{"prompt":"a lighthouse at dusk","preset":"ultra-image-to-image-turbo","seed":%d,"image":%q}`, seed, picture.String())
	}
	served := func(ctx context.Context, raw string) (capabilityruntime.Measured, latentimage.EncodedImage) {
		t.Helper()
		measured, ok := testcheck.Must(serve(ctx, store, root, execution, raw))(t).(capabilityruntime.Measured)
		if !ok {
			t.Fatal("a served picture came with no measure")
		}
		picture, ok := measured.Output.(latentimage.EncodedImage)
		if !ok {
			t.Fatalf("a served picture is %T", measured.Output)
		}
		return measured, picture
	}
	resultReads := func(picture latentimage.EncodedImage) int {
		return store.reads[testcheck.Must(latentimage.PNGContent(picture))(t).Descriptor.ID]
	}

	first, firstPicture := served(t.Context(), request(42, sourceA))
	if work != (repeatWork{loads: 1, draws: 1}) || store.reads[sourceA] != 1 || opens != len(adapters) {
		t.Fatalf("the first call did %+v, read its source picture %d times and opened %d adapters; want one load and one draw, the picture read once and each adapter opened once",
			work, store.reads[sourceA], opens)
	}
	if bounds := testcheck.Must(png.DecodeConfig(bytes.NewReader(firstPicture.Data)))(t); bounds.Width != firstPicture.Width || firstPicture.Width <= 64 {
		t.Fatalf("the first picture is %dx%d; want the preset's canvas", firstPicture.Width, firstPicture.Height)
	}
	firstWork, firstOpens, firstHashed, firstResultReads := work, opens, hashed, resultReads(firstPicture)

	repeat, repeatPicture := served(t.Context(), request(42, sourceA))
	if repeat.Run != first.Run || !bytes.Equal(repeatPicture.Data, firstPicture.Data) {
		t.Errorf("the repeat gave run %s; want the first call's run %s and its pixel bytes", repeat.Run, first.Run)
	}
	if store.reads[sourceA] != 1 {
		t.Errorf("the repeat read its source picture's payload %d times more; want none", store.reads[sourceA]-1)
	}
	if opens != firstOpens || hashed != firstHashed {
		t.Errorf("the repeat opened %d adapters and hashed %d bytes more; want none", opens-firstOpens, hashed-firstHashed)
	}
	if work != firstWork {
		t.Errorf("the repeat did %+v after the first call's %+v; want no weights placed and nothing drawn", work, firstWork)
	}
	t.Logf("the repeat read the picture it returned %d times (the first call %d)", resultReads(firstPicture)-firstResultReads, firstResultReads)

	seeded, _ := served(t.Context(), request(43, sourceA))
	if seeded.Run == first.Run || work.draws != 2 || work.loads != 1 || opens != firstOpens || hashed != firstHashed {
		t.Errorf("another seed gave run %s (the first %s), did %+v, opened %d adapters more; want its own draw through the held adapters, no weights placed again",
			seeded.Run, first.Run, work, opens-firstOpens)
	}
	served(capabilityruntime.FreshRun(t.Context()), request(42, sourceA))
	if work.draws != 3 {
		t.Errorf("a fresh run drew %d times in all; want a draw of its own", work.draws)
	}
	other, otherPicture := served(t.Context(), request(42, sourceB))
	if other.Run == first.Run || store.reads[sourceB] != 1 || work.draws != 4 || bytes.Equal(otherPicture.Data, firstPicture.Data) {
		t.Errorf("another source picture gave run %s (the first %s), was read %d times, drew %d times in all; want it read once and drawn as a run of its own",
			other.Run, first.Run, store.reads[sourceB], work.draws)
	}
	if _, err := os.Stat(filepath.Join(root, "model_index.json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("the model's directory holds a model_index.json (%v); want none, so a request served reads none", err)
	}
	// The request's answer is kept under its preset's adapters, named by the profile's preset where no model_index.json is.
	if named := testcheck.Must(pictureFiles(t.Context(), store, root, execution, request(42, sourceA)))(t); !slices.Equal(named, adapters) {
		t.Errorf("the request named %v beside its model; want its preset's adapters, %v", named, adapters)
	}
	if repeat.Input.Descriptor.ID != first.Input.Descriptor.ID {
		t.Errorf("the repeat recorded its request as %s; want the first call's, %s", repeat.Input.Descriptor.ID, first.Input.Descriptor.ID)
	}
}
