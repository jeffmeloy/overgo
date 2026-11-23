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

// standInPictures is the picture capability's serving (pictureExecutor) of the MOSS fixture model on a store holding
// its profile and two stored source PNGs, through a session standing in for MOSS's (repeatSession): the session holds
// the ultra-image-to-image-turbo preset's two adapters, installed beside the model, through its held files, each open
// counted with the bytes it hashes, and draws a picture its seed and source picture shade. The model's directory holds
// no model_index.json once its profile is stored.
type standInPictures struct {
	t                *testing.T
	root             string
	store            countedStore
	execution        modelrecipe.CapabilityEvidenceSelection
	adapters         []string
	sourceA, sourceB artifact.ID
	opens            int
	hashed           int64
	work             repeatWork
	serve            capabilityruntime.Executor
}

// newStandInPictures serves the fixture model's pictures, its prompt mode 2's enhancer chat opened through opened.
func newStandInPictures(t *testing.T, opened func(artifact.Repository) latentimage.EnhancerOpener) *standInPictures {
	t.Helper()
	pictures := &standInPictures{t: t}
	pictures.root = testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	source := testcheck.Must(Catalog[recipe.TaskImageGen].Resolve(t.Context(), pictures.root))(t)
	definition, facts, err := source.Define(source.Inventory.Manifest.ID)
	testcheck.NoError(t, err)
	program := testcheck.Must(modelrecipe.CompileCapability(definition))(t)
	pictures.store = countedStore{Store: overgodbtest.Open(t, t.TempDir()), reads: map[artifact.ID]int{}}
	batch := testcheck.Must(source.Inventory.Batch(t.Name()))(t)
	batch.Contents = append(batch.Contents, facts...)
	batch.Manifests = append(batch.Manifests, source.Manifests...)
	testcheck.NoError(t, modelrecipe.PublishFacts(t.Context(), pictures.store, batch))
	pictures.execution = testcheck.Must(modelrecipe.CompileCandidateExecution(t.Context(), pictures.store, program))(t)
	testcheck.NoError(t, os.Remove(filepath.Join(pictures.root, "model_index.json")))

	// The preset's adapters, installed beside the model.
	for _, name := range []string{"8-steps-dmd2-lora", "moss_image_sft"} {
		adapter := filepath.Join(pictures.root, "adapters", name+".safetensors")
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(adapter), 0o755))
		testcheck.NoError(t, os.WriteFile(adapter, []byte(name+"'s weights"), 0o600))
		pictures.adapters = append(pictures.adapters, adapter)
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
		testcheck.Must(artifact.Publish(t.Context(), pictures.store, artifact.Batch{Key: fmt.Sprintf("fixture/source-%d", shade), Artifacts: []artifact.Descriptor{content.Descriptor}, Contents: []artifact.Content{content}}))(t)
		return content.Descriptor.ID
	}
	pictures.sourceA, pictures.sourceB = storedPNG(90), storedPNG(200)
	sources := map[string]artifact.ID{}
	for _, id := range []artifact.ID{pictures.sourceA, pictures.sourceB} {
		content := testcheck.Must(func() (artifact.Content, error) {
			content, _, err := artifact.ReadContent(t.Context(), pictures.store.Store, id)
			return content, err
		}())(t)
		sources[base64.StdEncoding.EncodeToString(content.Data)] = id
	}

	// The session's held files, each open counted with the bytes it hashes.
	held := &heldfile.Set{Open: func(path string) (*os.File, error) {
		file, err := holdForCheck(path)
		if err == nil {
			info, statErr := file.Stat()
			if statErr != nil {
				return nil, errors.Join(statErr, file.Close())
			}
			pictures.opens, pictures.hashed = pictures.opens+1, pictures.hashed+info.Size()
		}
		return file, err
	}}
	director := testcheck.Must(capabilityruntime.NewModelSessionDirector[latentimage.Request, *repeatSession, latentimage.EncodedImage](
		"image-gen", imageDevice, imageSessionCapacity, latentimage.Admit,
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, _ latentimage.Request) (*repeatSession, error) {
			release, err := held.Own(path)
			if err != nil {
				return nil, err
			}
			pictures.work.loads++
			return &repeatSession{release: release}, nil
		},
		func(context.Context, *repeatSession, latentimage.Request) error {
			pictures.work.resets++
			return nil
		},
		func(runtime *workflowruntime.Runtime, modelID artifact.ID, _ *repeatSession) error {
			return workflowruntime.RegisterContextPipeline(runtime, modelID,
				modelrecipe.ModuleLatentImagePrepare, func(_ context.Context, request latentimage.Request) (latentimage.Request, error) {
					// The session takes a source picture named by its stored identity as its stored bytes, read where no answer
					// was stored, and reads its adapters as the picture session reads them.
					if _, named := artifact.ParseID(request.Image); named == nil {
						return latentimage.Request{}, fmt.Errorf("the session took the source picture as %.40q, not its stored bytes", request.Image)
					}
					for _, adapter := range pictures.adapters {
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
					pictures.work.draws++
					shade := float32(request.Seed%7) / 7
					if sources[request.Image] == pictures.sourceB {
						shade += 0.1
					}
					picture := func(shade float32) (latentimage.EncodedImage, error) {
						pixels := make([]float32, 3*request.Width*request.Height)
						for index := range pixels {
							pixels[index] = shade
						}
						return latentimage.EncodePlanarPNG(pixels, request.Height, request.Width)
					}
					// A sharpened picture is its own, and comes with the picture it drew, as MOSS's does (EncodedImage.Drawn).
					// It reports what it drew from as MOSS's does (EncodedImage.Prompt): the prompt and negative it was given.
					if request.Sharpen == nil {
						finished, err := picture(shade)
						finished.Prompt, finished.NegativePrompt = request.Prompt, request.NegativePrompt
						return finished, err
					}
					drawn, err := picture(shade)
					if err != nil {
						return latentimage.EncodedImage{}, err
					}
					finished, err := picture(shade + 0.05)
					finished.Drawn, finished.Sharpened = drawn.Data, true
					finished.Prompt, finished.NegativePrompt = request.Prompt, request.NegativePrompt
					return finished, err
				},
				latentimage.PNGContent)
		}))(t)
	t.Cleanup(testutil.Closer(t, context.Background(), director))
	pictures.serve = pictureExecutor(director, nil, held, opened)
	return pictures
}

// served serves raw and gives what it measured and the picture it gave.
func (pictures *standInPictures) served(ctx context.Context, raw string) (capabilityruntime.Measured, latentimage.EncodedImage) {
	pictures.t.Helper()
	measured, ok := testcheck.Must(pictures.serve(ctx, pictures.store, pictures.root, pictures.execution, raw))(pictures.t).(capabilityruntime.Measured)
	if !ok {
		pictures.t.Fatal("a served picture came with no measure")
	}
	picture, ok := measured.Output.(latentimage.EncodedImage)
	if !ok {
		pictures.t.Fatalf("a served picture is %T", measured.Output)
	}
	return measured, picture
}

// TestServedMOSSPictureReusesItsCompletedDraw holds moss-served-repeat: the same admitted, fixed-seed MOSS request sent
// twice through the picture capability's serving (pictureExecutor), on a store holding the model's profile and a
// source PNG the request names by its stored identity, the ultra-image-to-image-turbo preset naming its canvas and its
// two adapters, held by a live session (standInPictures). The first call reads the source picture once and draws
// once; the second gives the same recorded run and pixel bytes, reads no byte of the source picture, opens and hashes
// no adapter again, and places no weights and draws nothing. The store's reads of the returned picture are counted
// apart. Another seed draws again through the held adapters, a fresh run draws again, and the same request with another
// source picture reads that picture and is a run of its own. The model's directory holds no model_index.json once its
// profile is stored, so a request served reads none: its preset is read from the profile. A repeat had read and encoded
// its source picture and read model_index.json before its stored answer was sought.
func TestServedMOSSPictureReusesItsCompletedDraw(t *testing.T) {
	pictures := newStandInPictures(t, enhancerOpener)
	store, sourceA, sourceB := pictures.store, pictures.sourceA, pictures.sourceB
	request := func(seed int, picture artifact.ID) string {
		return fmt.Sprintf(`{"prompt":"a lighthouse at dusk","preset":"ultra-image-to-image-turbo","seed":%d,"image":%q}`, seed, picture.String())
	}
	resultReads := func(picture latentimage.EncodedImage) int {
		return store.reads[testcheck.Must(latentimage.PNGContent(picture))(t).Descriptor.ID]
	}

	first, firstPicture := pictures.served(t.Context(), request(42, sourceA))
	if pictures.work != (repeatWork{loads: 1, draws: 1}) || store.reads[sourceA] != 1 || pictures.opens != len(pictures.adapters) {
		t.Fatalf("the first call did %+v, read its source picture %d times and opened %d adapters; want one load and one draw, the picture read once and each adapter opened once",
			pictures.work, store.reads[sourceA], pictures.opens)
	}
	if bounds := testcheck.Must(png.DecodeConfig(bytes.NewReader(firstPicture.Data)))(t); bounds.Width != firstPicture.Width || firstPicture.Width <= 64 {
		t.Fatalf("the first picture is %dx%d; want the preset's canvas", firstPicture.Width, firstPicture.Height)
	}
	firstWork, firstOpens, firstHashed, firstResultReads := pictures.work, pictures.opens, pictures.hashed, resultReads(firstPicture)

	repeat, repeatPicture := pictures.served(t.Context(), request(42, sourceA))
	if repeat.Run != first.Run || !bytes.Equal(repeatPicture.Data, firstPicture.Data) {
		t.Errorf("the repeat gave run %s; want the first call's run %s and its pixel bytes", repeat.Run, first.Run)
	}
	if store.reads[sourceA] != 1 {
		t.Errorf("the repeat read its source picture's payload %d times more; want none", store.reads[sourceA]-1)
	}
	if pictures.opens != firstOpens || pictures.hashed != firstHashed {
		t.Errorf("the repeat opened %d adapters and hashed %d bytes more; want none", pictures.opens-firstOpens, pictures.hashed-firstHashed)
	}
	if pictures.work != firstWork {
		t.Errorf("the repeat did %+v after the first call's %+v; want no weights placed and nothing drawn", pictures.work, firstWork)
	}
	t.Logf("the repeat read the picture it returned %d times (the first call %d)", resultReads(firstPicture)-firstResultReads, firstResultReads)

	seeded, _ := pictures.served(t.Context(), request(43, sourceA))
	if seeded.Run == first.Run || pictures.work.draws != 2 || pictures.work.loads != 1 || pictures.opens != firstOpens || pictures.hashed != firstHashed {
		t.Errorf("another seed gave run %s (the first %s), did %+v, opened %d adapters more; want its own draw through the held adapters, no weights placed again",
			seeded.Run, first.Run, pictures.work, pictures.opens-firstOpens)
	}
	pictures.served(capabilityruntime.FreshRun(t.Context()), request(42, sourceA))
	if pictures.work.draws != 3 {
		t.Errorf("a fresh run drew %d times in all; want a draw of its own", pictures.work.draws)
	}
	other, otherPicture := pictures.served(t.Context(), request(42, sourceB))
	if other.Run == first.Run || store.reads[sourceB] != 1 || pictures.work.draws != 4 || bytes.Equal(otherPicture.Data, firstPicture.Data) {
		t.Errorf("another source picture gave run %s (the first %s), was read %d times, drew %d times in all; want it read once and drawn as a run of its own",
			other.Run, first.Run, store.reads[sourceB], pictures.work.draws)
	}
	if _, err := os.Stat(filepath.Join(pictures.root, "model_index.json")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("the model's directory holds a model_index.json (%v); want none, so a request served reads none", err)
	}
	// The request's answer is kept under its preset's adapters, named by the profile's preset where no model_index.json is.
	if named := testcheck.Must(pictureFiles(t.Context(), store, pictures.root, pictures.execution, request(42, sourceA)))(t); !slices.Equal(named, pictures.adapters) {
		t.Errorf("the request named %v beside its model; want its preset's adapters, %v", named, pictures.adapters)
	}
	if repeat.Input.Descriptor.ID != first.Input.Descriptor.ID {
		t.Errorf("the repeat recorded its request as %s; want the first call's, %s", repeat.Input.Descriptor.ID, first.Input.Descriptor.ID)
	}
}

// TestASharpenedServedPictureComesWithThePictureItDrew holds moss-sharpened-pair: a sharpened picture served through
// the picture capability (pictureExecutor, standInPictures) comes with the picture it drew before its sharpening, as
// the MOSS Ultra workflow saves both, and so does the same request its stored run answers, which holds the sharpened
// picture alone: the picture drawn is stored as the one the sharpened picture was derived from (keptDrawn). An
// unsharpened request's picture comes with none.
func TestASharpenedServedPictureComesWithThePictureItDrew(t *testing.T) {
	pictures := newStandInPictures(t, enhancerOpener)
	const sharpened = `{"prompt":"a lighthouse","seed":3,"width":256,"height":144,"sharpen":{"radius":1,"sigma":0.6,"alpha":0.7}}`
	first, drawn := pictures.served(t.Context(), sharpened)
	if len(drawn.Drawn) == 0 || bytes.Equal(drawn.Drawn, drawn.Data) {
		t.Fatalf("the sharpened picture came with %d bytes as the picture it drew; want the picture before its sharpening", len(drawn.Drawn))
	}
	replayed, again := pictures.served(t.Context(), sharpened)
	if replayed.Run != first.Run || pictures.work.draws != 1 || !bytes.Equal(again.Drawn, drawn.Drawn) || !bytes.Equal(again.Data, drawn.Data) {
		t.Errorf("the stored run's answer (run %s, the first %s, %d draws) came with %d bytes as the picture drawn; want the first's, %d bytes, drawn once",
			replayed.Run, first.Run, pictures.work.draws, len(again.Drawn), len(drawn.Drawn))
	}
	if _, plain := pictures.served(t.Context(), `{"prompt":"a lighthouse","seed":3,"width":256,"height":144}`); len(plain.Drawn) != 0 {
		t.Error("an unsharpened picture came with a picture drawn before it")
	}
}
