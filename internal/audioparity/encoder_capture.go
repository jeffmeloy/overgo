package audioparity

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"

	"overgo/internal/artifact"
	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
)

type encoderCapture struct {
	Version       int            `json:"version"`
	Source        SourceIdentity `json:"source"`
	Dataset       DatasetBinding `json:"dataset"`
	CaptureSHA256 string         `json:"capture_sha256"`
	TensorsSHA256 string         `json:"tensors_sha256"`
	TensorCount   int            `json:"tensor_count"`
	Runtime       struct {
		Python       string            `json:"python"`
		Torch        string            `json:"torch"`
		Torchaudio   string            `json:"torchaudio"`
		Transformers string            `json:"transformers"`
		Tokenizers   string            `json:"tokenizers"`
		Device       string            `json:"device"`
		Attention    string            `json:"attention"`
		Files        map[string]string `json:"files"`
	} `json:"runtime"`
	Cases []struct {
		Fixture    string      `json:"fixture"`
		Audio      artifact.ID `json:"audio"`
		SampleRate uint32      `json:"sample_rate"`
		Samples    uint64      `json:"samples"`
		Frames     []int       `json:"frames"`
		Text       string      `json:"text"`
		WallNS     uint64      `json:"wall_ns"`
	} `json:"cases"`
}

func (c encoderCapture) validate(e Election) error {
	if c.Version != 1 || c.Source != e.ModelSource || !reflect.DeepEqual(c.Dataset, e.Dataset) || len(c.Cases) == 0 || len(c.Cases) != len(e.Observations) || c.TensorCount <= 0 ||
		c.Runtime.Device != "cpu" || c.Runtime.Attention != "eager" || c.Runtime.Python != e.Runtime.Python || c.Runtime.Torch != e.Runtime.Torch || c.Runtime.Transformers != e.Runtime.Transformers || len(c.Runtime.Files) == 0 {
		return errors.New("audio capture: source, corpus, runtime or case count differs from election")
	}
	for i, x := range c.Cases {
		want := e.Observations[i]
		if x.Fixture != want.Fixture || x.Audio != want.Audio || x.SampleRate != want.SampleRate || x.Samples != want.Samples || x.Text != want.Observed || x.WallNS == 0 || len(x.Frames) != 3 || x.Frames[0] != 1 || x.Frames[1] <= 0 || x.Frames[2] <= 0 {
			return fmt.Errorf("audio capture: case %d differs from pinned observation", i)
		}
	}
	return nil
}

// PublishEncoderCapture registers native CPU reference tensors by their exact
// file identity and stores the source-bound capture document and capture script.
// It verifies identities and source bindings, not Go/reference numerical parity.
// Tensors remain an external read-only file; no corpus or model is copied.
func PublishEncoderCapture(ctx context.Context, repository artifact.Repository, election Election, reportPath, sourcePath, tensorsPath string) (artifact.ID, error) {
	if ctx == nil || repository == nil {
		return artifact.ID{}, errors.New("audio capture: incomplete execution")
	}
	if err := ctx.Err(); err != nil {
		return artifact.ID{}, err
	}
	data, err := artifact.ReadContentFile(reportPath)
	if err != nil {
		return artifact.ID{}, err
	}
	var capture encoderCapture
	if err = strictjson.DecodeBytes(data, &capture); err != nil {
		return artifact.ID{}, err
	}
	if err = capture.validate(election); err != nil {
		return artifact.ID{}, err
	}
	golden, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/audio-encoder-golden/v1"), capture)
	if err != nil {
		return artifact.ID{}, err
	}
	return publishTensorCapture(ctx, repository, golden, election.ID, capture.CaptureSHA256, capture.TensorsSHA256, capture.TensorCount, sourcePath, tensorsPath)
}

func publishTensorCapture(ctx context.Context, repository artifact.Repository, golden artifact.Content, parent artifact.ID, sourceSHA, tensorSHA string, tensorCount int, sourcePath, tensorsPath string) (artifact.ID, error) {
	source, err := artifact.ReadContentFile(sourcePath)
	if err != nil {
		return artifact.ID{}, err
	}
	sourceID, err := artifact.IdentifyBytes(artifact.KindFile, source)
	if err != nil {
		return artifact.ID{}, err
	}
	if sourceID.DigestHex() != sourceSHA {
		return artifact.ID{}, errors.New("audio capture: capture script hash differs")
	}
	file, err := os.Open(tensorsPath)
	if err != nil {
		return artifact.ID{}, err
	}
	tensorID, size, err := artifact.Identify(artifact.KindFile, file)
	_ = file.Close()
	if err != nil {
		return artifact.ID{}, err
	}
	if tensorID.DigestHex() != tensorSHA {
		return artifact.ID{}, errors.New("audio capture: tensor file hash differs")
	}
	tensors, err := safetensors.OpenSource(filepath.Dir(tensorsPath))
	if err != nil {
		return artifact.ID{}, err
	}
	count := len(tensors.Tensors)
	single := tensors.ContainsOnlyShard(filepath.Base(tensorsPath))
	_ = tensors.Close()
	if !single {
		return artifact.ID{}, errors.New("audio capture: tensor file is not the complete source catalog")
	}
	if count != tensorCount {
		return artifact.ID{}, errors.New("audio capture: tensor count differs")
	}
	script, err := artifact.JSONContent(artifact.JSONContract(artifact.KindEvidence, "overgo/audio-oracle-capture-source/v1"), struct{ SHA256, Source string }{sourceSHA, string(source)})
	if err != nil {
		return artifact.ID{}, err
	}
	batch, err := artifact.NewDocumentBatch("audio/encoder-golden/"+golden.Descriptor.ID.DigestHex(), []artifact.Content{script, golden}, artifact.DependencyLineage(golden.Descriptor.ID, parent, script.Descriptor.ID, tensorID), nil)
	if err != nil {
		return artifact.ID{}, err
	}
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: tensorID, Size: size, MediaType: "application/vnd.safetensors"})
	location, err := artifact.CanonicalLocalLocation(tensorID, artifact.LocationFile, tensorsPath)
	if err != nil {
		return artifact.ID{}, err
	}
	locationID, err := artifact.JSONID(artifact.KindEvidence, location)
	if err != nil {
		return artifact.ID{}, err
	}
	// Availability belongs in the replay identity: the same capture can be
	// installed at another location without reusing a key for different bytes.
	batch.Key += "/" + locationID.DigestHex()
	batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
	if _, err = artifact.CommitBatch(ctx, repository, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return artifact.ID{}, err
	}
	return golden.Descriptor.ID, nil
}
