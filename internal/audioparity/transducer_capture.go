package audioparity

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/strictjson"
)

type transducerCapture struct {
	Version              int               `json:"version"`
	Model                artifact.ID       `json:"model"`
	Revision             string            `json:"revision"`
	License              string            `json:"license"`
	Files                map[string]string `json:"files"`
	Fixture              string            `json:"fixture"`
	Reference            string            `json:"reference"`
	AudioSHA256          string            `json:"audio_sha256"`
	Samples              int               `json:"samples"`
	SampleRate           int               `json:"sample_rate"`
	PromptID             int               `json:"prompt_id"`
	Lookahead            int               `json:"lookahead"`
	FirstChunkFrames     int               `json:"first_chunk_frames"`
	FollowingChunkFrames int               `json:"following_chunk_frames"`
	Dataset              struct {
		Path   string `json:"path"`
		SHA256 string `json:"sha256"`
		Row    int    `json:"row"`
	} `json:"dataset"`
	Runtime struct {
		Python       string            `json:"python"`
		Torch        string            `json:"torch"`
		Transformers string            `json:"transformers"`
		Librosa      string            `json:"librosa"`
		Tokenizers   string            `json:"tokenizers"`
		Threads      int               `json:"threads"`
		Device       string            `json:"device"`
		Attention    string            `json:"attention"`
		Sources      map[string]string `json:"sources"`
	} `json:"runtime"`
	CaptureSHA256 string `json:"capture_sha256"`
	TensorsSHA256 string `json:"tensors_sha256"`
	Captures      map[string]struct {
		Chunks []struct {
			Frames int `json:"frames"`
		} `json:"chunks"`
		Steps  []int  `json:"steps"`
		WallNS uint64 `json:"wall_ns"`
		Text   string `json:"text"`
	} `json:"captures"`
	TensorCount int            `json:"tensor_count"`
	Nonfinite   map[string]int `json:"nonfinite"`
}

var transducerCaptureContract = artifact.JSONContract(artifact.KindEvidence, "overgo/audio-transducer-golden/v1")

func (c transducerCapture) validate() error {
	if c.Version != 1 || !c.Model.Valid() || c.Model.Kind() != artifact.KindModel || !gitauthority.ValidObjectID(c.Revision) || strings.ToLower(c.Revision) != c.Revision || c.License == "" || c.Fixture == "" || c.Reference == "" ||
		c.Samples <= 0 || c.SampleRate <= 0 || c.PromptID < 0 || c.Lookahead < 0 || c.FirstChunkFrames <= 0 || c.FollowingChunkFrames <= 0 || c.Dataset.Row < 0 || c.Dataset.Path == "" ||
		c.Runtime.Device != "cpu" || !slices.Contains([]string{"eager", "sdpa"}, c.Runtime.Attention) || c.Runtime.Threads <= 0 || c.Runtime.Python == "" || c.Runtime.Torch == "" || c.Runtime.Transformers == "" || c.Runtime.Librosa == "" || c.Runtime.Tokenizers == "" ||
		c.TensorCount <= 0 || len(c.Nonfinite) != 0 || len(c.Captures) != 2 || len(c.Files) == 0 || len(c.Runtime.Sources) == 0 {
		return errors.New("transducer capture: incomplete source, runtime, corpus or finite trace declaration")
	}
	for _, paths := range []map[string]string{c.Files, c.Runtime.Sources, {c.Dataset.Path: c.Dataset.SHA256}} {
		for name := range paths {
			if err := validateRelativePath(name); err != nil || name == "." || name == ".." || strings.Contains(name, ":") {
				return fmt.Errorf("transducer capture: invalid member path %q", name)
			}
		}
	}
	for _, digest := range append([]string{c.AudioSHA256, c.Dataset.SHA256, c.CaptureSHA256, c.TensorsSHA256}, mapDigests(c.Files, c.Runtime.Sources)...) {
		if _, err := artifact.ParseID("file:sha256:" + digest); err != nil {
			return err
		}
	}
	for _, mode := range []string{"offline", "streaming"} {
		trace, ok := c.Captures[mode]
		if !ok || len(trace.Chunks) == 0 || len(trace.Steps) == 0 || trace.WallNS == 0 || trace.Text == "" {
			return fmt.Errorf("transducer capture: incomplete %s trace", mode)
		}
		for _, chunk := range trace.Chunks {
			if chunk.Frames <= 0 {
				return errors.New("transducer capture: invalid chunk")
			}
		}
		for _, token := range trace.Steps {
			if token < 0 {
				return errors.New("transducer capture: negative token")
			}
		}
	}
	return nil
}

func mapDigests(groups ...map[string]string) []string {
	var values []string
	for _, m := range groups {
		values = slices.AppendSeq(values, maps.Values(m))
	}
	return values
}

// PublishTransducerCapture uses the existing external-tensor publication owner.
// It admits source-bound native traces, not a Go parity or quality claim.
func PublishTransducerCapture(ctx context.Context, repository artifact.Repository, reportPath, sourcePath, tensorsPath string) (artifact.ID, error) {
	if ctx == nil || repository == nil {
		return artifact.ID{}, errors.New("transducer capture: incomplete invocation")
	}
	if err := ctx.Err(); err != nil {
		return artifact.ID{}, err
	}
	data, err := artifact.ReadContentFile(reportPath)
	if err != nil {
		return artifact.ID{}, err
	}
	var capture transducerCapture
	if err := strictjson.DecodeBytes(data, &capture); err != nil {
		return artifact.ID{}, err
	}
	if err := capture.validate(); err != nil {
		return artifact.ID{}, err
	}
	manifest, found, err := repository.Manifest(ctx, capture.Model)
	if err != nil || !found {
		return artifact.ID{}, errors.Join(errors.New("transducer capture: registered model absent"), err)
	}
	if len(manifest.Components) == 0 {
		return artifact.ID{}, errors.New("transducer capture: registered model has no components")
	}
	fileDigests := mapDigests(capture.Files)
	for _, component := range manifest.Components {
		if !slices.Contains(fileDigests, component.Artifact.DigestHex()) {
			return artifact.ID{}, errors.New("transducer capture: model component identity differs")
		}
	}
	golden, err := artifact.JSONContent(transducerCaptureContract, capture)
	if err != nil {
		return artifact.ID{}, err
	}
	return publishTensorCapture(ctx, repository, golden, capture.Model, capture.CaptureSHA256, capture.TensorsSHA256, capture.TensorCount, sourcePath, tensorsPath)
}
