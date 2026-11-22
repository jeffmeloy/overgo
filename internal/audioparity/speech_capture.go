package audioparity

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/fsatomic"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/safetensors"
)

// SpeechCaptureCommand is the command that remakes a speech capture from overgo's own speech path, which every capture
// it made names as its maker (cmd/speech-capture).
const SpeechCaptureCommand = "cmd/speech-capture"

// SpeechCaptureRemake is where a speech capture is remade from overgo's own speech path: the store that records the
// capture's inputs and receives the remade capture, the datasets root its corpus shards lie under, the references
// directory its tensors are written beneath, and the full commit whose speech path computes them, one at which the
// speech acceptances passed against the captures being remade.
type SpeechCaptureRemake struct {
	Store      artifact.Repository
	Datasets   string
	References string
	Commit     string
}

// captureMemoryBytes is the byte ceiling a remade capture's model and frontend run under, the ceiling the acceptances
// that read the capture run the same path under.
const captureMemoryBytes = 4 << 30

// captureShard is the one file a capture's tensors are written to, alone in their directory.
const captureShard = "tensors.safetensors"

// captureProducer names overgo as the maker of a capture's tensors: the commit whose speech path computed them and the
// command that ran it. A capture the reference software made names its Python and Torch runtime instead and leaves this
// empty.
type captureProducer struct {
	Commit  string `json:"commit"`
	Command string `json:"command"`
}

// producer is the maker the remake records: overgo at its commit through SpeechCaptureCommand.
func (r SpeechCaptureRemake) producer() captureProducer {
	return captureProducer{Commit: r.Commit, Command: SpeechCaptureCommand}
}

// ready refuses a remake before it computes anything: one without a context or with one already ended, without a store
// or a references directory, or whose commit is no maker a capture can name.
func (r SpeechCaptureRemake) ready(ctx context.Context) error {
	if ctx == nil || r.Store == nil || r.References == "" {
		return errors.New("speech capture: a remake needs a context, a store and a references directory")
	}
	return errors.Join(ctx.Err(), r.producer().validate())
}

// directory is where the remake writes the tensors of the capture named label.
func (r SpeechCaptureRemake) directory(label string) (string, error) {
	if err := fsatomic.PathComponent(label); err != nil {
		return "", fmt.Errorf("speech capture: %w", err)
	}
	return filepath.Join(r.References, "speech-captures", label), nil
}

// validate refuses a producer that is not overgo at a full lowercase commit through SpeechCaptureCommand.
func (p captureProducer) validate() error {
	if !gitauthority.ValidObjectID(p.Commit) || strings.ToLower(p.Commit) != p.Commit || p.Command != SpeechCaptureCommand {
		return fmt.Errorf("speech capture: its maker is not overgo at a full commit through %s", SpeechCaptureCommand)
	}
	return nil
}

// captureMaker holds a capture to one maker: overgo at a full commit through SpeechCaptureCommand (producer), or the
// reference runtime it names (runtime), complete where reference says so. A capture naming both, or neither, has no
// maker a test can read, so a test tells the two apart by producer alone.
func captureMaker(producer captureProducer, runtime any, reference bool) error {
	if producer == (captureProducer{}) {
		if !reference {
			return errors.New("speech capture: its reference runtime is incomplete and it names no other maker")
		}
		return nil
	}
	if !reflect.ValueOf(runtime).IsZero() {
		return errors.New("speech capture: it names both overgo and a reference runtime as its maker")
	}
	return producer.validate()
}

// recorded gives a capture made by overgo what its record names beside the producer: the producer's own bytes, which it
// records as its capture source where the reference software recorded its capture script, their digest, and the
// digest and tensor count of the tensors file at path.
func (p captureProducer) recorded(path string) (source []byte, sourceSHA, tensorSHA string, count int, err error) {
	source, err = json.Marshal(p)
	if err != nil {
		return nil, "", "", 0, err
	}
	sourceID, err := artifact.IdentifyBytes(artifact.KindFile, source)
	if err != nil {
		return nil, "", "", 0, err
	}
	tensorID, _, count, err := captureTensorFile(path)
	if err != nil {
		return nil, "", "", 0, err
	}
	return source, sourceID.DigestHex(), tensorID.DigestHex(), count, nil
}

// captureTensorFile is the identity, size and tensor count of the capture tensors file at path, which must be the only
// shard its directory holds.
func captureTensorFile(path string) (artifact.ID, uint64, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return artifact.ID{}, 0, 0, err
	}
	id, size, err := artifact.Identify(artifact.KindFile, file)
	if err = errors.Join(err, file.Close()); err != nil {
		return artifact.ID{}, 0, 0, err
	}
	tensors, err := safetensors.OpenSource(filepath.Dir(path))
	if err != nil {
		return artifact.ID{}, 0, 0, err
	}
	count, single := len(tensors.Tensors), tensors.ContainsOnlyShard(filepath.Base(path))
	if err = tensors.Close(); err != nil {
		return artifact.ID{}, 0, 0, err
	}
	if !single {
		return artifact.ID{}, 0, 0, errors.New("audio capture: tensor file is not the complete source catalog")
	}
	return id, size, count, nil
}

// captureTensors are a capture's tensors, in the order they were traced, each stored with the dtype and shape an
// acceptance reads it with: F32 values, I64 identifiers and BOOL masks. The first refused tensor is its error, which
// writing it returns.
type captureTensors struct {
	specs    []safetensors.StreamingTensorSpec
	payloads map[string][]byte
	err      error
}

// add keeps payload as the tensor name of dtype and shape, refusing a name kept already and a shape whose element count
// differs from the elements payload holds.
func (c *captureTensors) add(name, dtype string, elements int, payload []byte, shape []int) {
	if c.err != nil {
		return
	}
	if _, kept := c.payloads[name]; kept || name == "" {
		c.err = fmt.Errorf("speech capture: tensor %q is empty or kept twice", name)
		return
	}
	dimensions, count := make([]uint64, len(shape)), 1
	for index, dimension := range shape {
		if dimension <= 0 {
			c.err = fmt.Errorf("speech capture: tensor %q has a dimension %d", name, dimension)
			return
		}
		dimensions[index], count = uint64(dimension), count*dimension
	}
	if count != elements {
		c.err = fmt.Errorf("speech capture: tensor %q has %d elements, not the %d its shape %v holds", name, elements, count, shape)
		return
	}
	if c.payloads == nil {
		c.payloads = map[string][]byte{}
	}
	c.specs = append(c.specs, safetensors.StreamingTensorSpec{Name: name, DType: dtype, Shape: dimensions})
	c.payloads[name] = payload
}

// f32 keeps a copy of values as an F32 tensor of shape.
func (c *captureTensors) f32(name string, values []float32, shape ...int) {
	payload := make([]byte, 4*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint32(payload[4*index:], math.Float32bits(value))
	}
	c.add(name, "F32", len(values), payload, shape)
}

// ids keeps values as an I64 tensor of shape, refusing a negative identifier.
func (c *captureTensors) ids(name string, values []int, shape ...int) {
	payload := make([]byte, 8*len(values))
	for index, value := range values {
		if value < 0 && c.err == nil {
			c.err = fmt.Errorf("speech capture: tensor %q holds the negative identifier %d", name, value)
		}
		binary.LittleEndian.PutUint64(payload[8*index:], uint64(value))
	}
	c.add(name, "I64", len(values), payload, shape)
}

// mask keeps values as a BOOL tensor of shape, one byte each, one where the value holds.
func (c *captureTensors) mask(name string, values []bool, shape ...int) {
	payload := make([]byte, len(values))
	for index, value := range values {
		if value {
			payload[index] = 1
		}
	}
	c.add(name, "BOOL", len(values), payload, shape)
}

// write writes the tensors as the one shard of directory and returns its path. A directory already there is kept where
// it holds exactly these tensors, so a remake run again over unchanged speech code records the same file, and refused
// where it holds anything else.
func (c *captureTensors) write(directory string) (string, error) {
	if c.err != nil {
		return "", c.err
	}
	if len(c.specs) == 0 {
		return "", errors.New("speech capture: no tensors to write")
	}
	path := filepath.Join(directory, captureShard)
	if _, err := os.Stat(directory); err == nil {
		return path, c.held(directory)
	} else if !errors.Is(err, fs.ErrNotExist) {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(directory), fsatomic.DirectoryMode); err != nil {
		return "", err
	}
	writer, err := safetensors.NewStreamingWriter(directory, safetensors.StreamingPlan{
		Shards: []safetensors.StreamingShardSpec{{Name: captureShard, Tensors: c.specs}},
	})
	if err != nil {
		return "", err
	}
	for _, spec := range c.specs {
		if writer.Completed(spec.Name) {
			continue
		}
		payload := c.payloads[spec.Name]
		if err := writer.WriteTensor(spec.Name, uint64(len(payload)), bytes.NewReader(payload)); err != nil {
			return "", err
		}
	}
	return path, writer.Finalize()
}

// held refuses a directory that does not hold exactly these tensors, by name, dtype, shape and bytes, in its one shard.
func (c *captureTensors) held(directory string) error {
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return err
	}
	defer source.Close()
	if !source.ContainsOnlyShard(captureShard) || len(source.Tensors) != len(c.specs) {
		return fmt.Errorf("speech capture: %s holds other tensors than the remade capture's; move it aside to write the capture there", directory)
	}
	for _, spec := range c.specs {
		tensor, found := source.Tensors[spec.Name]
		if !found || tensor.DType != spec.DType || !slices.Equal(tensor.Shape, spec.Shape) {
			return fmt.Errorf("speech capture: %s holds tensor %q otherwise than the remade capture", directory, spec.Name)
		}
		data, err := io.ReadAll(tensor.Reader())
		if err != nil {
			return err
		}
		if !bytes.Equal(data, c.payloads[spec.Name]) {
			return fmt.Errorf("speech capture: %s holds other values of tensor %q than the remade capture; move it aside to write the capture there", directory, spec.Name)
		}
	}
	return nil
}

// listedCapture is the capture list at path and where in it id stands, refusing a list that does not name id.
func listedCapture(path string, id artifact.ID) ([]artifact.ID, int, error) {
	var listed []artifact.ID
	if err := jsonfile.DecodeStrict(path, &listed); err != nil {
		return nil, 0, err
	}
	index := slices.Index(listed, id)
	if index < 0 {
		return nil, 0, fmt.Errorf("speech capture: %s lists no capture %s", path, id)
	}
	return listed, index, nil
}
