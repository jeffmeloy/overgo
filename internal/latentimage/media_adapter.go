package latentimage

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/safetensors"
	"overgo/internal/tensor"
)

// A media adapter (media-lora/unmerged) changes a model's weights by patches its file names: a projection by a
// low-rank pair, applied unmerged beside the weight as y = xW + s * scale * (x down) up at F32, so a request's
// strength s changes nothing resident; or a weight or bias by its full difference, added times s. Three formats name
// the patches:
//   - kohya: <name>.lora_down.weight [rank, in], <name>.lora_up.weight [out, rank] and an optional <name>.alpha;
//   - PEFT: <name>.lora_A.weight [rank, in] and <name>.lora_B.weight [out, rank];
//   - full differences: <name>.diff for a weight and <name>.diff_b for a bias;
//   - replacements: <name>.weight or <name>.bias named plainly, which ComfyUI sets in place of the model's whatever the
//     strength, and which may be wider (moss-control's video_embedder, which reads a control beside the latent).
//
// A pair's scale is alpha over its rank where the file holds an alpha tensor, and 1 where it holds none, as ComfyUI's
// calculate_weight reads it. A name may open with diffusion_model., which the graph's names do not. A joint block's
// fused layers.N.proj.qkv pair applies to the graph's q, k and v projections, each a third of its up rows, since the
// graph reads them apart.
const (
	adapterPrefix  = "diffusion_model."
	adapterDownKey = ".lora_down.weight"
	adapterUpKey   = ".lora_up.weight"
	adapterAKey    = ".lora_A.weight"
	adapterBKey    = ".lora_B.weight"
	adapterAlpha   = ".alpha"
	adapterDiff    = ".diff"
	adapterDiffB   = ".diff_b"
	// adapterThird names the graph input of a fused pair's third of its up rows: <up tensor>#<third>.
	adapterThird = "#"
)

// AppliedAdapter is one adapter a request applies: its name among the model's adapters
// (<model>/adapters/<name>.safetensors) and its strength, 1 where it states none.
type AppliedAdapter struct {
	Name     string   `json:"name"`
	Strength *float32 `json:"strength,omitempty"`
}

// strength is the strength the adapter is applied at, 1 where the request states none.
func (a AppliedAdapter) strength() float32 {
	if a.Strength == nil {
		return tensor.UnitScale
	}
	return *a.Strength
}

// mediaAdapter is an adapter file's patches by the graph weight each changes.
type mediaAdapter struct {
	path string
	// lowRank holds each projection's pair, by the graph's projection name (layers.0.proj.q).
	lowRank map[string]lowRankPatch
	// diffs holds each weight's or bias's full difference, by the graph's weight name (layers.0.norm1.weight), as the
	// file names the tensor.
	diffs map[string]string
	// replaced holds each weight or bias the adapter replaces outright, by the graph's weight name, as ComfyUI sets a
	// tensor a LoRA file names plainly: moss-control's video_embedder, widened to read its control beside the latent.
	replaced map[string]replacement
}

// replacement is a tensor an adapter replaces outright: the file's tensor and its shape, which may be wider than the
// weight it replaces.
type replacement struct {
	tensor string
	shape  []uint64
}

// lowRankPatch is one projection's pair: the file's down and up tensors, the rows of up it reads (all, or where split a
// third of a fused qkv), and its scale: the alpha the file holds for it over its rank in float64, as ComfyUI's
// calculate_weight divides them, or 1 where the file holds no alpha.
type lowRankPatch struct {
	down, up         string
	rank, rows, from uint64
	in, out          uint64
	split            bool
	scale            float64
}

// factor is what ComfyUI multiplies the pair's product by at a strength (comfy/weight_adapter/lora.py calculate_weight):
// the strength times the scale in float64, rounded to F32 as torch rounds a scalar it multiplies a BF16 tensor by.
func (patch lowRankPatch) factor(strength float32) float32 {
	return float32(float64(strength) * patch.scale)
}

// readMediaAdapter reads the patches of the adapter file at path, refusing a tensor no format above names, a pair
// missing a half, and halves whose shapes do not compose.
func readMediaAdapter(path string) (*mediaAdapter, error) {
	source, err := safetensors.OpenFile(path)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	adapter := &mediaAdapter{path: path, lowRank: map[string]lowRankPatch{}, diffs: map[string]string{}, replaced: map[string]replacement{}}
	downs, ups, alphas := map[string]string{}, map[string]string{}, map[string]float32{}
	for name, value := range source.Tensors {
		base := strings.TrimPrefix(name, adapterPrefix)
		switch {
		case strings.HasSuffix(base, adapterDownKey):
			downs[strings.TrimSuffix(base, adapterDownKey)] = name
		case strings.HasSuffix(base, adapterAKey):
			downs[strings.TrimSuffix(base, adapterAKey)] = name
		case strings.HasSuffix(base, adapterUpKey):
			ups[strings.TrimSuffix(base, adapterUpKey)] = name
		case strings.HasSuffix(base, adapterBKey):
			ups[strings.TrimSuffix(base, adapterBKey)] = name
		case strings.HasSuffix(base, adapterAlpha):
			values, err := safetensors.ReadF32(value)
			if err != nil || len(values) != 1 {
				return nil, fmt.Errorf("media adapter %s: alpha %s is no scalar", path, name)
			}
			alphas[strings.TrimSuffix(base, adapterAlpha)] = values[0]
		case strings.HasSuffix(base, adapterDiffB):
			adapter.diffs[strings.TrimSuffix(base, adapterDiffB)+".bias"] = name
		case strings.HasSuffix(base, adapterDiff):
			adapter.diffs[strings.TrimSuffix(base, adapterDiff)+".weight"] = name
		case strings.HasSuffix(base, ".weight") || strings.HasSuffix(base, ".bias"):
			adapter.replaced[base] = replacement{tensor: name, shape: slices.Clone(value.Shape)}
		default:
			return nil, fmt.Errorf("media adapter %s: tensor %s is no patch this reader applies", path, name)
		}
	}
	for base, down := range downs {
		up, paired := ups[base]
		if !paired {
			return nil, fmt.Errorf("media adapter %s: %s has a down half and no up half", path, base)
		}
		downShape, upShape := source.Tensors[down].Shape, source.Tensors[up].Shape
		if len(downShape) != 2 || len(upShape) != 2 || downShape[0] != upShape[1] || downShape[0] == 0 {
			return nil, fmt.Errorf("media adapter %s: %s's halves %v and %v do not compose", path, base, downShape, upShape)
		}
		patch := lowRankPatch{down: down, up: up, rank: downShape[0], in: downShape[1], out: upShape[0], rows: upShape[0], scale: 1}
		if alpha, held := alphas[base]; held {
			patch.scale = float64(alpha) / float64(patch.rank)
		}
		if !fusedJointQKV(base) {
			adapter.lowRank[base] = patch
			continue
		}
		if patch.out%tensor.TripleExtent != 0 {
			return nil, fmt.Errorf("media adapter %s: fused %s's %d rows are no three thirds", path, base, patch.out)
		}
		third := patch.out / tensor.TripleExtent
		for index, projection := range []string{"q", "k", "v"} {
			split := patch
			split.rows, split.from, split.out, split.split = third, uint64(index)*third, third, true
			adapter.lowRank[strings.TrimSuffix(base, "qkv")+projection] = split
		}
	}
	for base := range ups {
		if _, paired := downs[base]; !paired {
			return nil, fmt.Errorf("media adapter %s: %s has an up half and no down half", path, base)
		}
	}
	if len(adapter.lowRank) == 0 && len(adapter.diffs) == 0 && len(adapter.replaced) == 0 {
		return nil, fmt.Errorf("media adapter %s holds no patch", path)
	}
	return adapter, nil
}

// fusedJointQKV says the pair is a joint block's fused qkv, which the graph reads as three projections.
func fusedJointQKV(base string) bool {
	return strings.HasPrefix(base, "layers.") && strings.HasSuffix(base, ".proj.qkv")
}

// upInputName is the graph input of the patch's up rows: the file's tensor, or its third of a fused qkv.
func (patch lowRankPatch) upInputName() string {
	if !patch.split {
		return patch.up
	}
	return fmt.Sprintf("%s%s%d", patch.up, adapterThird, patch.from/patch.rows)
}

// adapterPayload is the F32 payload of an adapter's graph input: a file tensor whole, or rows of a fused up tensor. An
// input of an adapter after the first carries its place (adapterPlace), which the file's name lacks.
func adapterPayload(source *safetensors.Source, node *tensor.Tensor) (func() ([]byte, error), error) {
	stored, _, _ := strings.Cut(node.Name, adapterPlace)
	name, third, split := strings.Cut(stored, adapterThird)
	value, held := source.Tensors[name]
	if !held {
		return nil, fmt.Errorf("missing from the adapter %s", name)
	}
	if !split {
		plain := *node
		plain.Name = stored
		return residentPayload(source, &plain)
	}
	if len(value.Shape) != 2 || value.Shape[0]%tensor.TripleExtent != 0 {
		return nil, errors.New("a fused up tensor is no three thirds")
	}
	rows, rank := value.Shape[0]/tensor.TripleExtent, value.Shape[1]
	index := strings.Index("012", third)
	if index < 0 || len(third) != 1 || node.Shape.Rank != 2 || node.Shape.Dims[0] != rank || node.Shape.Dims[1] != rows {
		return nil, errors.New("a third differs from the graph")
	}
	return func() ([]byte, error) {
		reader, err := safetensors.F32Reader(value)
		if err != nil {
			return nil, err
		}
		width := rows * rank * 4
		if _, err := io.CopyN(io.Discard, reader, int64(uint64(index)*width)); err != nil {
			return nil, err
		}
		payload := make([]byte, width)
		_, err = io.ReadFull(reader, payload)
		return payload, err
	}, nil
}

// AdapterNames are the adapters the model in directory holds (<directory>/adapters/<name>.safetensors), in order, for a
// page to offer; none where it holds none.
func AdapterNames(directory string) ([]string, error) {
	matches, err := filepath.Glob(filepath.Join(directory, "adapters", "*.safetensors"))
	names := make([]string, 0, len(matches))
	for _, match := range matches {
		names = append(names, strings.TrimSuffix(filepath.Base(match), ".safetensors"))
	}
	return names, err
}

// adapterPath is the file of the adapter a request names among its model's: <model>/adapters/<name>.safetensors.
func adapterPath(modelRoot, name string) (string, error) {
	if name == "" || name != filepath.Base(name) || strings.ContainsAny(name, `/\`) {
		return "", fmt.Errorf("latent image: adapter %q is no file name", name)
	}
	return filepath.Join(modelRoot, "adapters", name+".safetensors"), nil
}

// modelAdapter is the adapter a request names among its model's (adapterPath).
func modelAdapter(modelRoot, name string) (*mediaAdapter, error) {
	path, err := adapterPath(modelRoot, name)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("latent image: the model holds no adapter %q: %w", name, err)
	}
	return readMediaAdapter(path)
}
