package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const llama4ProjectorType = "llama4"

type Llama4VisionSpec struct {
	visionBackboneSpec
	OutputHidden        int
	AdapterIntermediate int
	AdapterHidden       int
	MergeSize           int
	MaxGridSide         int
	Activation          visionActivation
	PreLayerNorm        bool
	PostLayerNorm       bool
	FusedQKV            []bool
}

type Llama4VisionTile gridImage

type Llama4VisionInput struct {
	Tiles []Llama4VisionTile
	GridH int
	GridW int
}

type Llama4VisionOutput struct {
	Embeddings reference.Value
	TileCount  int
	GridH      int
	GridW      int
}

type Llama4VisionRunner struct {
	projectorResources
	spec      Llama4VisionSpec
	attention visionAttentionPlan
}

func (r *Llama4VisionRunner) Spec() Llama4VisionSpec {
	if r == nil {
		return Llama4VisionSpec{}
	}
	return r.spec
}

func ReadLlama4VisionSpec(file *gguf.File) (Llama4VisionSpec, error) {
	spec := Llama4VisionSpec{}
	if err := readRotaryVisionBackbone(file, llama4ProjectorType, &spec.OutputHidden, &spec.visionBackboneSpec); err != nil {
		return Llama4VisionSpec{}, err
	}
	merge, err := metadataUint32(file, visionProjectorScaleKey)
	if err != nil {
		return Llama4VisionSpec{}, err
	}
	maxGridSide, err := metadataUint32(file, visionMaxGridSideKey)
	if err != nil {
		return Llama4VisionSpec{}, err
	}
	mlp1, ok := file.Tensor("mm.model.mlp.1.weight")
	adapterIntermediate, mlp1OK := matrixRowsInt(mlp1, ok)
	if !mlp1OK {
		return Llama4VisionSpec{}, errors.New("projector: Llama-4 first adapter tensor is unavailable or invalid")
	}
	mlp2, ok := file.Tensor("mm.model.mlp.2.weight")
	adapterHidden, mlp2OK := matrixRowsInt(mlp2, ok)
	if !mlp2OK {
		return Llama4VisionSpec{}, errors.New("projector: Llama-4 second adapter tensor is unavailable or invalid")
	}
	spec.AdapterIntermediate = adapterIntermediate
	spec.AdapterHidden = adapterHidden
	spec.MergeSize = int(merge)
	spec.MaxGridSide = int(maxGridSide)
	spec.PreLayerNorm = hasTensor(file, visionPreNormWeightTensor)
	spec.PostLayerNorm = hasTensor(file, visionPostNormWeightTensor)
	spec.FusedQKV = make([]bool, spec.Layers)
	for layer := range spec.FusedQKV {
		spec.FusedQKV[layer] = hasTensor(file, fmt.Sprintf("v.blk.%d.attn_qkv.weight", layer))
	}
	useGELU, err := optionalMetadataBool(file, visionUseGELUKey)
	if err != nil {
		return Llama4VisionSpec{}, err
	}
	useSiLU, err := optionalMetadataBool(file, visionUseSiLUKey)
	if err != nil {
		return Llama4VisionSpec{}, err
	}
	if useGELU && useSiLU {
		return Llama4VisionSpec{}, errors.New("projector: Llama-4 GELU and SiLU flags conflict")
	}
	if useGELU {
		spec.Activation = visionGELU
	} else if useSiLU {
		spec.Activation = visionSiLU
	}
	if err := spec.validate(); err != nil {
		return Llama4VisionSpec{}, err
	}
	return spec, nil
}

func (s Llama4VisionSpec) validate() error {
	if err := s.visionBackboneSpec.validateRotary(); err != nil {
		return err
	}
	headWidth, headsOK := checked.DivExactInt(s.Hidden, s.Heads)
	patchSide, patchOK := checked.DivExactInt(s.ImageSize, s.PatchSize)
	_, mergeOK := checked.DivExactInt(patchSide, s.MergeSize)
	_, rotaryOK := checked.DivExactInt(headWidth, tensor.PairedExtent*tensor.PairedExtent)
	if !checked.PositiveInts(s.OutputHidden, s.AdapterIntermediate, s.AdapterHidden, s.MergeSize) ||
		s.MaxGridSide < tensor.PairedExtent ||
		!headsOK || !patchOK || !mergeOK || !rotaryOK || len(s.FusedQKV) != s.Layers {
		return fmt.Errorf("projector: invalid Llama-4 vision metadata: %+v", s)
	}
	return nil
}

func validateLlama4VisionCatalog(file *gguf.File, spec Llama4VisionSpec) ([]string, error) {
	patches := spec.ImageSize / spec.PatchSize
	shuffleWidth := spec.Hidden * spec.MergeSize * spec.MergeSize
	required := map[string][]uint64{
		visionClassEmbeddingTensor: {uint64(spec.Hidden)},
		"mm.model.mlp.1.weight":    {uint64(shuffleWidth), uint64(spec.AdapterIntermediate)},
		"mm.model.mlp.2.weight":    {uint64(spec.AdapterIntermediate), uint64(spec.AdapterHidden)},
		multimodalProjectionWeight: {uint64(spec.AdapterHidden), uint64(spec.OutputHidden)},
	}
	addSpatialVisionEmbeddingCatalog(file, required, spec.visionBackboneSpec,
		patches*patches+tensor.SingletonExtent, tensorOptional)
	if err := addOptionalVisionNormCatalog(file, required, spec.Hidden); err != nil {
		return nil, err
	}
	addStandardVisionLayerCatalog(file, required, spec.Layers, spec.Hidden, spec.Intermediate, spec.FusedQKV, false, tensorOptional)
	return validateProjectorTensorCatalog(file, required)
}

func PreprocessLlama4VisionImage(source image.Image, spec Llama4VisionSpec) (Llama4VisionInput, error) {
	if err := spec.validate(); err != nil {
		return Llama4VisionInput{}, err
	}
	bounds, err := imageBounds(source)
	if err != nil {
		return Llama4VisionInput{}, err
	}
	gridTiles, _ := checked.MulInt(spec.MaxGridSide, spec.MaxGridSide)
	images := make([]image.Image, tensor.FirstOffset, gridTiles+tensor.SingletonExtent)
	gridW, gridH := tensor.FirstOffset, tensor.FirstOffset
	if bounds.Dx() > spec.ImageSize || bounds.Dy() > spec.ImageSize {
		gridW, gridH = llama4BestGrid(bounds.Dx(), bounds.Dy(), spec.ImageSize, spec.MaxGridSide)
		refined := resizeFitBicubic(source, gridW*spec.ImageSize, gridH*spec.ImageSize, nil)
		for y := range gridH {
			for x := range gridW {
				images = append(images, cropImage(refined, image.Rect(x*spec.ImageSize, y*spec.ImageSize,
					(x+tensor.SingletonExtent)*spec.ImageSize, (y+tensor.SingletonExtent)*spec.ImageSize)))
			}
		}
	}
	images = append(images, resizeFitBicubic(source, spec.ImageSize, spec.ImageSize, nil))
	tiles := make([]Llama4VisionTile, len(images))
	for index, current := range images {
		patches, err := patchRasterImage(current, spec.PatchSize, spec.ImageMean, spec.ImageStd)
		if err != nil {
			return Llama4VisionInput{}, err
		}
		tiles[index] = Llama4VisionTile(patches)
	}
	return Llama4VisionInput{Tiles: tiles, GridH: gridH, GridW: gridW}, nil
}

func llama4BestGrid(width, height, size, maxSide int) (int, int) {
	bestW, bestH := tensor.SingletonExtent, tensor.PairedExtent
	bestEffective, bestWaste := -tensor.SingletonExtent, math.MaxInt
	for gridW := tensor.SingletonExtent; gridW <= maxSide; gridW++ {
		for gridH := tensor.SingletonExtent; gridH <= maxSide; gridH++ {
			if gridW == tensor.SingletonExtent && gridH == tensor.SingletonExtent {
				continue
			}
			candidateW, candidateH := gridW*size, gridH*size
			scale := min(float64(candidateW)/float64(width), float64(candidateH)/float64(height))
			targetW, targetH := int(float64(width)*scale), int(float64(height)*scale)
			effective := min(targetW*targetH, width*height)
			waste := candidateW*candidateH - effective
			if effective > bestEffective || effective == bestEffective && waste < bestWaste {
				bestW, bestH, bestEffective, bestWaste = gridW, gridH, effective, waste
			}
		}
	}
	return bestW, bestH
}

func cropImage(source image.Image, rectangle image.Rectangle) image.Image {
	output := image.NewRGBA(image.Rect(tensor.FirstOffset, tensor.FirstOffset, rectangle.Dx(), rectangle.Dy()))
	for y := range rectangle.Dy() {
		for x := range rectangle.Dx() {
			output.Set(x, y, source.At(rectangle.Min.X+x, rectangle.Min.Y+y))
		}
	}
	return output
}

func (r *Llama4VisionRunner) EncodeImage(ctx context.Context, source image.Image) (Llama4VisionOutput, error) {
	if r == nil || r.file == nil {
		return Llama4VisionOutput{}, errRunnerClosed
	}
	input, err := PreprocessLlama4VisionImage(source, r.spec)
	if err != nil {
		return Llama4VisionOutput{}, err
	}
	var embeddings []float32
	rows := tensor.FirstOffset
	for index, tile := range input.Tiles {
		var value reference.Value
		var tileErr error
		if r.cuda != nil {
			value, tileErr = r.encodeTileCUDA(ctx, tile)
		} else {
			value, tileErr = r.encodeTile(ctx, tile)
		}
		if tileErr != nil {
			return Llama4VisionOutput{}, fmt.Errorf("projector: encode Llama-4 tile %d: %w", index, tileErr)
		}
		embeddings = append(embeddings, value.Data...)
		rows += int(value.Shape.Dims[tensor.SingletonExtent])
	}
	value, err := reference.NewValue(tensor.MustShape(uint64(r.spec.OutputHidden), uint64(rows)), embeddings)
	if err != nil {
		return Llama4VisionOutput{}, err
	}
	return Llama4VisionOutput{Embeddings: value, TileCount: len(input.Tiles), GridH: input.GridH, GridW: input.GridW}, nil
}

func (r *Llama4VisionRunner) encodeTile(ctx context.Context, input Llama4VisionTile) (reference.Value, error) {
	patchRows, patchWidth, err := validateSpatialPatchStorage(
		len(input.PixelValues), input.GridH, input.GridW, r.spec.PatchSize, media.RGBChannels,
	)
	if err != nil || input.GridH != input.GridW {
		return reference.Value{}, errors.New("projector: Llama-4 tile shape is inconsistent")
	}
	patchWeight, err := r.load(ctx, visionPatchWeightTensor)
	if err != nil {
		return reference.Value{}, err
	}
	patchBias, err := r.optionalBias(ctx, visionPatchBiasTensor)
	if err != nil {
		return reference.Value{}, err
	}
	hidden := hostmath.LinearF64BiasFirstNew(input.PixelValues, patchWeight.Data, patchBias, patchRows, patchWidth, r.spec.Hidden)
	classEmbedding, err := r.load(ctx, visionClassEmbeddingTensor)
	if err != nil {
		return reference.Value{}, err
	}
	hidden = append(hidden, classEmbedding.Data...)
	rows := patchRows + tensor.SingletonExtent
	positions, err := r.load(ctx, visionPositionWeightTensor)
	if err != nil {
		return reference.Value{}, err
	}
	for index := range hidden {
		hidden[index] += positions.Data[index]
	}
	if r.spec.PreLayerNorm {
		hidden, err = r.affineNormalize(ctx, hidden, rows, "v.pre_ln")
		if err != nil {
			return reference.Value{}, err
		}
	}
	for layer := range r.spec.Layers {
		if err := r.runLayer(ctx, hidden, input.GridH, input.GridW, layer); err != nil {
			return reference.Value{}, err
		}
	}
	if r.spec.PostLayerNorm {
		hidden, err = r.affineNormalize(ctx, hidden, rows, "v.post_ln")
		if err != nil {
			return reference.Value{}, err
		}
	}
	hidden = hidden[:patchRows*r.spec.Hidden]
	mergePlan, err := newPixelMergePlan(input.GridH, input.GridW, r.spec.MergeSize)
	if err != nil {
		return reference.Value{}, err
	}
	shuffleWidth := r.spec.Hidden * r.spec.MergeSize * r.spec.MergeSize
	shuffled, err := mergePlan.shuffle(hidden, r.spec.Hidden)
	if err != nil {
		return reference.Value{}, err
	}
	mlp1, err := r.load(ctx, "mm.model.mlp.1.weight")
	if err != nil {
		return reference.Value{}, err
	}
	adapted := hostmath.LinearF64BiasFirstNew(shuffled, mlp1.Data, nil, mergePlan.outputRows, shuffleWidth, r.spec.AdapterIntermediate)
	hostmath.GELUTanhInPlace(adapted)
	mlp2, err := r.load(ctx, "mm.model.mlp.2.weight")
	if err != nil {
		return reference.Value{}, err
	}
	adapted = hostmath.LinearF64BiasFirstNew(adapted, mlp2.Data, nil, mergePlan.outputRows, r.spec.AdapterIntermediate, r.spec.AdapterHidden)
	hostmath.GELUTanhInPlace(adapted)
	projection, err := r.load(ctx, multimodalProjectionWeight)
	if err != nil {
		return reference.Value{}, err
	}
	output := hostmath.LinearF64BiasFirstNew(adapted, projection.Data, nil, mergePlan.outputRows, r.spec.AdapterHidden, r.spec.OutputHidden)
	return reference.NewValue(tensor.MustShape(uint64(r.spec.OutputHidden), uint64(mergePlan.outputRows)), output)
}

func (r *Llama4VisionRunner) runLayer(ctx context.Context, hidden []float32, gridH, gridW, layer int) error {
	rows := gridH*gridW + tensor.SingletonExtent
	prefix := fmt.Sprintf("v.blk.%d.", layer)
	norm, err := r.affineNormalize(ctx, hidden, rows, prefix+"ln1")
	if err != nil {
		return err
	}
	qkv, err := r.projectQKV(ctx, norm, rows, prefix, layer)
	if err != nil {
		return err
	}
	applySpatialRotaryQK(qkv, gridH, gridW, r.spec.Hidden, r.spec.Heads, r.spec.RopeFrequency)
	attention := hostmath.InterleavedQKVAttentionF64(qkv, rows, r.spec.Hidden, r.spec.Heads, float64(r.attention.scale()))
	outWeight, err := r.load(ctx, prefix+"attn_out.weight")
	if err != nil {
		return err
	}
	outBias, err := r.optionalBias(ctx, prefix+"attn_out.bias")
	if err != nil {
		return err
	}
	projected := hostmath.LinearF64BiasFirstNew(attention, outWeight.Data, outBias, rows, r.spec.Hidden, r.spec.Hidden)
	for index := range hidden {
		hidden[index] += projected[index]
	}
	norm, err = r.affineNormalize(ctx, hidden, rows, prefix+"ln2")
	if err != nil {
		return err
	}
	upWeight, err := r.load(ctx, prefix+"ffn_up.weight")
	if err != nil {
		return err
	}
	upBias, err := r.optionalBias(ctx, prefix+"ffn_up.bias")
	if err != nil {
		return err
	}
	up := hostmath.LinearF64BiasFirstNew(norm, upWeight.Data, upBias, rows, r.spec.Hidden, r.spec.Intermediate)
	for index, value := range up {
		up[index] = r.activate(value)
	}
	downWeight, err := r.load(ctx, prefix+"ffn_down.weight")
	if err != nil {
		return err
	}
	downBias, err := r.optionalBias(ctx, prefix+"ffn_down.bias")
	if err != nil {
		return err
	}
	down := hostmath.LinearF64BiasFirstNew(up, downWeight.Data, downBias, rows, r.spec.Intermediate, r.spec.Hidden)
	for index := range hidden {
		hidden[index] += down[index]
	}
	return nil
}

func applySpatialRotaryQK(qkv []float32, gridH, gridW, hidden, heads int, theta float32) {
	rows, headWidth := gridH*gridW+tensor.SingletonExtent, hidden/heads
	axisWidth := headWidth / tensor.PairedExtent
	invFrequency := hostmath.RopeInvFreq(float64(theta), axisWidth)
	for token := range rows {
		positionW, positionH := tensor.FirstOffset, tensor.FirstOffset
		if token < rows-tensor.SingletonExtent {
			positionW = token%gridW + tensor.SingletonExtent
			positionH = token/gridW + tensor.SingletonExtent
		}
		for head := range heads {
			for part := range tensor.PairedExtent {
				base := token*tensor.TripleExtent*hidden + part*hidden + head*headWidth
				for axis, position := range [tensor.PairedExtent]int{positionW, positionH} {
					axisBase := base + axis*axisWidth
					for pair, frequency := range invFrequency {
						angle := float64(position) * frequency
						cosine, sine := float32(math.Cos(angle)), float32(math.Sin(angle))
						index := axisBase + pair*tensor.PairedExtent
						left, right := qkv[index], qkv[index+tensor.SingletonExtent]
						qkv[index], qkv[index+tensor.SingletonExtent] = left*cosine-right*sine, left*sine+right*cosine
					}
				}
			}
		}
	}
}

func (r *Llama4VisionRunner) activate(value float32) float32 {
	switch r.spec.Activation {
	case visionGELU:
		return float32(hostmath.GELUTanh(float64(value)))
	case visionSiLU:
		return float32(hostmath.SiLU(float64(value)))
	default:
		return float32(hostmath.QuickGELU(float64(value)))
	}
}

func (r *Llama4VisionRunner) affineNormalize(ctx context.Context, input []float32, rows int, prefix string) ([]float32, error) {
	weight, bias, err := r.loadPair(ctx, prefix+".weight", prefix+".bias")
	if err != nil {
		return nil, err
	}
	output := make([]float32, len(input))
	hostmath.LayerNormF32AffineInto(output, input, weight.Data, bias.Data, rows, len(input)/rows, r.spec.LayerNormEpsilon)
	return output, nil
}

func (r *Llama4VisionRunner) projectQKV(ctx context.Context, input []float32, rows int, prefix string, layer int) ([]float32, error) {
	if r.spec.FusedQKV[layer] {
		weight, err := r.load(ctx, prefix+"attn_qkv.weight")
		if err != nil {
			return nil, err
		}
		bias, err := r.optionalBias(ctx, prefix+"attn_qkv.bias")
		if err != nil {
			return nil, err
		}
		return hostmath.LinearF64BiasFirstNew(input, weight.Data, bias, rows, r.spec.Hidden, tensor.TripleExtent*r.spec.Hidden), nil
	}
	rowWidth := tensor.TripleExtent * r.spec.Hidden
	qkv := make([]float32, rows*rowWidth)
	for partIndex, part := range []string{"q", "k", "v"} {
		weight, err := r.load(ctx, prefix+"attn_"+part+".weight")
		if err != nil {
			return nil, err
		}
		bias, err := r.optionalBias(ctx, prefix+"attn_"+part+".bias")
		if err != nil {
			return nil, err
		}
		hostmath.LinearF64BiasFirstStrided(
			qkv, input, weight.Data, bias, rows, r.spec.Hidden, r.spec.Hidden, rowWidth, partIndex*r.spec.Hidden,
		)
	}
	return qkv, nil
}

func (r *Llama4VisionRunner) optionalBias(ctx context.Context, name string) ([]float32, error) {
	if !hasTensor(r.file, name) {
		return nil, nil
	}
	value, err := r.load(ctx, name)
	return value.Data, err
}
