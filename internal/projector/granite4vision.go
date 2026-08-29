package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const granite4VisionProjectorType = "granite4_vision"

type Granite4VisionResolution struct {
	Width  int
	Height int
}

type Granite4VisionSpec struct {
	visionBackboneSpec
	ProjectionDim  int
	QFormerWidth   int
	WindowSide     int
	QuerySide      int
	FeatureLayers  []int
	SpatialOffsets []int
	GridCandidates []Granite4VisionResolution
}

type Granite4VisionTile struct {
	PixelValues []float32
	AddNewline  bool
}

type Granite4VisionInput struct {
	Tiles []Granite4VisionTile
	GridH int
	GridW int
}

type Granite4VisionOutput struct {
	Embeddings          reference.Value
	DeepstackEmbeddings []reference.Value
	TileCount           int
	GridH               int
	GridW               int
}

type Granite4VisionRunner struct {
	projectorResources
	spec      Granite4VisionSpec
	attention visionAttentionPlan
}

func (r *Granite4VisionRunner) Spec() Granite4VisionSpec {
	if r == nil {
		return Granite4VisionSpec{}
	}
	return r.spec
}

func ReadGranite4VisionSpec(file *gguf.File) (Granite4VisionSpec, error) {
	spec := Granite4VisionSpec{}
	if err := readVisionBackbone(file, granite4VisionProjectorType, &spec.ProjectionDim, &spec.visionBackboneSpec); err != nil {
		return Granite4VisionSpec{}, err
	}
	if err := readProjectionNorm(file, &spec.ProjectionNormEpsilon); err != nil {
		return Granite4VisionSpec{}, err
	}
	if err := readMetadataIntFields(file,
		metadataIntField{"clip.vision.projector.window_side", &spec.WindowSide},
		metadataIntField{"clip.vision.projector.query_side", &spec.QuerySide},
	); err != nil {
		return Granite4VisionSpec{}, err
	}
	features, err := metadataInts(file, "clip.vision.feature_layer", tensorRequired, false)
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	offsets, err := metadataInts(file, "clip.vision.projector.spatial_offsets", tensorRequired, false)
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	candidates, err := granite4GridCandidates(file)
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	qfUp, ok := file.Tensor("v.proj_blk.0.ffn_up.weight")
	qfWidth, widthOK := matrixRowsInt(qfUp, ok)
	if !widthOK {
		return Granite4VisionSpec{}, errors.New("projector: Granite 4 Vision QFormer FFN is unavailable")
	}
	spec.QFormerWidth = qfWidth
	spec.FeatureLayers = features
	spec.SpatialOffsets = offsets
	spec.GridCandidates = candidates
	if err := spec.validate(); err != nil {
		return Granite4VisionSpec{}, err
	}
	return spec, nil
}

func granite4GridCandidates(file *gguf.File) ([]Granite4VisionResolution, error) {
	values, err := metadataInts(file, "clip.vision.image_grid_pinpoints", tensorRequired, false)
	if err != nil {
		return nil, err
	}
	if len(values)%tensor.PairedExtent != tensor.FirstOffset {
		return nil, errors.New("projector: Granite 4 Vision grid candidates must contain width/height pairs")
	}
	result := make([]Granite4VisionResolution, len(values)/tensor.PairedExtent)
	for index := range result {
		base := index * tensor.PairedExtent
		result[index] = Granite4VisionResolution{
			Width: values[base], Height: values[base+tensor.SingletonExtent],
		}
	}
	return result, nil
}

func (s Granite4VisionSpec) validate() error {
	if err := s.visionBackboneSpec.validate(); err != nil {
		return err
	}
	patchSide, patchOK := checked.DivExactInt(s.ImageSize, s.PatchSize)
	windows, windowsOK := checked.DivExactInt(patchSide, s.WindowSide)
	newSide, newSideOK := checked.MulInt(windows, s.QuerySide)
	_, queryOK := checked.DivExactInt(s.WindowSide, s.QuerySide)
	_, downsampleOK := checked.DivExactInt(patchSide, newSide)
	_, headsOK := checked.DivExactInt(s.Hidden, s.Heads)
	if !checked.PositiveInts(s.ProjectionDim, s.QFormerWidth, s.WindowSide, s.QuerySide, newSide, len(s.FeatureLayers)) ||
		!checked.PositiveFinite32(s.ProjectionNormEpsilon) || !patchOK || !windowsOK || !newSideOK ||
		!queryOK || !downsampleOK || !headsOK || len(s.FeatureLayers) != len(s.SpatialOffsets) {
		return fmt.Errorf("projector: invalid Granite 4 Vision metadata: %+v", s)
	}
	for _, layer := range s.FeatureLayers {
		if layer < tensor.FirstOffset || layer >= s.Layers {
			return fmt.Errorf("projector: Granite 4 Vision feature layer %d is invalid", layer)
		}
	}
	for index, offset := range s.SpatialOffsets {
		if offset < -tensor.SingletonExtent || offset > tensor.TripleExtent ||
			offset >= tensor.FirstOffset && newSide != patchSide/tensor.PairedExtent {
			return fmt.Errorf("projector: Granite 4 Vision spatial offset %d at block %d is invalid", offset, index)
		}
	}
	for _, candidate := range s.GridCandidates {
		_, widthOK := checked.DivExactInt(candidate.Width, s.ImageSize)
		_, heightOK := checked.DivExactInt(candidate.Height, s.ImageSize)
		if !widthOK || !heightOK {
			return fmt.Errorf("projector: Granite 4 Vision grid candidate %+v is invalid", candidate)
		}
	}
	return nil
}

func validateGranite4VisionCatalog(file *gguf.File, spec Granite4VisionSpec) ([]string, error) {
	patches := spec.ImageSize / spec.PatchSize
	required := map[string][]uint64{
		visionImageNewlineTensor: {uint64(spec.ProjectionDim)},
	}
	addSpatialVisionEmbeddingCatalog(file, required, spec.visionBackboneSpec, patches*patches, tensorRequired)
	addStandardVisionLayerCatalog(file, required, spec.Layers, spec.Hidden, spec.Intermediate, nil, false, tensorRequired)
	queryLength := spec.QuerySide * spec.QuerySide
	windowLength := spec.WindowSide * spec.WindowSide
	for block := range spec.FeatureLayers {
		prefix := fmt.Sprintf("v.proj_blk.%d.", block)
		required[prefix+"img_pos"] = []uint64{uint64(spec.Hidden), uint64(windowLength)}
		required[prefix+"query"] = []uint64{uint64(spec.Hidden), uint64(queryLength)}
		required[prefix+"linear.weight"] = []uint64{uint64(spec.Hidden), uint64(spec.ProjectionDim)}
		required[prefix+"linear.bias"] = []uint64{uint64(spec.ProjectionDim)}
		for _, name := range []string{"norm", "post_norm", "self_attn_norm", "cross_attn_norm", "ffn_norm"} {
			required[prefix+name+".weight"] = []uint64{uint64(spec.Hidden)}
			required[prefix+name+".bias"] = []uint64{uint64(spec.Hidden)}
		}
		for _, name := range []string{
			"self_attn_q", "self_attn_k", "self_attn_v", "self_attn_out",
			"cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_out",
		} {
			required[prefix+name+".weight"] = []uint64{uint64(spec.Hidden), uint64(spec.Hidden)}
			required[prefix+name+".bias"] = []uint64{uint64(spec.Hidden)}
		}
		required[prefix+"ffn_up.weight"] = []uint64{uint64(spec.Hidden), uint64(spec.QFormerWidth)}
		required[prefix+"ffn_up.bias"] = []uint64{uint64(spec.QFormerWidth)}
		required[prefix+"ffn_down.weight"] = []uint64{uint64(spec.QFormerWidth), uint64(spec.Hidden)}
		required[prefix+"ffn_down.bias"] = []uint64{uint64(spec.Hidden)}
	}
	return validateProjectorTensorCatalog(file, required)
}

func PreprocessGranite4VisionImage(source image.Image, spec Granite4VisionSpec) (Granite4VisionInput, error) {
	if err := spec.validate(); err != nil {
		return Granite4VisionInput{}, err
	}
	bounds, err := imageBounds(source)
	if err != nil {
		return Granite4VisionInput{}, err
	}
	images := []image.Image{resizeFitBicubic(source, spec.ImageSize, spec.ImageSize, nil)}
	gridW, gridH := tensor.FirstOffset, tensor.FirstOffset
	if bounds.Dx() > spec.ImageSize || bounds.Dy() > spec.ImageSize {
		resolution := granite4BestResolution(bounds.Dx(), bounds.Dy(), spec)
		gridW, gridH = resolution.Width/spec.ImageSize, resolution.Height/spec.ImageSize
		refined := resizeFitBicubic(source, resolution.Width, resolution.Height, nil)
		for y := range gridH {
			for x := range gridW {
				images = append(images, cropImage(refined, image.Rect(x*spec.ImageSize, y*spec.ImageSize,
					(x+tensor.SingletonExtent)*spec.ImageSize, (y+tensor.SingletonExtent)*spec.ImageSize)))
			}
		}
	}
	tiles := make([]Granite4VisionTile, len(images))
	for index, current := range images {
		patches, err := patchRasterImage(current, spec.PatchSize, spec.ImageMean, spec.ImageStd)
		if err != nil {
			return Granite4VisionInput{}, err
		}
		tiles[index] = Granite4VisionTile{PixelValues: patches.PixelValues,
			AddNewline: len(images) == tensor.SingletonExtent || index > tensor.FirstOffset}
	}
	return Granite4VisionInput{Tiles: tiles, GridH: gridH, GridW: gridW}, nil
}

func granite4BestResolution(width, height int, spec Granite4VisionSpec) Granite4VisionResolution {
	best := spec.GridCandidates[tensor.FirstOffset]
	bestEffective, bestWaste := -tensor.SingletonExtent, math.MaxInt
	for _, candidate := range spec.GridCandidates {
		scale := min(float64(candidate.Width)/float64(width), float64(candidate.Height)/float64(height))
		targetW, targetH := int(float64(width)*scale), int(float64(height)*scale)
		effective := min(targetW*targetH, width*height)
		waste := candidate.Width*candidate.Height - effective
		if effective > bestEffective || effective == bestEffective && waste < bestWaste {
			best, bestEffective, bestWaste = candidate, effective, waste
		}
	}
	return best
}

func (r *Granite4VisionRunner) EncodeImage(ctx context.Context, source image.Image) (Granite4VisionOutput, error) {
	if r == nil || r.file == nil {
		return Granite4VisionOutput{}, errRunnerClosed
	}
	input, err := PreprocessGranite4VisionImage(source, r.spec)
	if err != nil {
		return Granite4VisionOutput{}, err
	}
	streams := make([][]float32, len(r.spec.FeatureLayers))
	rows := tensor.FirstOffset
	for tileIndex, tile := range input.Tiles {
		values, tileErr := r.encodeTile(ctx, tile)
		if tileErr != nil {
			return Granite4VisionOutput{}, fmt.Errorf("projector: encode Granite 4 Vision tile %d: %w", tileIndex, tileErr)
		}
		for streamIndex, value := range values {
			streams[streamIndex] = append(streams[streamIndex], value.Data...)
		}
		rows += int(values[tensor.FirstOffset].Shape.Dims[tensor.SingletonExtent])
	}
	base, err := reference.NewValue(tensor.MustShape(uint64(r.spec.ProjectionDim), uint64(rows)), streams[tensor.FirstOffset])
	if err != nil {
		return Granite4VisionOutput{}, err
	}
	deepstack := make([]reference.Value, len(streams)-tensor.SingletonExtent)
	for index := range deepstack {
		deepstack[index], err = reference.NewValue(tensor.MustShape(uint64(r.spec.ProjectionDim), uint64(rows)), streams[index+tensor.SingletonExtent])
		if err != nil {
			return Granite4VisionOutput{}, err
		}
	}
	return Granite4VisionOutput{
		Embeddings: base, DeepstackEmbeddings: deepstack,
		TileCount: len(input.Tiles), GridH: input.GridH, GridW: input.GridW,
	}, nil
}

func (r *Granite4VisionRunner) encodeTile(ctx context.Context, tile Granite4VisionTile) ([]reference.Value, error) {
	if r.cuda != nil {
		return r.encodeTileCUDA(ctx, tile)
	}
	side := r.spec.ImageSize / r.spec.PatchSize
	rows, patchWidth, err := validateSpatialPatchStorage(
		len(tile.PixelValues), side, side, r.spec.PatchSize, media.RGBChannels,
	)
	if err != nil {
		return nil, fmt.Errorf("projector: Granite 4 Vision tile: %w", err)
	}
	patchWeight, err := r.load(ctx, visionPatchWeightTensor)
	if err != nil {
		return nil, err
	}
	patchBias, err := r.load(ctx, visionPatchBiasTensor)
	if err != nil {
		return nil, err
	}
	hidden := hostmath.LinearF64BiasFirstNew(tile.PixelValues, patchWeight.Data, patchBias.Data, rows, patchWidth, r.spec.Hidden)
	positions, err := r.load(ctx, visionPositionWeightTensor)
	if err != nil {
		return nil, err
	}
	for index := range hidden {
		hidden[index] += positions.Data[index]
	}
	layerOutputs := make([][]float32, r.spec.Layers)
	for layer := 0; layer < r.spec.Layers; layer++ {
		if err := r.runVisionLayer(ctx, hidden, rows, layer); err != nil {
			return nil, err
		}
		layerOutputs[layer] = slices.Clone(hidden)
	}
	outputs := make([]reference.Value, len(r.spec.FeatureLayers))
	for block, layer := range r.spec.FeatureLayers {
		projected, projectErr := r.runQFormerBlock(ctx, layerOutputs[layer], side, block)
		if projectErr != nil {
			return nil, projectErr
		}
		outRows := len(projected) / r.spec.ProjectionDim
		if tile.AddNewline {
			newline, loadErr := r.load(ctx, visionImageNewlineTensor)
			if loadErr != nil {
				return nil, loadErr
			}
			projected = append(projected, newline.Data...)
			outRows++
		}
		outputs[block], err = reference.NewValue(tensor.MustShape(uint64(r.spec.ProjectionDim), uint64(outRows)), projected)
		if err != nil {
			return nil, err
		}
	}
	return outputs, nil
}

func (r *Granite4VisionRunner) runVisionLayer(ctx context.Context, hidden []float32, rows, layer int) error {
	prefix := fmt.Sprintf("v.blk.%d.", layer)
	norm, err := r.affineNormalize(ctx, hidden, rows, prefix+"ln1", r.spec.LayerNormEpsilon)
	if err != nil {
		return err
	}
	rowWidth := tensor.TripleExtent * r.spec.Hidden
	qkv := make([]float32, rows*rowWidth)
	for partIndex, part := range []string{"q", "k", "v"} {
		weight, bias, loadErr := r.loadPair(ctx, prefix+"attn_"+part+".weight", prefix+"attn_"+part+".bias")
		if loadErr != nil {
			return loadErr
		}
		hostmath.LinearF64BiasFirstStrided(
			qkv, norm, weight.Data, bias.Data, rows, r.spec.Hidden, r.spec.Hidden, rowWidth, partIndex*r.spec.Hidden,
		)
	}
	attention := hostmath.InterleavedQKVAttentionF64(qkv, rows, r.spec.Hidden, r.spec.Heads, float64(r.attention.scale()))
	outWeight, outBias, err := r.loadPair(ctx, prefix+"attn_out.weight", prefix+"attn_out.bias")
	if err != nil {
		return err
	}
	projected := hostmath.LinearF64BiasFirstNew(attention, outWeight.Data, outBias.Data, rows, r.spec.Hidden, r.spec.Hidden)
	for index := range hidden {
		hidden[index] += projected[index]
	}
	norm, err = r.affineNormalize(ctx, hidden, rows, prefix+"ln2", r.spec.LayerNormEpsilon)
	if err != nil {
		return err
	}
	upWeight, upBias, err := r.loadPair(ctx, prefix+"ffn_up.weight", prefix+"ffn_up.bias")
	if err != nil {
		return err
	}
	up := hostmath.LinearF64BiasFirstNew(norm, upWeight.Data, upBias.Data, rows, r.spec.Hidden, r.spec.Intermediate)
	hostmath.GELUTanhInPlace(up)
	downWeight, downBias, err := r.loadPair(ctx, prefix+"ffn_down.weight", prefix+"ffn_down.bias")
	if err != nil {
		return err
	}
	down := hostmath.LinearF64BiasFirstNew(up, downWeight.Data, downBias.Data, rows, r.spec.Intermediate, r.spec.Hidden)
	for index := range hidden {
		hidden[index] += down[index]
	}
	return nil
}

func (r *Granite4VisionRunner) runQFormerBlock(ctx context.Context, hidden []float32, side, block int) ([]float32, error) {
	prefix := fmt.Sprintf("v.proj_blk.%d.", block)
	rows := side * side
	x, err := r.affineNormalize(ctx, hidden, rows, prefix+"norm", r.spec.LayerNormEpsilon)
	if err != nil {
		return nil, err
	}
	windowSide, querySide := r.spec.WindowSide, r.spec.QuerySide
	windowsPerSide := side / windowSide
	windows := windowsPerSide * windowsPerSide
	encLength, queryLength := windowSide*windowSide, querySide*querySide
	newSide := windowsPerSide * querySide
	enc := windowSpatialRows(x, side, windowSide, r.spec.Hidden)
	down := downsampleSpatialRows(x, side, newSide, r.spec.Hidden, r.spec.SpatialOffsets[block])
	queryWindows := windowSpatialRows(down, newSide, querySide, r.spec.Hidden)
	query, imagePosition, err := r.loadPair(ctx, prefix+"query", prefix+"img_pos")
	if err != nil {
		return nil, err
	}
	for window := range windows {
		for row := range queryLength {
			for channel := 0; channel < r.spec.Hidden; channel++ {
				queryWindows[(window*queryLength+row)*r.spec.Hidden+channel] += query.Data[row*r.spec.Hidden+channel]
			}
		}
		for row := range encLength {
			for channel := 0; channel < r.spec.Hidden; channel++ {
				enc[(window*encLength+row)*r.spec.Hidden+channel] += imagePosition.Data[row*r.spec.Hidden+channel]
			}
		}
	}
	queryWindows, err = r.affineNormalize(ctx, queryWindows, windows*queryLength, prefix+"post_norm", r.spec.ProjectionNormEpsilon)
	if err != nil {
		return nil, err
	}
	self, err := r.qformerAttention(ctx, queryWindows, queryWindows, windows, queryLength, queryLength, prefix+"self_attn")
	if err != nil {
		return nil, err
	}
	for index := range self {
		self[index] += queryWindows[index]
	}
	self, err = r.affineNormalize(ctx, self, windows*queryLength, prefix+"self_attn_norm", r.spec.ProjectionNormEpsilon)
	if err != nil {
		return nil, err
	}
	cross, err := r.qformerAttention(ctx, self, enc, windows, queryLength, encLength, prefix+"cross_attn")
	if err != nil {
		return nil, err
	}
	for index := range cross {
		cross[index] += self[index]
	}
	cross, err = r.affineNormalize(ctx, cross, windows*queryLength, prefix+"cross_attn_norm", r.spec.ProjectionNormEpsilon)
	if err != nil {
		return nil, err
	}
	upWeight, upBias, err := r.loadPair(ctx, prefix+"ffn_up.weight", prefix+"ffn_up.bias")
	if err != nil {
		return nil, err
	}
	up := hostmath.LinearF64BiasFirstNew(cross, upWeight.Data, upBias.Data, windows*queryLength, r.spec.Hidden, r.spec.QFormerWidth)
	for index, value := range up {
		up[index] = float32(0.5 * float64(value) * (1 + math.Erf(float64(value)/math.Sqrt2)))
	}
	downWeight, downBias, err := r.loadPair(ctx, prefix+"ffn_down.weight", prefix+"ffn_down.bias")
	if err != nil {
		return nil, err
	}
	ffn := hostmath.LinearF64BiasFirstNew(up, downWeight.Data, downBias.Data, windows*queryLength, r.spec.QFormerWidth, r.spec.Hidden)
	for index := range ffn {
		ffn[index] += cross[index]
	}
	ffn, err = r.affineNormalize(ctx, ffn, windows*queryLength, prefix+"ffn_norm", r.spec.ProjectionNormEpsilon)
	if err != nil {
		return nil, err
	}
	unwinned := unwindowSpatialRows(ffn, newSide, querySide, r.spec.Hidden)
	linearWeight, linearBias, err := r.loadPair(ctx, prefix+"linear.weight", prefix+"linear.bias")
	if err != nil {
		return nil, err
	}
	return hostmath.LinearF64BiasFirstNew(unwinned, linearWeight.Data, linearBias.Data, newSide*newSide, r.spec.Hidden, r.spec.ProjectionDim), nil
}

func (r *Granite4VisionRunner) qformerAttention(
	ctx context.Context,
	queryInput, keyValueInput []float32,
	windows, queryRows, keyRows int,
	prefix string,
) ([]float32, error) {
	parts := make([][]float32, tensor.TripleExtent)
	for index, name := range [tensor.TripleExtent]string{"q", "k", "v"} {
		weight, bias, err := r.loadPair(ctx, prefix+"_"+name+".weight", prefix+"_"+name+".bias")
		if err != nil {
			return nil, err
		}
		input, rows := keyValueInput, windows*keyRows
		if index == tensor.FirstOffset {
			input, rows = queryInput, windows*queryRows
		}
		parts[index] = hostmath.LinearF64BiasFirstNew(input, weight.Data, bias.Data, rows, r.spec.Hidden, r.spec.Hidden)
	}
	attention := hostmath.BatchedAttentionF32(
		parts[tensor.FirstOffset], parts[tensor.SingletonExtent], parts[tensor.PairedExtent], windows, queryRows, keyRows,
		r.spec.Hidden, r.spec.Heads,
	)
	outWeight, outBias, err := r.loadPair(ctx, prefix+"_out.weight", prefix+"_out.bias")
	if err != nil {
		return nil, err
	}
	return hostmath.LinearF64BiasFirstNew(attention, outWeight.Data, outBias.Data, windows*queryRows, r.spec.Hidden, r.spec.Hidden), nil
}

func windowSpatialRows(input []float32, side, windowSide, hidden int) []float32 {
	windowsPerSide := side / windowSide
	output := make([]float32, len(input))
	destination := tensor.FirstOffset
	for windowY := range windowsPerSide {
		for windowX := range windowsPerSide {
			for y := range windowSide {
				for x := range windowSide {
					source := ((windowY*windowSide+y)*side + windowX*windowSide + x) * hidden
					copy(output[destination:destination+hidden], input[source:source+hidden])
					destination += hidden
				}
			}
		}
	}
	return output
}

func unwindowSpatialRows(input []float32, side, windowSide, hidden int) []float32 {
	windowsPerSide := side / windowSide
	output := make([]float32, len(input))
	source := tensor.FirstOffset
	for windowY := range windowsPerSide {
		for windowX := range windowsPerSide {
			for y := range windowSide {
				for x := range windowSide {
					destination := ((windowY*windowSide+y)*side + windowX*windowSide + x) * hidden
					copy(output[destination:destination+hidden], input[source:source+hidden])
					source += hidden
				}
			}
		}
	}
	return output
}

func downsampleSpatialRows(input []float32, side, newSide, hidden, spatialOffset int) []float32 {
	output := make([]float32, newSide*newSide*hidden)
	if spatialOffset >= tensor.FirstOffset {
		offsetY := (spatialOffset >> tensor.SingletonExtent) & tensor.SingletonExtent
		offsetX := spatialOffset & tensor.SingletonExtent
		for y := range newSide {
			for x := range newSide {
				source := ((y*tensor.PairedExtent+offsetY)*side + x*tensor.PairedExtent + offsetX) * hidden
				copy(output[(y*newSide+x)*hidden:], input[source:source+hidden])
			}
		}
		return output
	}
	kernel := side / newSide
	for y := range newSide {
		for x := range newSide {
			destination := (y*newSide + x) * hidden
			for ky := range kernel {
				for kx := range kernel {
					source := ((y*kernel+ky)*side + x*kernel + kx) * hidden
					for channel := range hidden {
						output[destination+channel] += input[source+channel] / float32(kernel*kernel)
					}
				}
			}
		}
	}
	return output
}

func (r *Granite4VisionRunner) affineNormalize(
	ctx context.Context,
	input []float32,
	rows int,
	prefix string,
	epsilon float32,
) ([]float32, error) {
	weight, bias, err := r.loadPair(ctx, prefix+".weight", prefix+".bias")
	if err != nil {
		return nil, err
	}
	output := make([]float32, len(input))
	hostmath.LayerNormF32AffineInto(output, input, weight.Data, bias.Data, rows, len(input)/rows, epsilon)
	return output, nil
}
