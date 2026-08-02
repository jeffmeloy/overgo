package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

const granite4VisionProjectorType = "granite4_vision"

type Granite4VisionResolution struct {
	Width  int
	Height int
}

type Granite4VisionSpec struct {
	ImageSize        int
	PatchSize        int
	Hidden           int
	Intermediate     int
	ProjectionDim    int
	QFormerWidth     int
	Layers           int
	Heads            int
	WindowSide       int
	QuerySide        int
	LayerNormEpsilon float32
	ImageMean        [3]float32
	ImageStd         [3]float32
	FeatureLayers    []int
	SpatialOffsets   []int
	GridCandidates   []Granite4VisionResolution
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
	file *gguf.File
	spec Granite4VisionSpec
	cuda *granite4VisionCUDA
}

type Granite4VisionOpenOptions struct {
	CUDA          bool
	DeviceOrdinal int
}

func OpenGranite4Vision(path string) (*Granite4VisionRunner, error) {
	return OpenGranite4VisionWithOptions(path, Granite4VisionOpenOptions{})
}

func OpenGranite4VisionWithOptions(path string, options Granite4VisionOpenOptions) (*Granite4VisionRunner, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(cause error) (*Granite4VisionRunner, error) {
		_ = file.Close()
		return nil, cause
	}
	spec, err := ReadGranite4VisionSpec(file)
	if err != nil {
		return fail(err)
	}
	if err := validateGranite4VisionCatalog(file, spec); err != nil {
		return fail(err)
	}
	runner := &Granite4VisionRunner{file: file, spec: spec}
	if options.CUDA {
		runner.cuda, err = openGranite4VisionCUDA(context.Background(), file, spec, options.DeviceOrdinal)
		if err != nil {
			return fail(fmt.Errorf("projector: initialize Granite 4 Vision CUDA: %w", err))
		}
	}
	return runner, nil
}

func (r *Granite4VisionRunner) Close() error {
	if r == nil {
		return nil
	}
	var closeErr error
	if r.cuda != nil {
		closeErr = r.cuda.Close()
		r.cuda = nil
	}
	if r.file == nil {
		return closeErr
	}
	file := r.file
	r.file = nil
	return errors.Join(closeErr, file.Close())
}

func (r *Granite4VisionRunner) Spec() Granite4VisionSpec {
	if r == nil {
		return Granite4VisionSpec{}
	}
	return r.spec
}

func ReadGranite4VisionSpec(file *gguf.File) (Granite4VisionSpec, error) {
	if file == nil {
		return Granite4VisionSpec{}, errors.New("projector: GGUF file is nil")
	}
	architecture, err := metadataString(file, "general.architecture")
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	if architecture != "clip" {
		return Granite4VisionSpec{}, fmt.Errorf("projector: architecture %q is not clip", architecture)
	}
	projectorType, err := metadataString(file, "clip.projector_type")
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	if projectorType != granite4VisionProjectorType {
		return Granite4VisionSpec{}, fmt.Errorf("projector: type %q is not %s", projectorType, granite4VisionProjectorType)
	}
	hasVision, err := metadataBool(file, "clip.has_vision_encoder")
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	if !hasVision {
		return Granite4VisionSpec{}, errors.New("projector: vision encoder is disabled")
	}
	values := make([]int, 9)
	for index, key := range []string{
		"clip.vision.image_size", "clip.vision.patch_size", "clip.vision.embedding_length",
		"clip.vision.feed_forward_length", "clip.vision.projection_dim", "clip.vision.block_count",
		"clip.vision.attention.head_count", "clip.vision.projector.window_side", "clip.vision.projector.query_side",
	} {
		value, valueErr := metadataUint32(file, key)
		if valueErr != nil {
			return Granite4VisionSpec{}, valueErr
		}
		values[index] = int(value)
	}
	epsilon, err := metadataFloat32(file, "clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", 3)
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", 3)
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	features, err := granite4MetadataInts(file, "clip.vision.feature_layer", true)
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	offsets, err := granite4MetadataInts(file, "clip.vision.projector.spatial_offsets", true)
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	candidates, err := granite4GridCandidates(file)
	if err != nil {
		return Granite4VisionSpec{}, err
	}
	qfUp, ok := file.Tensor("v.proj_blk.0.ffn_up.weight")
	if !ok || qfUp.Dimensions != 2 {
		return Granite4VisionSpec{}, errors.New("projector: Granite 4 Vision QFormer FFN is unavailable")
	}
	spec := Granite4VisionSpec{
		ImageSize: values[0], PatchSize: values[1], Hidden: values[2], Intermediate: values[3],
		ProjectionDim: values[4], Layers: values[5], Heads: values[6], WindowSide: values[7], QuerySide: values[8],
		QFormerWidth: int(qfUp.Shape[1]), LayerNormEpsilon: epsilon,
		FeatureLayers: features, SpatialOffsets: offsets, GridCandidates: candidates,
	}
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	if err := spec.validate(); err != nil {
		return Granite4VisionSpec{}, err
	}
	return spec, nil
}

func granite4MetadataInts(file *gguf.File, key string, required bool) ([]int, error) {
	value, ok := file.MetadataValue(key)
	if !ok {
		if required {
			return nil, fmt.Errorf("projector: metadata %q is unavailable", key)
		}
		return nil, nil
	}
	if value.Type != gguf.ValueTypeArray {
		return nil, fmt.Errorf("projector: metadata %q must be an array", key)
	}
	var result []int
	switch values := value.Data.(type) {
	case []int32:
		if value.ArrayType != gguf.ValueTypeInt32 {
			return nil, fmt.Errorf("projector: metadata %q has mismatched array type", key)
		}
		result = make([]int, len(values))
		for index, item := range values {
			result[index] = int(item)
		}
	case []uint32:
		if value.ArrayType != gguf.ValueTypeUint32 {
			return nil, fmt.Errorf("projector: metadata %q has mismatched array type", key)
		}
		result = make([]int, len(values))
		for index, item := range values {
			result[index] = int(item)
		}
	default:
		return nil, fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	return result, nil
}

func granite4GridCandidates(file *gguf.File) ([]Granite4VisionResolution, error) {
	values, err := granite4MetadataInts(file, "clip.vision.image_grid_pinpoints", false)
	if err != nil {
		return nil, err
	}
	if len(values)%2 != 0 {
		return nil, errors.New("projector: Granite 4 Vision grid candidates must contain width/height pairs")
	}
	result := make([]Granite4VisionResolution, len(values)/2)
	for index := range result {
		result[index] = Granite4VisionResolution{Width: values[index*2], Height: values[index*2+1]}
	}
	return result, nil
}

func (s Granite4VisionSpec) validate() error {
	patchSide := 0
	if s.PatchSize > 0 {
		patchSide = s.ImageSize / s.PatchSize
	}
	newSide := 0
	if s.WindowSide > 0 {
		newSide = patchSide / s.WindowSide * s.QuerySide
	}
	if s.ImageSize <= 0 || s.PatchSize <= 0 || s.Hidden <= 0 || s.Intermediate <= 0 || s.ProjectionDim <= 0 ||
		s.QFormerWidth <= 0 || s.Layers <= 0 || s.Heads <= 0 || s.WindowSide <= 0 || s.QuerySide <= 0 ||
		s.ImageSize%s.PatchSize != 0 || s.Hidden%s.Heads != 0 || s.Hidden%64 != 0 || patchSide%s.WindowSide != 0 ||
		s.WindowSide%s.QuerySide != 0 || newSide <= 0 || patchSide%newSide != 0 ||
		len(s.FeatureLayers) == 0 || len(s.FeatureLayers) != len(s.SpatialOffsets) || s.LayerNormEpsilon <= 0 {
		return fmt.Errorf("projector: invalid Granite 4 Vision metadata: %+v", s)
	}
	for _, layer := range s.FeatureLayers {
		if layer < 0 || layer >= s.Layers {
			return fmt.Errorf("projector: Granite 4 Vision feature layer %d is invalid", layer)
		}
	}
	for index, offset := range s.SpatialOffsets {
		if offset < -1 || offset > 3 || offset >= 0 && newSide != patchSide/2 {
			return fmt.Errorf("projector: Granite 4 Vision spatial offset %d at block %d is invalid", offset, index)
		}
	}
	for _, candidate := range s.GridCandidates {
		if candidate.Width <= 0 || candidate.Height <= 0 || candidate.Width%s.ImageSize != 0 || candidate.Height%s.ImageSize != 0 {
			return fmt.Errorf("projector: Granite 4 Vision grid candidate %+v is invalid", candidate)
		}
	}
	for channel := range s.ImageStd {
		if s.ImageStd[channel] <= 0 || !finite32(s.ImageMean[channel]) || !finite32(s.ImageStd[channel]) {
			return fmt.Errorf("projector: invalid Granite 4 Vision normalization channel %d", channel)
		}
	}
	return nil
}

func validateGranite4VisionCatalog(file *gguf.File, spec Granite4VisionSpec) error {
	patches := spec.ImageSize / spec.PatchSize
	required := map[string][]uint64{
		"v.patch_embd.weight":    {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.patch_embd.bias":      {uint64(spec.Hidden)},
		"v.position_embd.weight": {uint64(spec.Hidden), uint64(patches * patches)},
		"v.image_newline":        {uint64(spec.ProjectionDim)},
	}
	for layer := 0; layer < spec.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for name, shape := range map[string][]uint64{
			"attn_q.weight": {uint64(spec.Hidden), uint64(spec.Hidden)}, "attn_q.bias": {uint64(spec.Hidden)},
			"attn_k.weight": {uint64(spec.Hidden), uint64(spec.Hidden)}, "attn_k.bias": {uint64(spec.Hidden)},
			"attn_v.weight": {uint64(spec.Hidden), uint64(spec.Hidden)}, "attn_v.bias": {uint64(spec.Hidden)},
			"attn_out.weight": {uint64(spec.Hidden), uint64(spec.Hidden)}, "attn_out.bias": {uint64(spec.Hidden)},
			"ffn_up.weight": {uint64(spec.Hidden), uint64(spec.Intermediate)}, "ffn_up.bias": {uint64(spec.Intermediate)},
			"ffn_down.weight": {uint64(spec.Intermediate), uint64(spec.Hidden)}, "ffn_down.bias": {uint64(spec.Hidden)},
			"ln1.weight": {uint64(spec.Hidden)}, "ln1.bias": {uint64(spec.Hidden)},
			"ln2.weight": {uint64(spec.Hidden)}, "ln2.bias": {uint64(spec.Hidden)},
		} {
			required[prefix+name] = shape
		}
	}
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
	return validateProjectorTensorShapes(file, required)
}

func PreprocessGranite4VisionImage(source image.Image, spec Granite4VisionSpec) (Granite4VisionInput, error) {
	if source == nil {
		return Granite4VisionInput{}, errors.New("projector: image is nil")
	}
	if err := spec.validate(); err != nil {
		return Granite4VisionInput{}, err
	}
	bounds := source.Bounds()
	if bounds.Dx() <= 0 || bounds.Dy() <= 0 {
		return Granite4VisionInput{}, errors.New("projector: image bounds are empty")
	}
	images := []image.Image{resizeFitBicubic(source, spec.ImageSize, spec.ImageSize)}
	gridW, gridH := 0, 0
	if bounds.Dx() > spec.ImageSize || bounds.Dy() > spec.ImageSize {
		resolution := granite4BestResolution(bounds.Dx(), bounds.Dy(), spec)
		gridW, gridH = resolution.Width/spec.ImageSize, resolution.Height/spec.ImageSize
		refined := resizeFitBicubic(source, resolution.Width, resolution.Height)
		for y := 0; y < gridH; y++ {
			for x := 0; x < gridW; x++ {
				images = append(images, cropImage(refined, image.Rect(x*spec.ImageSize, y*spec.ImageSize, (x+1)*spec.ImageSize, (y+1)*spec.ImageSize)))
			}
		}
	}
	tiles := make([]Granite4VisionTile, len(images))
	for index, current := range images {
		addNewline := len(images) == 1 || index > 0
		tiles[index] = Granite4VisionTile{PixelValues: granite4TilePixels(current, spec), AddNewline: addNewline}
	}
	return Granite4VisionInput{Tiles: tiles, GridH: gridH, GridW: gridW}, nil
}

func granite4BestResolution(width, height int, spec Granite4VisionSpec) Granite4VisionResolution {
	candidates := spec.GridCandidates
	if len(candidates) == 0 {
		candidates = make([]Granite4VisionResolution, 0, 9)
		for gridW := 1; gridW <= 3; gridW++ {
			for gridH := 1; gridH <= 3; gridH++ {
				candidates = append(candidates, Granite4VisionResolution{Width: gridW * spec.ImageSize, Height: gridH * spec.ImageSize})
			}
		}
	}
	best := candidates[0]
	bestEffective, bestWaste := -1, math.MaxInt
	for _, candidate := range candidates {
		scale := math.Min(float64(candidate.Width)/float64(width), float64(candidate.Height)/float64(height))
		targetW, targetH := int(float64(width)*scale), int(float64(height)*scale)
		effective := min(targetW*targetH, width*height)
		waste := candidate.Width*candidate.Height - effective
		if effective > bestEffective || effective == bestEffective && waste < bestWaste {
			best, bestEffective, bestWaste = candidate, effective, waste
		}
	}
	return best
}

func granite4TilePixels(source image.Image, spec Granite4VisionSpec) []float32 {
	grid := spec.ImageSize / spec.PatchSize
	patchArea := spec.PatchSize * spec.PatchSize
	pixels := make([]float32, grid*grid*3*patchArea)
	for patchY := 0; patchY < grid; patchY++ {
		for patchX := 0; patchX < grid; patchX++ {
			row := (patchY*grid + patchX) * 3 * patchArea
			for channel := 0; channel < 3; channel++ {
				position := row + channel*patchArea
				for y := 0; y < spec.PatchSize; y++ {
					for x := 0; x < spec.PatchSize; x++ {
						r, g, b, _ := source.At(patchX*spec.PatchSize+x, patchY*spec.PatchSize+y).RGBA()
						value := [3]uint32{r, g, b}[channel]
						pixels[position] = (float32(value>>8)/255 - spec.ImageMean[channel]) / spec.ImageStd[channel]
						position++
					}
				}
			}
		}
	}
	return pixels
}

func (r *Granite4VisionRunner) EncodeImage(ctx context.Context, source image.Image) (Granite4VisionOutput, error) {
	if r == nil || r.file == nil {
		return Granite4VisionOutput{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessGranite4VisionImage(source, r.spec)
	if err != nil {
		return Granite4VisionOutput{}, err
	}
	streams := make([][]float32, len(r.spec.FeatureLayers))
	rows := 0
	for tileIndex, tile := range input.Tiles {
		values, tileErr := r.encodeTile(ctx, tile)
		if tileErr != nil {
			return Granite4VisionOutput{}, fmt.Errorf("projector: encode Granite 4 Vision tile %d: %w", tileIndex, tileErr)
		}
		for streamIndex, value := range values {
			streams[streamIndex] = append(streams[streamIndex], value.Data...)
		}
		rows += int(values[0].Shape.Dims[1])
	}
	base, err := reference.NewValue(tensor.MustShape(uint64(r.spec.ProjectionDim), uint64(rows)), streams[0])
	if err != nil {
		return Granite4VisionOutput{}, err
	}
	deepstack := make([]reference.Value, len(streams)-1)
	for index := range deepstack {
		deepstack[index], err = reference.NewValue(tensor.MustShape(uint64(r.spec.ProjectionDim), uint64(rows)), streams[index+1])
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
	rows, patchWidth := side*side, 3*r.spec.PatchSize*r.spec.PatchSize
	if len(tile.PixelValues) != rows*patchWidth {
		return nil, errors.New("projector: Granite 4 Vision tile shape is inconsistent")
	}
	patchWeight, err := r.load(ctx, "v.patch_embd.weight")
	if err != nil {
		return nil, err
	}
	patchBias, err := r.load(ctx, "v.patch_embd.bias")
	if err != nil {
		return nil, err
	}
	hidden := linear(tile.PixelValues, patchWeight.Data, patchBias.Data, rows, patchWidth, r.spec.Hidden)
	positions, err := r.load(ctx, "v.position_embd.weight")
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
		layerOutputs[layer] = append([]float32(nil), hidden...)
	}
	outputs := make([]reference.Value, len(r.spec.FeatureLayers))
	for block, layer := range r.spec.FeatureLayers {
		projected, projectErr := r.runQFormerBlock(ctx, layerOutputs[layer], side, block)
		if projectErr != nil {
			return nil, projectErr
		}
		outRows := len(projected) / r.spec.ProjectionDim
		if tile.AddNewline {
			newline, loadErr := r.load(ctx, "v.image_newline")
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
	qkv := make([]float32, rows*3*r.spec.Hidden)
	for partIndex, part := range []string{"q", "k", "v"} {
		weight, bias, loadErr := r.loadPair(ctx, prefix+"attn_"+part+".weight", prefix+"attn_"+part+".bias")
		if loadErr != nil {
			return loadErr
		}
		projected := linear(norm, weight.Data, bias.Data, rows, r.spec.Hidden, r.spec.Hidden)
		for row := 0; row < rows; row++ {
			copy(qkv[row*3*r.spec.Hidden+partIndex*r.spec.Hidden:], projected[row*r.spec.Hidden:(row+1)*r.spec.Hidden])
		}
	}
	attention := visionAttention(qkv, rows, r.spec.Hidden, r.spec.Heads)
	outWeight, outBias, err := r.loadPair(ctx, prefix+"attn_out.weight", prefix+"attn_out.bias")
	if err != nil {
		return err
	}
	projected := linear(attention, outWeight.Data, outBias.Data, rows, r.spec.Hidden, r.spec.Hidden)
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
	up := linear(norm, upWeight.Data, upBias.Data, rows, r.spec.Hidden, r.spec.Intermediate)
	for index, value := range up {
		up[index] = geluTanh(value)
	}
	downWeight, downBias, err := r.loadPair(ctx, prefix+"ffn_down.weight", prefix+"ffn_down.bias")
	if err != nil {
		return err
	}
	down := linear(up, downWeight.Data, downBias.Data, rows, r.spec.Intermediate, r.spec.Hidden)
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
	enc := granite4Window(x, side, windowSide, r.spec.Hidden)
	down := granite4Downsample(x, side, newSide, r.spec.Hidden, r.spec.SpatialOffsets[block])
	queryWindows := granite4Window(down, newSide, querySide, r.spec.Hidden)
	query, imagePosition, err := r.loadPair(ctx, prefix+"query", prefix+"img_pos")
	if err != nil {
		return nil, err
	}
	for window := 0; window < windows; window++ {
		for row := 0; row < queryLength; row++ {
			for channel := 0; channel < r.spec.Hidden; channel++ {
				queryWindows[(window*queryLength+row)*r.spec.Hidden+channel] += query.Data[row*r.spec.Hidden+channel]
			}
		}
		for row := 0; row < encLength; row++ {
			for channel := 0; channel < r.spec.Hidden; channel++ {
				enc[(window*encLength+row)*r.spec.Hidden+channel] += imagePosition.Data[row*r.spec.Hidden+channel]
			}
		}
	}
	queryWindows, err = r.affineNormalize(ctx, queryWindows, windows*queryLength, prefix+"post_norm", 1e-12)
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
	self, err = r.affineNormalize(ctx, self, windows*queryLength, prefix+"self_attn_norm", 1e-12)
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
	cross, err = r.affineNormalize(ctx, cross, windows*queryLength, prefix+"cross_attn_norm", 1e-12)
	if err != nil {
		return nil, err
	}
	upWeight, upBias, err := r.loadPair(ctx, prefix+"ffn_up.weight", prefix+"ffn_up.bias")
	if err != nil {
		return nil, err
	}
	up := linear(cross, upWeight.Data, upBias.Data, windows*queryLength, r.spec.Hidden, r.spec.QFormerWidth)
	for index, value := range up {
		up[index] = float32(0.5 * float64(value) * (1 + math.Erf(float64(value)/math.Sqrt2)))
	}
	downWeight, downBias, err := r.loadPair(ctx, prefix+"ffn_down.weight", prefix+"ffn_down.bias")
	if err != nil {
		return nil, err
	}
	ffn := linear(up, downWeight.Data, downBias.Data, windows*queryLength, r.spec.QFormerWidth, r.spec.Hidden)
	for index := range ffn {
		ffn[index] += cross[index]
	}
	ffn, err = r.affineNormalize(ctx, ffn, windows*queryLength, prefix+"ffn_norm", 1e-12)
	if err != nil {
		return nil, err
	}
	unwinned := granite4Unwindow(ffn, newSide, querySide, r.spec.Hidden)
	linearWeight, linearBias, err := r.loadPair(ctx, prefix+"linear.weight", prefix+"linear.bias")
	if err != nil {
		return nil, err
	}
	return linear(unwinned, linearWeight.Data, linearBias.Data, newSide*newSide, r.spec.Hidden, r.spec.ProjectionDim), nil
}

func (r *Granite4VisionRunner) qformerAttention(
	ctx context.Context,
	queryInput, keyValueInput []float32,
	windows, queryRows, keyRows int,
	prefix string,
) ([]float32, error) {
	parts := make([][]float32, 3)
	for index, name := range []string{"q", "k", "v"} {
		weight, bias, err := r.loadPair(ctx, prefix+"_"+name+".weight", prefix+"_"+name+".bias")
		if err != nil {
			return nil, err
		}
		input, rows := keyValueInput, windows*keyRows
		if index == 0 {
			input, rows = queryInput, windows*queryRows
		}
		parts[index] = linear(input, weight.Data, bias.Data, rows, r.spec.Hidden, r.spec.Hidden)
	}
	attention := granite4BatchedAttention(parts[0], parts[1], parts[2], windows, queryRows, keyRows, r.spec.Hidden, r.spec.Hidden/64)
	outWeight, outBias, err := r.loadPair(ctx, prefix+"_out.weight", prefix+"_out.bias")
	if err != nil {
		return nil, err
	}
	return linear(attention, outWeight.Data, outBias.Data, windows*queryRows, r.spec.Hidden, r.spec.Hidden), nil
}

func granite4BatchedAttention(q, k, v []float32, batches, queryRows, keyRows, hidden, heads int) []float32 {
	headWidth := hidden / heads
	output := make([]float32, batches*queryRows*hidden)
	scores := make([]float64, keyRows)
	scale := 1 / math.Sqrt(float64(headWidth))
	for batch := 0; batch < batches; batch++ {
		for query := 0; query < queryRows; query++ {
			for head := 0; head < heads; head++ {
				maximum := math.Inf(-1)
				for key := 0; key < keyRows; key++ {
					score := 0.0
					for channel := 0; channel < headWidth; channel++ {
						qIndex := ((batch*queryRows+query)*hidden + head*headWidth + channel)
						kIndex := ((batch*keyRows+key)*hidden + head*headWidth + channel)
						score += float64(q[qIndex] * k[kIndex])
					}
					scores[key] = score * scale
					maximum = math.Max(maximum, scores[key])
				}
				total := 0.0
				for key := range scores {
					scores[key] = math.Exp(scores[key] - maximum)
					total += scores[key]
				}
				for key := 0; key < keyRows; key++ {
					factor := float32(scores[key] / total)
					for channel := 0; channel < headWidth; channel++ {
						outIndex := ((batch*queryRows+query)*hidden + head*headWidth + channel)
						vIndex := ((batch*keyRows+key)*hidden + head*headWidth + channel)
						output[outIndex] += factor * v[vIndex]
					}
				}
			}
		}
	}
	return output
}

func granite4Window(input []float32, side, windowSide, hidden int) []float32 {
	windowsPerSide := side / windowSide
	output := make([]float32, len(input))
	destination := 0
	for windowY := 0; windowY < windowsPerSide; windowY++ {
		for windowX := 0; windowX < windowsPerSide; windowX++ {
			for y := 0; y < windowSide; y++ {
				for x := 0; x < windowSide; x++ {
					source := ((windowY*windowSide+y)*side + windowX*windowSide + x) * hidden
					copy(output[destination:destination+hidden], input[source:source+hidden])
					destination += hidden
				}
			}
		}
	}
	return output
}

func granite4Unwindow(input []float32, side, windowSide, hidden int) []float32 {
	windowsPerSide := side / windowSide
	output := make([]float32, len(input))
	source := 0
	for windowY := 0; windowY < windowsPerSide; windowY++ {
		for windowX := 0; windowX < windowsPerSide; windowX++ {
			for y := 0; y < windowSide; y++ {
				for x := 0; x < windowSide; x++ {
					destination := ((windowY*windowSide+y)*side + windowX*windowSide + x) * hidden
					copy(output[destination:destination+hidden], input[source:source+hidden])
					source += hidden
				}
			}
		}
	}
	return output
}

func granite4Downsample(input []float32, side, newSide, hidden, spatialOffset int) []float32 {
	output := make([]float32, newSide*newSide*hidden)
	if spatialOffset >= 0 {
		offsetY, offsetX := (spatialOffset>>1)&1, spatialOffset&1
		for y := 0; y < newSide; y++ {
			for x := 0; x < newSide; x++ {
				source := ((y*2+offsetY)*side + x*2 + offsetX) * hidden
				copy(output[(y*newSide+x)*hidden:], input[source:source+hidden])
			}
		}
		return output
	}
	kernel := side / newSide
	for y := 0; y < newSide; y++ {
		for x := 0; x < newSide; x++ {
			destination := (y*newSide + x) * hidden
			for ky := 0; ky < kernel; ky++ {
				for kx := 0; kx < kernel; kx++ {
					source := ((y*kernel+ky)*side + x*kernel + kx) * hidden
					for channel := 0; channel < hidden; channel++ {
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
	layerNorm(output, input, weight.Data, bias.Data, rows, len(input)/rows, epsilon)
	return output, nil
}

func (r *Granite4VisionRunner) load(ctx context.Context, name string) (reference.Value, error) {
	info, ok := r.file.Tensor(name)
	if !ok {
		return reference.Value{}, fmt.Errorf("projector: tensor %q is unavailable", name)
	}
	return model.LoadHostTensor(ctx, r.file, info)
}

func (r *Granite4VisionRunner) loadPair(ctx context.Context, first, second string) (reference.Value, reference.Value, error) {
	a, err := r.load(ctx, first)
	if err != nil {
		return reference.Value{}, reference.Value{}, err
	}
	b, err := r.load(ctx, second)
	return a, b, err
}
