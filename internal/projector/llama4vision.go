package projector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const llama4ProjectorType = "llama4"

type llama4VisionActivation uint8

const (
	llama4QuickGELU llama4VisionActivation = iota
	llama4GELU
	llama4SiLU
)

type Llama4VisionSpec struct {
	ImageSize           int
	PatchSize           int
	Hidden              int
	Intermediate        int
	OutputHidden        int
	AdapterIntermediate int
	AdapterHidden       int
	Layers              int
	Heads               int
	MergeSize           int
	LayerNormEpsilon    float32
	RopeTheta           float32
	ImageMean           [3]float32
	ImageStd            [3]float32
	Activation          llama4VisionActivation
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
	file      *gguf.File
	spec      Llama4VisionSpec
	attention visionAttentionPlan
	cuda      *projectorCUDA
}

type Llama4VisionOpenOptions = OpenOptions

func OpenLlama4Vision(path string) (*Llama4VisionRunner, error) {
	return OpenLlama4VisionWithOptions(path, Llama4VisionOpenOptions{})
}

func OpenLlama4VisionWithOptions(path string, options Llama4VisionOpenOptions) (*Llama4VisionRunner, error) {
	return openCatalogProjector(path, options, "Llama-4", nil,
		ReadLlama4VisionSpec, validateLlama4VisionCatalog,
		func(file *gguf.File, spec Llama4VisionSpec, cuda *projectorCUDA) *Llama4VisionRunner {
			return &Llama4VisionRunner{file: file, spec: spec, attention: compileVisionAttention(spec.Hidden, spec.Heads), cuda: cuda}
		})
}

func (r *Llama4VisionRunner) Close() error {
	if r == nil {
		return nil
	}
	return closeProjectorResources(&r.file, &r.cuda)
}

func (r *Llama4VisionRunner) Spec() Llama4VisionSpec {
	if r == nil {
		return Llama4VisionSpec{}
	}
	return r.spec
}

func ReadLlama4VisionSpec(file *gguf.File) (Llama4VisionSpec, error) {
	if err := validateVisionProjector(file, "clip.projector_type", llama4ProjectorType); err != nil {
		return Llama4VisionSpec{}, err
	}
	spec := Llama4VisionSpec{}
	if err := readMetadataIntFields(file,
		metadataIntField{"clip.vision.image_size", &spec.ImageSize},
		metadataIntField{"clip.vision.patch_size", &spec.PatchSize},
		metadataIntField{"clip.vision.embedding_length", &spec.Hidden},
		metadataIntField{"clip.vision.feed_forward_length", &spec.Intermediate},
		metadataIntField{"clip.vision.projection_dim", &spec.OutputHidden},
		metadataIntField{"clip.vision.block_count", &spec.Layers},
		metadataIntField{"clip.vision.attention.head_count", &spec.Heads},
	); err != nil {
		return Llama4VisionSpec{}, err
	}
	merge, ok, err := optionalMetadataUint32(file, "clip.vision.projector.scale_factor")
	if err != nil {
		return Llama4VisionSpec{}, err
	}
	if !ok {
		merge = 2
	}
	epsilon, err := metadataFloat32(file, "clip.vision.attention.layer_norm_epsilon")
	if err != nil {
		return Llama4VisionSpec{}, err
	}
	mean, err := metadataFloat32Array(file, "clip.vision.image_mean", 3)
	if err != nil {
		return Llama4VisionSpec{}, err
	}
	std, err := metadataFloat32Array(file, "clip.vision.image_std", 3)
	if err != nil {
		return Llama4VisionSpec{}, err
	}
	mlp1, ok := file.Tensor("mm.model.mlp.1.weight")
	if !ok || mlp1.Dimensions != 2 {
		return Llama4VisionSpec{}, errors.New("projector: Llama-4 first adapter tensor is unavailable or invalid")
	}
	mlp2, ok := file.Tensor("mm.model.mlp.2.weight")
	if !ok || mlp2.Dimensions != 2 {
		return Llama4VisionSpec{}, errors.New("projector: Llama-4 second adapter tensor is unavailable or invalid")
	}
	spec.AdapterIntermediate = int(mlp1.Shape[1])
	spec.AdapterHidden = int(mlp2.Shape[1])
	spec.MergeSize = int(merge)
	spec.LayerNormEpsilon = epsilon
	spec.RopeTheta = 10000
	spec.PreLayerNorm = hasTensor(file, "v.pre_ln.weight")
	spec.PostLayerNorm = hasTensor(file, "v.post_ln.weight")
	spec.FusedQKV = make([]bool, spec.Layers)
	copy(spec.ImageMean[:], mean)
	copy(spec.ImageStd[:], std)
	for layer := range spec.FusedQKV {
		spec.FusedQKV[layer] = hasTensor(file, fmt.Sprintf("v.blk.%d.attn_qkv.weight", layer))
	}
	useGELU, err := optionalMetadataBool(file, "clip.use_gelu")
	if err != nil {
		return Llama4VisionSpec{}, err
	}
	useSiLU, err := optionalMetadataBool(file, "clip.use_silu")
	if err != nil {
		return Llama4VisionSpec{}, err
	}
	if useGELU && useSiLU {
		return Llama4VisionSpec{}, errors.New("projector: Llama-4 GELU and SiLU flags conflict")
	}
	if useGELU {
		spec.Activation = llama4GELU
	} else if useSiLU {
		spec.Activation = llama4SiLU
	}
	if err := spec.validate(); err != nil {
		return Llama4VisionSpec{}, err
	}
	return spec, nil
}

func (s Llama4VisionSpec) validate() error {
	headWidth := 0
	if s.Heads > 0 {
		headWidth = s.Hidden / s.Heads
	}
	if s.ImageSize <= 0 || s.PatchSize <= 0 || s.Hidden <= 0 || s.Intermediate <= 0 || s.OutputHidden <= 0 ||
		s.AdapterIntermediate <= 0 || s.AdapterHidden <= 0 || s.Layers <= 0 || s.Heads <= 0 || s.MergeSize <= 0 ||
		s.Hidden%s.Heads != 0 || headWidth%4 != 0 || s.ImageSize%s.PatchSize != 0 ||
		(s.ImageSize/s.PatchSize)%s.MergeSize != 0 || s.LayerNormEpsilon <= 0 || s.RopeTheta <= 0 || len(s.FusedQKV) != s.Layers {
		return fmt.Errorf("projector: invalid Llama-4 vision metadata: %+v", s)
	}
	for channel := range s.ImageStd {
		if s.ImageStd[channel] <= 0 || !finite32(s.ImageMean[channel]) || !finite32(s.ImageStd[channel]) {
			return fmt.Errorf("projector: invalid Llama-4 normalization channel %d", channel)
		}
	}
	return nil
}

func validateLlama4VisionCatalog(file *gguf.File, spec Llama4VisionSpec) ([]string, error) {
	patches := spec.ImageSize / spec.PatchSize
	shuffleWidth := spec.Hidden * spec.MergeSize * spec.MergeSize
	required := map[string][]uint64{
		"v.patch_embd.weight":    {uint64(spec.PatchSize), uint64(spec.PatchSize), 3, uint64(spec.Hidden)},
		"v.class_embd":           {uint64(spec.Hidden)},
		"v.position_embd.weight": {uint64(spec.Hidden), uint64(patches*patches + 1)},
		"mm.model.mlp.1.weight":  {uint64(shuffleWidth), uint64(spec.AdapterIntermediate)},
		"mm.model.mlp.2.weight":  {uint64(spec.AdapterIntermediate), uint64(spec.AdapterHidden)},
		"mm.model.fc.weight":     {uint64(spec.AdapterHidden), uint64(spec.OutputHidden)},
	}
	addOptionalProjectorTensor(file, required, "v.patch_embd.bias", []uint64{uint64(spec.Hidden)})
	for _, prefix := range []string{"v.pre_ln", "v.post_ln"} {
		if err := addOptionalProjectorPair(
			file, required, prefix+".weight", prefix+".bias", []uint64{uint64(spec.Hidden)},
		); err != nil {
			return nil, err
		}
	}
	addStandardVisionLayerCatalog(file, required, spec.Layers, spec.Hidden, spec.Intermediate, spec.FusedQKV)
	return validateProjectorTensorCatalog(file, required)
}

func PreprocessLlama4VisionImage(source image.Image, spec Llama4VisionSpec) (Llama4VisionInput, error) {
	if source == nil {
		return Llama4VisionInput{}, errors.New("projector: image is nil")
	}
	if err := spec.validate(); err != nil {
		return Llama4VisionInput{}, err
	}
	bounds := source.Bounds()
	if bounds.Dx() <= 0 || bounds.Dy() <= 0 {
		return Llama4VisionInput{}, errors.New("projector: image bounds are empty")
	}
	images := make([]image.Image, 0, 10)
	gridW, gridH := 0, 0
	if bounds.Dx() > spec.ImageSize || bounds.Dy() > spec.ImageSize {
		gridW, gridH = llama4BestGrid(bounds.Dx(), bounds.Dy(), spec.ImageSize)
		refined := resizeFitBicubic(source, gridW*spec.ImageSize, gridH*spec.ImageSize)
		for y := 0; y < gridH; y++ {
			for x := 0; x < gridW; x++ {
				images = append(images, cropImage(refined, image.Rect(x*spec.ImageSize, y*spec.ImageSize, (x+1)*spec.ImageSize, (y+1)*spec.ImageSize)))
			}
		}
	}
	images = append(images, resizeFitBicubic(source, spec.ImageSize, spec.ImageSize))
	tiles := make([]Llama4VisionTile, len(images))
	for index, current := range images {
		tiles[index] = llama4TilePixels(current, spec)
	}
	return Llama4VisionInput{Tiles: tiles, GridH: gridH, GridW: gridW}, nil
}

func llama4BestGrid(width, height, size int) (int, int) {
	bestW, bestH := 1, 2
	bestEffective, bestWaste := -1, math.MaxInt
	for gridW := 1; gridW <= 3; gridW++ {
		for gridH := 1; gridH <= 3; gridH++ {
			if gridW == 1 && gridH == 1 {
				continue
			}
			candidateW, candidateH := gridW*size, gridH*size
			scale := math.Min(float64(candidateW)/float64(width), float64(candidateH)/float64(height))
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

func resizeFitBicubic(source image.Image, width, height int) image.Image {
	bounds := source.Bounds()
	scale := math.Min(float64(width)/float64(bounds.Dx()), float64(height)/float64(bounds.Dy()))
	resizedW := max(1, min(width, int(math.Ceil(float64(bounds.Dx())*scale))))
	resizedH := max(1, min(height, int(math.Ceil(float64(bounds.Dy())*scale))))
	resized := resizeImageBicubic(source, resizedW, resizedH)
	output := image.NewRGBA(image.Rect(0, 0, width, height))
	offsetX, offsetY := (width-resizedW)/2, (height-resizedH)/2
	for y := 0; y < resizedH; y++ {
		for x := 0; x < resizedW; x++ {
			output.Set(offsetX+x, offsetY+y, resized.At(x, y))
		}
	}
	return output
}

func cropImage(source image.Image, rectangle image.Rectangle) image.Image {
	output := image.NewRGBA(image.Rect(0, 0, rectangle.Dx(), rectangle.Dy()))
	for y := 0; y < rectangle.Dy(); y++ {
		for x := 0; x < rectangle.Dx(); x++ {
			output.Set(x, y, source.At(rectangle.Min.X+x, rectangle.Min.Y+y))
		}
	}
	return output
}

func llama4TilePixels(source image.Image, spec Llama4VisionSpec) Llama4VisionTile {
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
						pixels[position] = (normalizedImageChannel(value) - spec.ImageMean[channel]) / spec.ImageStd[channel]
						position++
					}
				}
			}
		}
	}
	return Llama4VisionTile{PixelValues: pixels, GridH: grid, GridW: grid}
}

func (r *Llama4VisionRunner) EncodeImage(ctx context.Context, source image.Image) (Llama4VisionOutput, error) {
	if r == nil || r.file == nil {
		return Llama4VisionOutput{}, errors.New("projector: runner is closed")
	}
	input, err := PreprocessLlama4VisionImage(source, r.spec)
	if err != nil {
		return Llama4VisionOutput{}, err
	}
	var embeddings []float32
	rows := 0
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
		rows += int(value.Shape.Dims[1])
	}
	value, err := reference.NewValue(tensor.MustShape(uint64(r.spec.OutputHidden), uint64(rows)), embeddings)
	if err != nil {
		return Llama4VisionOutput{}, err
	}
	return Llama4VisionOutput{Embeddings: value, TileCount: len(input.Tiles), GridH: input.GridH, GridW: input.GridW}, nil
}

func (r *Llama4VisionRunner) encodeTile(ctx context.Context, input Llama4VisionTile) (reference.Value, error) {
	patchRows := input.GridH * input.GridW
	patchWidth := 3 * r.spec.PatchSize * r.spec.PatchSize
	if patchRows <= 0 || input.GridH != input.GridW || input.GridH%r.spec.MergeSize != 0 || len(input.PixelValues) != patchRows*patchWidth {
		return reference.Value{}, errors.New("projector: Llama-4 tile shape is inconsistent")
	}
	patchWeight, err := r.load(ctx, "v.patch_embd.weight")
	if err != nil {
		return reference.Value{}, err
	}
	patchBias, err := r.optionalBias(ctx, "v.patch_embd.bias")
	if err != nil {
		return reference.Value{}, err
	}
	hidden := linear(input.PixelValues, patchWeight.Data, patchBias, patchRows, patchWidth, r.spec.Hidden)
	classEmbedding, err := r.load(ctx, "v.class_embd")
	if err != nil {
		return reference.Value{}, err
	}
	hidden = append(hidden, classEmbedding.Data...)
	rows := patchRows + 1
	positions, err := r.load(ctx, "v.position_embd.weight")
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
	for layer := 0; layer < r.spec.Layers; layer++ {
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
	adapted := linear(shuffled, mlp1.Data, nil, mergePlan.outputRows, shuffleWidth, r.spec.AdapterIntermediate)
	for index, value := range adapted {
		adapted[index] = geluTanh(value)
	}
	mlp2, err := r.load(ctx, "mm.model.mlp.2.weight")
	if err != nil {
		return reference.Value{}, err
	}
	adapted = linear(adapted, mlp2.Data, nil, mergePlan.outputRows, r.spec.AdapterIntermediate, r.spec.AdapterHidden)
	for index, value := range adapted {
		adapted[index] = geluTanh(value)
	}
	projection, err := r.load(ctx, "mm.model.fc.weight")
	if err != nil {
		return reference.Value{}, err
	}
	output := linear(adapted, projection.Data, nil, mergePlan.outputRows, r.spec.AdapterHidden, r.spec.OutputHidden)
	return reference.NewValue(tensor.MustShape(uint64(r.spec.OutputHidden), uint64(mergePlan.outputRows)), output)
}

func (r *Llama4VisionRunner) runLayer(ctx context.Context, hidden []float32, gridH, gridW, layer int) error {
	rows := gridH*gridW + 1
	prefix := fmt.Sprintf("v.blk.%d.", layer)
	norm, err := r.affineNormalize(ctx, hidden, rows, prefix+"ln1")
	if err != nil {
		return err
	}
	qkv, err := r.projectQKV(ctx, norm, rows, prefix, layer)
	if err != nil {
		return err
	}
	llama4VisionRoPE(qkv, gridH, gridW, r.spec.Hidden, r.spec.Heads, r.spec.RopeTheta)
	attention := r.attention.cpu(qkv, rows)
	outWeight, err := r.load(ctx, prefix+"attn_out.weight")
	if err != nil {
		return err
	}
	outBias, err := r.optionalBias(ctx, prefix+"attn_out.bias")
	if err != nil {
		return err
	}
	projected := linear(attention, outWeight.Data, outBias, rows, r.spec.Hidden, r.spec.Hidden)
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
	up := linear(norm, upWeight.Data, upBias, rows, r.spec.Hidden, r.spec.Intermediate)
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
	down := linear(up, downWeight.Data, downBias, rows, r.spec.Intermediate, r.spec.Hidden)
	for index := range hidden {
		hidden[index] += down[index]
	}
	return nil
}

func llama4VisionRoPE(qkv []float32, gridH, gridW, hidden, heads int, theta float32) {
	rows, headWidth := gridH*gridW+1, hidden/heads
	half := headWidth / 2
	for token := 0; token < rows; token++ {
		positionW, positionH := 0, 0
		if token < rows-1 {
			positionW, positionH = token%gridW+1, token/gridW+1
		}
		for head := 0; head < heads; head++ {
			for _, part := range []int{0, 1} {
				base := token*3*hidden + part*hidden + head*headWidth
				for section, position := range []int{positionW, positionH} {
					sectionBase := base + section*half
					for pair := 0; pair < half/2; pair++ {
						frequency := math.Pow(float64(theta), -2*float64(pair)/float64(half))
						angle := float64(position) * frequency
						cosine, sine := float32(math.Cos(angle)), float32(math.Sin(angle))
						index := sectionBase + pair*2
						left, right := qkv[index], qkv[index+1]
						qkv[index], qkv[index+1] = left*cosine-right*sine, left*sine+right*cosine
					}
				}
			}
		}
	}
}

func (r *Llama4VisionRunner) activate(value float32) float32 {
	switch r.spec.Activation {
	case llama4GELU:
		return geluTanh(value)
	case llama4SiLU:
		return value / (1 + float32(math.Exp(float64(-value))))
	default:
		return value / (1 + float32(math.Exp(float64(-1.702*value))))
	}
}

func (r *Llama4VisionRunner) affineNormalize(ctx context.Context, input []float32, rows int, prefix string) ([]float32, error) {
	weight, bias, err := r.loadPair(ctx, prefix+".weight", prefix+".bias")
	if err != nil {
		return nil, err
	}
	output := make([]float32, len(input))
	layerNorm(output, input, weight.Data, bias.Data, rows, len(input)/rows, r.spec.LayerNormEpsilon)
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
		return linear(input, weight.Data, bias, rows, r.spec.Hidden, 3*r.spec.Hidden), nil
	}
	qkv := make([]float32, rows*3*r.spec.Hidden)
	for partIndex, part := range []string{"q", "k", "v"} {
		weight, err := r.load(ctx, prefix+"attn_"+part+".weight")
		if err != nil {
			return nil, err
		}
		bias, err := r.optionalBias(ctx, prefix+"attn_"+part+".bias")
		if err != nil {
			return nil, err
		}
		projected := linear(input, weight.Data, bias, rows, r.spec.Hidden, r.spec.Hidden)
		for row := 0; row < rows; row++ {
			copy(qkv[row*3*r.spec.Hidden+partIndex*r.spec.Hidden:], projected[row*r.spec.Hidden:(row+1)*r.spec.Hidden])
		}
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

func (r *Llama4VisionRunner) load(ctx context.Context, name string) (reference.Value, error) {
	return loadProjectorHostTensor(ctx, r.file, name)
}

func (r *Llama4VisionRunner) loadPair(ctx context.Context, first, second string) (reference.Value, reference.Value, error) {
	return loadProjectorHostTensorPair(ctx, r.file, first, second)
}
