package diffusionimage

import (
	"fmt"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestABatchIsMixedAsTheAuthorsTransformerBlockMixesIt holds the card graph's transformer blocks to the author's
// network over a batch (readme-samples/simplediffusion-fifty): models/uvit.py's TransformerBlock writes its tokens back
// by x.permute(1, 2, 0).reshape(b, c, h, w), which interleaves the samples, and the weights were trained so. The host
// forward reproduces that (attnBlock.forward, tokensToImageNCHW); the graph built for three images at once, run on the
// reference backend, must give each image what the host gives it.
func TestABatchIsMixedAsTheAuthorsTransformerBlockMixesIt(t *testing.T) {
	t.Parallel()
	m := compileTiny(t, loadGolden[uditTinyFixture](t, "udit_tiny"))
	if len(m.middle) == 0 {
		t.Fatal("the tiny model has no transformer block in its middle")
	}
	block := m.middle[0]
	const batch, height, width = 3, 2, 2
	channels := len(block.norm1Weight)
	plane := channels * height * width
	images := make([]float32, batch*plane)
	for index := range images {
		images[index] = float32((index*37)%23-11) / 16
	}
	want := block.forward(m, images, batch, channels, height, width)

	graph := newModelGraph()
	inputs := make([]*tensor.Tensor, batch)
	feeds := map[*tensor.Tensor]reference.Value{}
	for index := range inputs {
		inputs[index] = graph.builder.Input(fmt.Sprintf("image.%d", index), dtype.F32, tensor.MustShape(uint64(channels), width, height))
		feeds[inputs[index]] = reference.Value{Shape: inputs[index].Shape, Data: nchwToGraphImage(images[index*plane:(index+1)*plane], channels, height, width)}
	}
	outputs := graph.transformerImages(block, m.Cfg, inputs, channels, height, width)
	testcheck.NoError(t, graph.err("batch mixing"))
	for _, binding := range graph.static {
		feeds[binding.Node] = binding.Value
	}
	results := testcheck.Must(reference.Execute(outputs, feeds))(t)
	var got []float32
	for _, output := range outputs {
		got = append(got, graphImageToNCHW(results[output].Data, channels, height, width)...)
	}
	requireWithin(t, "three images through one transformer block", got, want, 1e-4)
}

// TestASheetLaysTheBatchOutAsTheAuthorSavesIt holds a sheet answer to torchvision's save_image with nrow=4 and
// padding=2, as train.py saves its samples: four pictures to a row, two black pixels between and around them, so 16
// pictures of 256 x 256 make the author's 1034 x 1034 sheet; without a sheet the answer is the batch's first picture.
func TestASheetLaysTheBatchOutAsTheAuthorSavesIt(t *testing.T) {
	t.Parallel()
	const channels, side = 3, 4
	plane := channels * side * side
	const batch = sampleBatchMost
	pixels := make([]float32, batch*plane)
	for picture := range batch {
		for index := range plane {
			pixels[picture*plane+index] = float32(picture) / 16
		}
	}
	first, height, width := sampleFeatures{pixels: pixels, height: side, width: side, batch: batch}.picture()
	if height != side || width != side || len(first) != plane || first[plane-1] != 0 {
		t.Errorf("without a sheet the answer is %d x %d of %d values; want the first picture, %d x %d", width, height, len(first), side, side)
	}
	sheet, height, width := sampleFeatures{pixels: pixels, height: side, width: side, batch: batch, sheet: true}.picture()
	if want := 4*side + 5*sheetPadding; height != want || width != want {
		t.Fatalf("the sheet is %d x %d; want %d x %d, as save_image lays 16 out four to a row", width, height, want, want)
	}
	at := func(channel, row, column int) float32 { return sheet[(channel*height+row)*width+column] }
	step := side + sheetPadding
	for picture := range batch {
		top, left := sheetPadding+picture/sheetColumns*step, sheetPadding+picture%sheetColumns*step
		if got := at(channels-1, top+side-1, left+side-1); got != float32(picture)/16 {
			t.Errorf("picture %d's corner on the sheet holds %v; want its own %v", picture, got, float32(picture)/16)
		}
	}
	if at(0, 0, 0) != -1 || at(1, sheetPadding+side, sheetPadding) != -1 {
		t.Error("the sheet's padding is not black, -1 in the model's range")
	}
}

// TestARequestDrawsItsBatch holds a request's batch to one through the sixteen the author's sampler draws, one where it
// names none, as the frozen review cases were drawn: their session key stays the one a picture drawn alone had, and a
// batch of more draws on its own resident generator.
func TestARequestDrawsItsBatch(t *testing.T) {
	t.Parallel()
	base := Request{Seed: 2026, Steps: 50, Height: 256, Width: 256}
	for batch, valid := range map[int]bool{-1: false, 0: true, 1: true, 16: true, 17: false} {
		request := base
		request.Batch = batch
		if err := ValidateRequest(request); (err == nil) != valid {
			t.Errorf("a batch of %d validates as %v (%v), want valid %t", batch, err == nil, err, valid)
		}
	}
	alone := testcheck.Must(base.SessionKey())(t)
	base.Batch = 16
	together := testcheck.Must(base.SessionKey())(t)
	if alone != "f32:256x256" || together == alone || base.batch() != 16 {
		t.Errorf("a picture drawn alone keys %q and a batch of sixteen %q", alone, together)
	}
}
