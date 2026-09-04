// Package speechrecognition composes common CPU acoustic encoder operations.
// Artifact names and numerical conventions enter through declarations, not
// model-family dispatch. Inference and frozen-base training share the forward.
package speechrecognition

// AffineBinding names a weight and optional bias in a tensor catalog.
type AffineBinding struct {
	Weight string `json:"weight"`
	Bias   string `json:"bias,omitzero"`
}

// FeedForwardBinding declares the two projections of a feed-forward operation.
type FeedForwardBinding struct {
	Norm AffineBinding `json:"norm"`
	In   AffineBinding `json:"in"`
	Out  AffineBinding `json:"out"`
}

// AttentionBinding declares block-local query-dependent relative attention.
type AttentionBinding struct {
	Norm        AffineBinding `json:"norm"`
	Query       AffineBinding `json:"query"`
	Key         AffineBinding `json:"key"`
	Value       AffineBinding `json:"value"`
	Out         AffineBinding `json:"out"`
	Relative    string        `json:"relative"`
	Heads       int           `json:"heads"`
	BlockFrames int           `json:"block_frames"`
}

// ConvolutionBinding declares GLU, centered depthwise convolution, frozen batch
// normalization and output projection. Stride also selects residual mean pooling.
type ConvolutionBinding struct {
	Norm      AffineBinding `json:"norm"`
	In        AffineBinding `json:"in"`
	Kernel    string        `json:"kernel"`
	BatchNorm AffineBinding `json:"batch_norm"`
	Mean      string        `json:"mean"`
	Variance  string        `json:"variance"`
	Out       AffineBinding `json:"out"`
	Stride    int           `json:"stride"`
}

// BlockBinding declares a macaron feed-forward/attention/convolution block.
type BlockBinding struct {
	First       FeedForwardBinding `json:"first"`
	Attention   AttentionBinding   `json:"attention"`
	Convolution ConvolutionBinding `json:"convolution"`
	Second      FeedForwardBinding `json:"second"`
	OutNorm     AffineBinding      `json:"out_norm"`
}

// Declaration supplies concrete composition and conventions absent from shapes.
// FeedbackAfter is a one-based block count; zero disables posterior feedback.
// The feedback classifier is always the final Output projection (shared weights).
type Declaration struct {
	Input            AffineBinding  `json:"input"`
	Blocks           []BlockBinding `json:"blocks"`
	Output           AffineBinding  `json:"output"`
	Feedback         AffineBinding  `json:"feedback"`
	FeedbackAfter    int            `json:"feedback_after"`
	Activation       string         `json:"activation"`
	FeedForwardScale float32        `json:"feed_forward_scale"`
	LayerNormEpsilon float64        `json:"layer_norm_epsilon"`
	BatchNormEpsilon float64        `json:"batch_norm_epsilon"`
}
