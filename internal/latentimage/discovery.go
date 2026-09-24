package latentimage

import ()

// ServingStatus: how far a recognized media artifact is toward being served.
type ServingStatus string

const (
	// StatusRecognized: arch recognized + geometry shape-derived + structurally
	// verified against the checkpoint; denoise/VAE-decode not yet ported. This
	// is the "servable-in-progress" state (serving-only, no training).
	StatusRecognized ServingStatus = "recognized (serving-in-progress)"
)

// MediaModel: one enumerated media artifact -- the discovery-facing view. The
// family tag lives here (discovery/recipe scope), the Spec's types stay
// functional. Kind is fixed to image diffusion for this pipeline.
type MediaModel struct {
	Dir     string
	Family  string
	Kind    string // "image-diffusion"
	Status  ServingStatus
	Serving bool // false: recognition only, no serving path yet
	Spec    *Spec
}

// mediaKind: the modality this executor serves.
const mediaKind = "image-diffusion"
