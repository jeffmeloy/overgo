package model

// MetadataReadPolicy: architecture-specific metadata decoding facts.
type MetadataReadPolicy uint16

const (
	metadataReadJaisScale MetadataReadPolicy = 1 << iota
	metadataReadGPTJRotary
	metadataReadCommandRLogits
	metadataReadBaichuanBlocks
	metadataReadVisualSections
	metadataReadQwen3VLDeepstack
	metadataReadSmolLM3NoRoPE
	metadataReadOLMoClamp
	metadataReadGLMDSAGating
	metadataReadALiBi
	metadataReadZeroALiBiDefault
)

const allMetadataReadPolicies = (metadataReadZeroALiBiDefault << 1) - 1

func (p ArchitectureProfile) readsMetadata(policy MetadataReadPolicy) bool {
	return p.MetadataRead&policy == policy
}
