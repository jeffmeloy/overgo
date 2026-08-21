package model

// MetadataReadPolicy: architecture-specific metadata decoding facts.
type MetadataReadPolicy uint16

const (
	MetadataReadJaisScale MetadataReadPolicy = 1 << iota
	MetadataReadGPTJRotary
	MetadataReadCommandRLogits
	MetadataReadBaichuanBlocks
	MetadataReadVisualSections
	MetadataReadQwen3VLDeepstack
	MetadataReadSmolLM3NoRoPE
	MetadataReadOLMoClamp
	MetadataReadGLMDSAGating
	MetadataReadALiBi
	MetadataReadZeroALiBiDefault
)

const allMetadataReadPolicies = (MetadataReadZeroALiBiDefault << 1) - 1

func (p ArchitectureProfile) readsMetadata(policy MetadataReadPolicy) bool {
	return p.MetadataRead&policy == policy
}
