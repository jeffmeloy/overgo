package gguf

import "fmt"

// general.file_type serialized values.
const (
	fileTypeAllF32         = 0
	fileTypeMostlyF16      = 1
	fileTypeMostlyQ4_0     = 2
	fileTypeMostlyQ4_1     = 3
	fileTypeMostlyQ8_0     = 7
	fileTypeMostlyQ5_0     = 8
	fileTypeMostlyQ5_1     = 9
	fileTypeMostlyQ2K      = 10
	fileTypeMostlyQ3KSmall = 11
	fileTypeMostlyQ3K      = 12
	fileTypeMostlyQ3KLarge = 13
	fileTypeMostlyQ4KSmall = 14
	fileTypeMostlyQ4K      = 15
	fileTypeMostlyQ5KSmall = 16
	fileTypeMostlyQ5K      = 17
	fileTypeMostlyQ6K      = 18
	fileTypeMostlyIQ2XXS   = 19
	fileTypeMostlyIQ2XS    = 20
	fileTypeMostlyQ2KSmall = 21
	fileTypeMostlyIQ3XS    = 22
	fileTypeMostlyIQ3XXS   = 23
	fileTypeMostlyIQ1S     = 24
	fileTypeMostlyIQ4NL    = 25
	fileTypeMostlyIQ3S     = 26
	fileTypeMostlyIQ3SMix  = 27
	fileTypeMostlyIQ2S     = 28
	fileTypeMostlyIQ2M     = 29
	fileTypeMostlyIQ4XS    = 30
	fileTypeMostlyIQ1M     = 31
	fileTypeMostlyBF16     = 32
	fileTypeMostlyTQ1_0    = 36
	fileTypeMostlyTQ2_0    = 37
	fileTypeMostlyMXFP4    = 38
	fileTypeMostlyNVFP4    = 39
	fileTypeMostlyQ1_0     = 40
	fileTypeMostlyQ2_0     = 41
)

var fileTypeNames = map[uint32]string{
	fileTypeAllF32:         "all F32",
	fileTypeMostlyF16:      "F16",
	fileTypeMostlyQ4_0:     "Q4_0",
	fileTypeMostlyQ4_1:     "Q4_1",
	fileTypeMostlyQ8_0:     "Q8_0",
	fileTypeMostlyQ5_0:     "Q5_0",
	fileTypeMostlyQ5_1:     "Q5_1",
	fileTypeMostlyQ2K:      "Q2_K - Medium",
	fileTypeMostlyQ3KSmall: "Q3_K - Small",
	fileTypeMostlyQ3K:      "Q3_K - Medium",
	fileTypeMostlyQ3KLarge: "Q3_K - Large",
	fileTypeMostlyQ4KSmall: "Q4_K - Small",
	fileTypeMostlyQ4K:      "Q4_K - Medium",
	fileTypeMostlyQ5KSmall: "Q5_K - Small",
	fileTypeMostlyQ5K:      "Q5_K - Medium",
	fileTypeMostlyQ6K:      "Q6_K",
	fileTypeMostlyIQ2XXS:   "IQ2_XXS - 2.0625 bpw",
	fileTypeMostlyIQ2XS:    "IQ2_XS - 2.3125 bpw",
	fileTypeMostlyQ2KSmall: "Q2_K - Small",
	fileTypeMostlyIQ3XS:    "IQ3_XS - 3.3 bpw",
	fileTypeMostlyIQ3XXS:   "IQ3_XXS - 3.0625 bpw",
	fileTypeMostlyIQ1S:     "IQ1_S - 1.5625 bpw",
	fileTypeMostlyIQ4NL:    "IQ4_NL - 4.5 bpw",
	fileTypeMostlyIQ3S:     "IQ3_S - 3.4375 bpw",
	fileTypeMostlyIQ3SMix:  "IQ3_S mix - 3.66 bpw",
	fileTypeMostlyIQ2S:     "IQ2_S - 2.5 bpw",
	fileTypeMostlyIQ2M:     "IQ2_M - 2.7 bpw",
	fileTypeMostlyIQ4XS:    "IQ4_XS - 4.25 bpw",
	fileTypeMostlyIQ1M:     "IQ1_M - 1.75 bpw",
	fileTypeMostlyBF16:     "BF16",
	fileTypeMostlyTQ1_0:    "TQ1_0 - 1.69 bpw ternary",
	fileTypeMostlyTQ2_0:    "TQ2_0 - 2.06 bpw ternary",
	fileTypeMostlyMXFP4:    "MXFP4 MoE",
	fileTypeMostlyNVFP4:    "NVFP4",
	fileTypeMostlyQ1_0:     "Q1_0",
	fileTypeMostlyQ2_0:     "Q2_0",
}

// FileTypeName: general.file_type display name.
func FileTypeName(value uint32) string {
	if name, ok := fileTypeNames[value]; ok {
		return name
	}
	return fmt.Sprintf("unknown (%d)", value)
}
