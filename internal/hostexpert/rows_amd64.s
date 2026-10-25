//go:build amd64

#include "textflag.h"

// SETUP loads the constants: Y15 the codebook in both lanes, X14 the low
// nibble mask, Y13 bytes of one.
#define SETUP \
	VBROADCASTI128 ·iq4Codebook(SB), Y15; \
	MOVL $0x0f0f0f0f, AX; \
	VMOVD AX, X14; \
	VPBROADCASTD X14, X14; \
	MOVL $0x01010101, AX; \
	VMOVD AX, X13; \
	VPBROADCASTD X13, Y13

// CODES looks up the 16 bytes of codes at ADDR, low nibbles the values 0 to
// 15 and high ones 16 to 31, into Y0 (their magnitudes) and Y1 (their signs).
#define CODES(ADDR) \
	VMOVDQU ADDR, X0; \
	VPSRLW $4, X0, X1; \
	VPAND X14, X0, X0; \
	VPAND X14, X1, X1; \
	VINSERTI128 $1, X1, Y0, Y0; \
	VPSHUFB Y0, Y15, Y0; \
	VPSIGNB Y0, Y13, Y1; \
	VPSIGNB Y0, Y0, Y0

// XSSCALE sets Y2's words to IQ4_XS sub-block H's 6-bit scale less 32: the
// low nibble or high one of low-scale byte H/2 and bits 2H, 2H+1 of the high
// scales.
#define XSSCALE(H) \
	MOVBLZX (4+(H)/2)(SI), AX; \
	SHRL $(4*((H)&1)), AX; \
	ANDL $15, AX; \
	MOVWLZX 2(SI), BX; \
	SHRL $(2*(H)), BX; \
	ANDL $3, BX; \
	SHLL $4, BX; \
	ORL BX, AX; \
	SUBL $32, AX; \
	VMOVD AX, X2; \
	VPBROADCASTW X2, Y2

// XSTOK adds sub-block H's product with the Q8_K block at ACT to ACC: the
// activations take the codes' signs, multiply their magnitudes in byte pairs
// and the sub-block's scale in word pairs.
#define XSTOK(H, ACT, ACC) \
	VMOVDQU (4+32*(H))(ACT), Y11; \
	VPSIGNB Y1, Y11, Y11; \
	VPMADDUBSW Y11, Y0, Y11; \
	VPMADDWD Y2, Y11, Y11; \
	VPADDD Y11, ACC, ACC

#define XSSUB4(H) \
	CODES((8+16*(H))(SI)); \
	XSSCALE(H); \
	XSTOK(H, R8, Y3); \
	XSTOK(H, R9, Y4); \
	XSTOK(H, R10, Y5); \
	XSTOK(H, R11, Y6)

#define XSSUB1(H) \
	CODES((8+16*(H))(SI)); \
	XSSCALE(H); \
	XSTOK(H, R8, Y3)

// WEIGHT sets X12 to the f16 scale at ADDR as f32.
#define WEIGHT(ADDR) \
	MOVWLZX ADDR, AX; \
	VMOVD AX, X12; \
	VCVTPH2PS X12, X12

// XSFOLD adds the integer sums ACC, weighed by the weight block's scale (X12)
// times the Q8_K block's at ACT, to SUM.
#define XSFOLD(ACT, ACC, SUM) \
	VMULSS 0(ACT), X12, X11; \
	VBROADCASTSS X11, Y11; \
	VCVTDQ2PS ACC, ACC; \
	VFMADD231PS Y11, ACC, SUM

// HSUM sets the low float of XSUM to the sum of SUM's eight.
#define HSUM(SUM, XSUM) \
	VEXTRACTF128 $1, SUM, X11; \
	VADDPS X11, XSUM, XSUM; \
	VHADDPS XSUM, XSUM, XSUM; \
	VHADDPS XSUM, XSUM, XSUM

// func iq4XSDots4(row *byte, blocks int, activations *[4]*byte, results *[4]float32)
TEXT ·iq4XSDots4(SB), NOSPLIT, $0-32
	MOVQ row+0(FP), SI
	MOVQ blocks+8(FP), CX
	MOVQ activations+16(FP), DI
	MOVQ results+24(FP), DX
	MOVQ 0(DI), R8
	MOVQ 8(DI), R9
	MOVQ 16(DI), R10
	MOVQ 24(DI), R11
	SETUP
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	TESTQ CX, CX
	JZ xs4done

xs4block:
	VPXOR Y3, Y3, Y3
	VPXOR Y4, Y4, Y4
	VPXOR Y5, Y5, Y5
	VPXOR Y6, Y6, Y6
	XSSUB4(0)
	XSSUB4(1)
	XSSUB4(2)
	XSSUB4(3)
	XSSUB4(4)
	XSSUB4(5)
	XSSUB4(6)
	XSSUB4(7)
	WEIGHT(0(SI))
	XSFOLD(R8, Y3, Y7)
	XSFOLD(R9, Y4, Y8)
	XSFOLD(R10, Y5, Y9)
	XSFOLD(R11, Y6, Y10)
	ADDQ $136, SI
	ADDQ $292, R8
	ADDQ $292, R9
	ADDQ $292, R10
	ADDQ $292, R11
	DECQ CX
	JNZ xs4block

xs4done:
	HSUM(Y7, X7)
	VMOVSS X7, 0(DX)
	HSUM(Y8, X8)
	VMOVSS X8, 4(DX)
	HSUM(Y9, X9)
	VMOVSS X9, 8(DX)
	HSUM(Y10, X10)
	VMOVSS X10, 12(DX)
	VZEROUPPER
	RET

// func iq4XSDots1(row *byte, blocks int, activation *byte) float32
TEXT ·iq4XSDots1(SB), NOSPLIT, $0-28
	MOVQ row+0(FP), SI
	MOVQ blocks+8(FP), CX
	MOVQ activation+16(FP), R8
	SETUP
	VXORPS Y7, Y7, Y7
	TESTQ CX, CX
	JZ xs1done

xs1block:
	VPXOR Y3, Y3, Y3
	XSSUB1(0)
	XSSUB1(1)
	XSSUB1(2)
	XSSUB1(3)
	XSSUB1(4)
	XSSUB1(5)
	XSSUB1(6)
	XSSUB1(7)
	WEIGHT(0(SI))
	XSFOLD(R8, Y3, Y7)
	ADDQ $136, SI
	ADDQ $292, R8
	DECQ CX
	JNZ xs1block

xs1done:
	HSUM(Y7, X7)
	VMOVSS X7, ret+24(FP)
	VZEROUPPER
	RET

// NLTOK adds the IQ4_NL block's product with the Q8_0 block at ACT to SUM:
// the integer sums of the 32 values, weighed by the weight block's scale
// (X12) times the activation block's.
#define NLTOK(ACT, SUM) \
	VMOVDQU 2(ACT), Y11; \
	VPSIGNB Y1, Y11, Y11; \
	VPMADDUBSW Y11, Y0, Y11; \
	VPMADDWD Y2, Y11, Y11; \
	VCVTDQ2PS Y11, Y11; \
	MOVWLZX 0(ACT), AX; \
	VMOVD AX, X3; \
	VCVTPH2PS X3, X3; \
	VMULSS X12, X3, X3; \
	VBROADCASTSS X3, Y3; \
	VFMADD231PS Y3, Y11, SUM

// NLSETUP adds Y2's words of one, which sum the byte-pair products in pairs.
#define NLSETUP \
	SETUP; \
	MOVL $0x00010001, AX; \
	VMOVD AX, X2; \
	VPBROADCASTD X2, Y2

// func iq4NLDots4(row *byte, blocks int, activations *[4]*byte, results *[4]float32)
TEXT ·iq4NLDots4(SB), NOSPLIT, $0-32
	MOVQ row+0(FP), SI
	MOVQ blocks+8(FP), CX
	MOVQ activations+16(FP), DI
	MOVQ results+24(FP), DX
	MOVQ 0(DI), R8
	MOVQ 8(DI), R9
	MOVQ 16(DI), R10
	MOVQ 24(DI), R11
	NLSETUP
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	TESTQ CX, CX
	JZ nl4done

nl4block:
	CODES(2(SI))
	WEIGHT(0(SI))
	NLTOK(R8, Y7)
	NLTOK(R9, Y8)
	NLTOK(R10, Y9)
	NLTOK(R11, Y10)
	ADDQ $18, SI
	ADDQ $34, R8
	ADDQ $34, R9
	ADDQ $34, R10
	ADDQ $34, R11
	DECQ CX
	JNZ nl4block

nl4done:
	HSUM(Y7, X7)
	VMOVSS X7, 0(DX)
	HSUM(Y8, X8)
	VMOVSS X8, 4(DX)
	HSUM(Y9, X9)
	VMOVSS X9, 8(DX)
	HSUM(Y10, X10)
	VMOVSS X10, 12(DX)
	VZEROUPPER
	RET

// func iq4NLDots1(row *byte, blocks int, activation *byte) float32
TEXT ·iq4NLDots1(SB), NOSPLIT, $0-28
	MOVQ row+0(FP), SI
	MOVQ blocks+8(FP), CX
	MOVQ activation+16(FP), R8
	NLSETUP
	VXORPS Y7, Y7, Y7
	TESTQ CX, CX
	JZ nl1done

nl1block:
	CODES(2(SI))
	WEIGHT(0(SI))
	NLTOK(R8, Y7)
	ADDQ $18, SI
	ADDQ $34, R8
	DECQ CX
	JNZ nl1block

nl1done:
	HSUM(Y7, X7)
	VMOVSS X7, ret+24(FP)
	VZEROUPPER
	RET

// func cpuid(leaf, subleaf uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL leaf+0(FP), AX
	MOVL subleaf+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() (eax, edx uint32)
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET

// The IQ3_S constants: each dword lane's shift to its index's high bit, a
// dword of one, the shuffle that spreads 4 sign bytes over 32 values (each
// byte over 8) and each value's sign bit in its byte.
DATA iq3sshifts<>+0(SB)/8, $0x0000000100000000
DATA iq3sshifts<>+8(SB)/8, $0x0000000300000002
DATA iq3sshifts<>+16(SB)/8, $0x0000000500000004
DATA iq3sshifts<>+24(SB)/8, $0x0000000700000006
GLOBL iq3sshifts<>(SB), RODATA|NOPTR, $32
DATA iq3sones<>+0(SB)/8, $0x0000000100000001
DATA iq3sones<>+8(SB)/8, $0x0000000100000001
DATA iq3sones<>+16(SB)/8, $0x0000000100000001
DATA iq3sones<>+24(SB)/8, $0x0000000100000001
GLOBL iq3sones<>(SB), RODATA|NOPTR, $32
DATA iq3sspread<>+0(SB)/8, $0x0000000000000000
DATA iq3sspread<>+8(SB)/8, $0x0101010101010101
DATA iq3sspread<>+16(SB)/8, $0x0202020202020202
DATA iq3sspread<>+24(SB)/8, $0x0303030303030303
GLOBL iq3sspread<>(SB), RODATA|NOPTR, $32
DATA iq3sbits<>+0(SB)/8, $0x8040201008040201
DATA iq3sbits<>+8(SB)/8, $0x8040201008040201
DATA iq3sbits<>+16(SB)/8, $0x8040201008040201
DATA iq3sbits<>+24(SB)/8, $0x8040201008040201
GLOBL iq3sbits<>(SB), RODATA|NOPTR, $32

// S3SUB decodes IQ3_S sub-block H of the block at SI into Y0 (its 32
// magnitudes, the grid entries of its eight 9-bit indices, gathered from the
// grid at R12), Y1 (their signs, a sign bit a value spread to bytes of -1 or
// one) and Y2 (its odd scale 2s+1, a word in every lane).
#define S3SUB(H) \
	VPMOVZXBD (2+8*(H))(SI), Y11; \
	MOVBLZX (66+(H))(SI), AX; \
	VMOVD AX, X12; \
	VPBROADCASTD X12, Y12; \
	VPSRLVD iq3sshifts<>(SB), Y12, Y12; \
	VPAND iq3sones<>(SB), Y12, Y12; \
	VPSLLD $8, Y12, Y12; \
	VPOR Y12, Y11, Y11; \
	VPCMPEQD Y12, Y12, Y12; \
	VPGATHERDD Y12, (R12)(Y11*4), Y0; \
	VPBROADCASTD (74+4*(H))(SI), Y1; \
	VPSHUFB iq3sspread<>(SB), Y1, Y1; \
	VPAND iq3sbits<>(SB), Y1, Y1; \
	VPCMPEQB iq3sbits<>(SB), Y1, Y1; \
	VPOR Y13, Y1, Y1; \
	MOVBLZX (106+(H)/2)(SI), AX; \
	SHRL $(4*((H)&1)), AX; \
	ANDL $15, AX; \
	LEAL 1(AX)(AX*1), AX; \
	VMOVD AX, X2; \
	VPBROADCASTW X2, Y2

#define S3SUB4(H) \
	S3SUB(H); \
	XSTOK(H, R8, Y3); \
	XSTOK(H, R9, Y4); \
	XSTOK(H, R10, Y5); \
	XSTOK(H, R11, Y6)

#define S3SUB1(H) \
	S3SUB(H); \
	XSTOK(H, R8, Y3)

// func iq3SDots4(row *byte, blocks int, activations *[4]*byte, results *[4]float32)
TEXT ·iq3SDots4(SB), NOSPLIT, $0-32
	MOVQ row+0(FP), SI
	MOVQ blocks+8(FP), CX
	MOVQ activations+16(FP), DI
	MOVQ results+24(FP), DX
	MOVQ 0(DI), R8
	MOVQ 8(DI), R9
	MOVQ 16(DI), R10
	MOVQ 24(DI), R11
	MOVQ ·iq3SGrid(SB), R12
	SETUP
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	TESTQ CX, CX
	JZ s34done

s34block:
	VPXOR Y3, Y3, Y3
	VPXOR Y4, Y4, Y4
	VPXOR Y5, Y5, Y5
	VPXOR Y6, Y6, Y6
	S3SUB4(0)
	S3SUB4(1)
	S3SUB4(2)
	S3SUB4(3)
	S3SUB4(4)
	S3SUB4(5)
	S3SUB4(6)
	S3SUB4(7)
	WEIGHT(0(SI))
	XSFOLD(R8, Y3, Y7)
	XSFOLD(R9, Y4, Y8)
	XSFOLD(R10, Y5, Y9)
	XSFOLD(R11, Y6, Y10)
	ADDQ $110, SI
	ADDQ $292, R8
	ADDQ $292, R9
	ADDQ $292, R10
	ADDQ $292, R11
	DECQ CX
	JNZ s34block

s34done:
	HSUM(Y7, X7)
	VMOVSS X7, 0(DX)
	HSUM(Y8, X8)
	VMOVSS X8, 4(DX)
	HSUM(Y9, X9)
	VMOVSS X9, 8(DX)
	HSUM(Y10, X10)
	VMOVSS X10, 12(DX)
	VZEROUPPER
	RET

// func iq3SDots1(row *byte, blocks int, activation *byte) float32
TEXT ·iq3SDots1(SB), NOSPLIT, $0-28
	MOVQ row+0(FP), SI
	MOVQ blocks+8(FP), CX
	MOVQ activation+16(FP), R8
	MOVQ ·iq3SGrid(SB), R12
	SETUP
	VXORPS Y7, Y7, Y7
	TESTQ CX, CX
	JZ s31done

s31block:
	VPXOR Y3, Y3, Y3
	S3SUB1(0)
	S3SUB1(1)
	S3SUB1(2)
	S3SUB1(3)
	S3SUB1(4)
	S3SUB1(5)
	S3SUB1(6)
	S3SUB1(7)
	WEIGHT(0(SI))
	XSFOLD(R8, Y3, Y7)
	ADDQ $110, SI
	ADDQ $292, R8
	DECQ CX
	JNZ s31block

s31done:
	HSUM(Y7, X7)
	VMOVSS X7, ret+24(FP)
	VZEROUPPER
	RET

// Q8CODES loads the Q8_0 block's 32 weights at SI into Y1 (themselves, whose
// signs the activations take) and Y0 (their magnitudes).
#define Q8CODES \
	VMOVDQU 2(SI), Y1; \
	VPSIGNB Y1, Y1, Y0

// func q80Dots4(row *byte, blocks int, activations *[4]*byte, results *[4]float32)
TEXT ·q80Dots4(SB), NOSPLIT, $0-32
	MOVQ row+0(FP), SI
	MOVQ blocks+8(FP), CX
	MOVQ activations+16(FP), DI
	MOVQ results+24(FP), DX
	MOVQ 0(DI), R8
	MOVQ 8(DI), R9
	MOVQ 16(DI), R10
	MOVQ 24(DI), R11
	NLSETUP
	VXORPS Y7, Y7, Y7
	VXORPS Y8, Y8, Y8
	VXORPS Y9, Y9, Y9
	VXORPS Y10, Y10, Y10
	TESTQ CX, CX
	JZ q84done

q84block:
	Q8CODES
	WEIGHT(0(SI))
	NLTOK(R8, Y7)
	NLTOK(R9, Y8)
	NLTOK(R10, Y9)
	NLTOK(R11, Y10)
	ADDQ $34, SI
	ADDQ $34, R8
	ADDQ $34, R9
	ADDQ $34, R10
	ADDQ $34, R11
	DECQ CX
	JNZ q84block

q84done:
	HSUM(Y7, X7)
	VMOVSS X7, 0(DX)
	HSUM(Y8, X8)
	VMOVSS X8, 4(DX)
	HSUM(Y9, X9)
	VMOVSS X9, 8(DX)
	HSUM(Y10, X10)
	VMOVSS X10, 12(DX)
	VZEROUPPER
	RET

// func q80Dots1(row *byte, blocks int, activation *byte) float32
TEXT ·q80Dots1(SB), NOSPLIT, $0-28
	MOVQ row+0(FP), SI
	MOVQ blocks+8(FP), CX
	MOVQ activation+16(FP), R8
	NLSETUP
	VXORPS Y7, Y7, Y7
	TESTQ CX, CX
	JZ q81done

q81block:
	Q8CODES
	WEIGHT(0(SI))
	NLTOK(R8, Y7)
	ADDQ $34, SI
	ADDQ $34, R8
	DECQ CX
	JNZ q81block

q81done:
	HSUM(Y7, X7)
	VMOVSS X7, ret+24(FP)
	VZEROUPPER
	RET
