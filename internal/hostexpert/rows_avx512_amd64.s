//go:build amd64

#include "textflag.h"

// The AVX-512 IQ3_S kernels, Strata's iq_avx512 scheme (src/kernels/cpu/iq_avx512.cpp): a block's 256 values in four
// chunks of 64, each decoded once - its 16 grid entries gathered with their high index bits added under a mask, its
// 64 sign bits a mask, its two sub-block scales a vector of words - and every token's activations taking the signs by
// a masked subtract, multiplied by the magnitudes in byte pairs and by the scales in word pairs.

// Z20 holds zeros and Z22 the dword 256 in every lane.
#define ZSETUP \
	VPXORD Z20, Z20, Z20; \
	MOVL $256, AX; \
	VPBROADCASTD AX, Z22

// ZCHUNK decodes chunk J of the IQ3_S block at SI into Z1 (its 64 magnitudes), K1 (their signs) and Z2 (its two
// sub-blocks' odd scales 2s+1, a word in each lane of the half each covers), using the grid at R12.
#define ZCHUNK(J) \
	VPMOVZXBD (2+16*(J))(SI), Z0; \
	MOVWLZX (66+2*(J))(SI), AX; \
	KMOVW AX, K2; \
	VPADDD Z22, Z0, K2, Z0; \
	KXNORW K3, K3, K3; \
	VPGATHERDD (R12)(Z0*4), K3, Z1; \
	KMOVQ (74+8*(J))(SI), K1; \
	MOVBLZX (106+(J))(SI), AX; \
	MOVL AX, BX; \
	ANDL $15, AX; \
	LEAL 1(AX)(AX*1), AX; \
	SHRL $4, BX; \
	LEAL 1(BX)(BX*1), BX; \
	VMOVD AX, X2; \
	VPBROADCASTW X2, Y2; \
	VMOVD BX, X3; \
	VPBROADCASTW X3, Y3; \
	VINSERTI64X4 $1, Y3, Z2, Z2

// ZTOK adds chunk J's product with the Q8_K block at ACT to the dwords ACC.
#define ZTOK(J, ACT, ACC) \
	VMOVDQU8 (4+64*(J))(ACT), Z4; \
	VPSUBB Z4, Z20, K1, Z4; \
	VPMADDUBSW Z4, Z1, Z4; \
	VPMADDWD Z2, Z4, Z4; \
	VPADDD Z4, ACC, ACC

#define ZCHUNK1(J) \
	ZCHUNK(J); \
	ZTOK(J, R8, Z12)

#define ZCHUNK4(J) \
	ZCHUNK(J); \
	ZTOK(J, R8, Z12); \
	ZTOK(J, R9, Z13); \
	ZTOK(J, R10, Z14); \
	ZTOK(J, R11, Z15)

// ZPREFETCH asks for the row's bytes 2,048 ahead of the block at SI, the next row's start included, as Strata's
// row_dot does (STRATA_IQ_PREFETCH's default): the rows stream from memory in 4 KB pages.
#define ZPREFETCH \
	PREFETCHT0 2048(SI); \
	PREFETCHT0 2112(SI)

// ZWEIGHT sets X5 to the block's f16 scale at SI as f32.
#define ZWEIGHT \
	MOVWLZX 0(SI), AX; \
	VMOVD AX, X5; \
	VCVTPH2PS X5, X5

// ZFOLD adds the dwords ACC, weighed by the block's scale (X5) times the Q8_K block's at ACT, to SUM.
#define ZFOLD(ACT, ACC, SUM) \
	VMULSS 0(ACT), X5, X6; \
	VBROADCASTSS X6, Z6; \
	VCVTDQ2PS ACC, ACC; \
	VFMADD231PS Z6, ACC, SUM

// ZHSUM sets the low float of XSUM to the sum of SUM's sixteen.
#define ZHSUM(SUM, YSUM, XSUM) \
	VEXTRACTF64X4 $1, SUM, Y7; \
	VADDPS Y7, YSUM, YSUM; \
	VEXTRACTF128 $1, YSUM, X7; \
	VADDPS X7, XSUM, XSUM; \
	VHADDPS XSUM, XSUM, XSUM; \
	VHADDPS XSUM, XSUM, XSUM

// func iq3SDots1Z(row *byte, blocks int, activation *byte) float32
TEXT ·iq3SDots1Z(SB), NOSPLIT, $0-28
	MOVQ row+0(FP), SI
	MOVQ blocks+8(FP), CX
	MOVQ activation+16(FP), R8
	MOVQ ·iq3SGrid(SB), R12
	ZSETUP
	VPXORD Z8, Z8, Z8
	TESTQ CX, CX
	JZ z31done

z31block:
	ZPREFETCH
	VPXORD Z12, Z12, Z12
	ZCHUNK1(0)
	ZCHUNK1(1)
	ZCHUNK1(2)
	ZCHUNK1(3)
	ZWEIGHT
	ZFOLD(R8, Z12, Z8)
	ADDQ $110, SI
	ADDQ $292, R8
	DECQ CX
	JNZ z31block

z31done:
	ZHSUM(Z8, Y8, X8)
	VMOVSS X8, ret+24(FP)
	VZEROUPPER
	RET

// func iq3SDots4Z(row *byte, blocks int, activations *[4]*byte, results *[4]float32)
TEXT ·iq3SDots4Z(SB), NOSPLIT, $0-32
	MOVQ row+0(FP), SI
	MOVQ blocks+8(FP), CX
	MOVQ activations+16(FP), DI
	MOVQ results+24(FP), DX
	MOVQ 0(DI), R8
	MOVQ 8(DI), R9
	MOVQ 16(DI), R10
	MOVQ 24(DI), R11
	MOVQ ·iq3SGrid(SB), R12
	ZSETUP
	VPXORD Z8, Z8, Z8
	VPXORD Z9, Z9, Z9
	VPXORD Z10, Z10, Z10
	VPXORD Z11, Z11, Z11
	TESTQ CX, CX
	JZ z34done

z34block:
	ZPREFETCH
	VPXORD Z12, Z12, Z12
	VPXORD Z13, Z13, Z13
	VPXORD Z14, Z14, Z14
	VPXORD Z15, Z15, Z15
	ZCHUNK4(0)
	ZCHUNK4(1)
	ZCHUNK4(2)
	ZCHUNK4(3)
	ZWEIGHT
	ZFOLD(R8, Z12, Z8)
	ZFOLD(R9, Z13, Z9)
	ZFOLD(R10, Z14, Z10)
	ZFOLD(R11, Z15, Z11)
	ADDQ $110, SI
	ADDQ $292, R8
	ADDQ $292, R9
	ADDQ $292, R10
	ADDQ $292, R11
	DECQ CX
	JNZ z34block

z34done:
	ZHSUM(Z8, Y8, X8)
	VMOVSS X8, 0(DX)
	ZHSUM(Z9, Y9, X9)
	VMOVSS X9, 4(DX)
	ZHSUM(Z10, Y10, X10)
	VMOVSS X10, 8(DX)
	ZHSUM(Z11, Y11, X11)
	VMOVSS X11, 12(DX)
	VZEROUPPER
	RET
