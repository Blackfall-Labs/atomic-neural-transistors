; MergeANT - Merge multiple signals into one
; Input: n * dim (concatenated signals), Output: dim
; n: 2, Dim: 32, Hidden: 24
;
; Architecture:
;   [sig1, sig2, ...] -> MATMUL -> ReLU -> MATMUL -> merged
;
; Total params: 24*64 + 32*24 = 2304 ternary weights

.registers
    H0: i32[64]      ; input (n signals concatenated)
    H1: i32[24]      ; hidden
    H2: i32[32]      ; output (merged signal)

    C0: ternary[24, 64]  key="ant.merge.w_in"   ; input projection
    C1: ternary[32, 24]  key="ant.merge.w_out"  ; output projection

.program
    ; Input projection (learns optimal combination)
    load_input H0
    ternary_matmul H1, C0, H0
    relu H1, H1

    ; Output projection
    ternary_matmul H2, C1, H1
    store_output H2
    halt
