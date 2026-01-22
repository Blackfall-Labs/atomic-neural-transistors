; DiffANT - Compute learned difference between two vectors
; Input: 2 * dim (concatenated), Output: dim (difference vector)
; Dim: 32, Hidden: 24
;
; Architecture:
;   [vec_a, vec_b] -> MATMUL -> ReLU -> MATMUL -> difference
;
; Total params: 24*64 + 32*24 = 2304 ternary weights

.registers
    H0: i32[64]      ; input (vec_a concat vec_b)
    H1: i32[24]      ; hidden
    H2: i32[32]      ; output difference

    C0: ternary[24, 64]  key="ant.diff.w_in"   ; input projection
    C1: ternary[32, 24]  key="ant.diff.w_out"  ; output projection

.program
    ; Input projection
    load_input H0
    ternary_matmul H1, C0, H0
    relu H1, H1

    ; Output (difference vector)
    ternary_matmul H2, C1, H1
    store_output H2
    halt
