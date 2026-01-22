; CompareANT - Compare two vectors for similarity
; Input: 2 * dim (concatenated), Output: 1 (similarity score)
; Dim: 32, Hidden: 16
;
; Architecture:
;   [vec_a, vec_b] -> MATMUL -> ReLU -> MATMUL -> similarity
;
; Total params: 16*64 + 16*16 + 1*16 = 1296 ternary weights

.registers
    H0: i32[64]      ; input (vec_a concat vec_b)
    H1: i32[16]      ; hidden
    H2: i32[16]      ; hidden2
    H3: i32[1]       ; output similarity

    C0: ternary[16, 64]  key="ant.compare.w_in"     ; input projection
    C1: ternary[16, 16]  key="ant.compare.w_hidden" ; hidden layer
    C2: ternary[1, 16]   key="ant.compare.w_out"    ; output projection

.program
    ; Input projection
    load_input H0
    ternary_matmul H1, C0, H0
    relu H1, H1

    ; Hidden processing
    ternary_matmul H2, C1, H1
    relu H2, H2

    ; Output (similarity score)
    ternary_matmul H3, C2, H2
    store_output H3
    halt
