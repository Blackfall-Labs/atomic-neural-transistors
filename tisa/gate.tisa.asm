; GateANT - Apply learned gating to a signal
; Input: 2 * dim (signal, context), Output: dim (gated signal)
; Dim: 32, Hidden: 16
;
; Architecture:
;   [signal, context] -> MATMUL -> sigmoid -> gate
;   output = signal * gate
;
; Total params: 16*64 + 32*16 = 1536 ternary weights

.registers
    H0: i32[64]      ; input (signal concat context)
    H1: i32[16]      ; hidden
    H2: i32[32]      ; gate values (sigmoid output)
    H3: i32[32]      ; output (gated signal)
    H4: i32[32]      ; original signal (extracted)

    C0: ternary[16, 64]  key="ant.gate.w_in"   ; input projection
    C1: ternary[32, 16]  key="ant.gate.w_out"  ; gate projection

.program
    ; Load input and extract signal portion
    load_input H0

    ; Compute gate values
    ternary_matmul H1, C0, H0
    relu H1, H1
    ternary_matmul H2, C1, H1
    sigmoid H2, H2

    ; Extract original signal (first 32 elements)
    ; Note: In practice this would need a slice operation
    ; For now, we use a second matmul with identity-like weights
    ; Alternative: Use the signal directly from H0[0:32]
    copy_reg H4, H0

    ; Apply gating: output = signal * gate
    mul H3, H4, H2
    shift H3, H3, 8    ; Normalize after multiply

    store_output H3
    halt
