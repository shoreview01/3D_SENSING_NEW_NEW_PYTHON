def print_results(iterations_1, iterations_2, iterations_3, iterations_4, elasped_init_1, elasped_init_2, elasped_init_3, elasped_init_4, elasped_loop_1, elasped_loop_2, elasped_loop_3, elasped_loop_4):
    print(f"Elasped time for GAMP1 : {(elasped_init_1+elasped_loop_1)*1000:.3f} ms")
    print(f"Elasped time for GAMP2 : {(elasped_init_2+elasped_loop_2)*1000:.3f} ms")
    print(f"Elasped time for Inverse1 : {(elasped_init_3+elasped_loop_3)*1000:.3f} ms")
    print(f"Elasped time for Inverse2 : {(elasped_init_4+elasped_loop_4)*1000:.3f} ms")
    print(f"Iteration for GAMP1 : {iterations_1:d}")
    print(f"Iteration for GAMP2 : {iterations_2:d}")
    print(f"Iteration for Inverse1 : {iterations_3:d}")
    print(f"Iteration for Inverse2 : {iterations_4:d}")